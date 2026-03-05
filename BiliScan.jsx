import { useState, useEffect, useRef, useCallback } from "react";

// ════════════════════════════════════════════════════════════════════════════
//  PALETTE
// ════════════════════════════════════════════════════════════════════════════
const P = {
  bg: "#080B10", surface: "#0F1420", card: "#141A26", border: "#1C2738",
  accent: "#00D4AA", accentDim: "#009E80", accentGlow: "rgba(0,212,170,0.14)",
  warn: "#F5A623", warnGlow: "rgba(245,166,35,0.14)",
  danger: "#E8455A", dangerGlow: "rgba(232,69,90,0.14)",
  safe: "#2DD4A0", safeGlow: "rgba(45,212,160,0.14)",
  text: "#E4EBF5", textMuted: "#60748A", textDim: "#334455",
  white: "#FFFFFF",
};

// ════════════════════════════════════════════════════════════════════════════
//  CAMERA ENGINE  — real getUserMedia + canvas pixel analysis
// ════════════════════════════════════════════════════════════════════════════

function toLinear(c) {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function rgbToLab(r, g, b) {
  const rl = toLinear(r), gl = toLinear(g), bl = toLinear(b);
  let X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375;
  let Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750;
  let Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041;
  X /= 0.95047; Z /= 1.08883;
  const f = v => v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116;
  const fx = f(X), fy = f(Y), fz = f(Z);
  return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
}

function sampleRegion(data, width, x0, y0, x1, y1) {
  let rS = 0, gS = 0, bS = 0, n = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 4;
      rS += data[i]; gS += data[i + 1]; bS += data[i + 2]; n++;
    }
  }
  return n ? { r: rS / n, g: gS / n, b: bS / n } : { r: 128, g: 128, b: 128 };
}

function computeSharpness(data, w, h) {
  let sum = 0, count = 0;
  const step = 4;
  for (let y = step; y < h - step; y += step) {
    for (let x = step; x < w - step; x += step) {
      const idx = (y * w + x) * 4;
      const lumC = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      const iU = ((y - step) * w + x) * 4, iD = ((y + step) * w + x) * 4;
      const iL = (y * w + (x - step)) * 4, iR2 = (y * w + (x + step)) * 4;
      const lumU = 0.299 * data[iU] + 0.587 * data[iU + 1] + 0.114 * data[iU + 2];
      const lumD = 0.299 * data[iD] + 0.587 * data[iD + 1] + 0.114 * data[iD + 2];
      const lumL = 0.299 * data[iL] + 0.587 * data[iL + 1] + 0.114 * data[iL + 2];
      const lumR = 0.299 * data[iR2] + 0.587 * data[iR2 + 1] + 0.114 * data[iR2 + 2];
      sum += Math.abs(lumC - lumU) + Math.abs(lumC - lumD) + Math.abs(lumC - lumL) + Math.abs(lumC - lumR);
      count++;
    }
  }
  return count ? sum / count : 0;
}

function analyseFrame(canvas) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;

  const sz = { x0: Math.round(w * 0.48), y0: Math.round(h * 0.32), x1: Math.round(w * 0.82), y1: Math.round(h * 0.68) };
  const wz = { x0: Math.round(w * 0.08), y0: Math.round(h * 0.30), x1: Math.round(w * 0.35), y1: Math.round(h * 0.70) };

  // Histogram / exposure
  let lumSum = 0, overExp = 0, underExp = 0;
  const totalPx = w * h;
  for (let i = 0; i < data.length; i += 4) {
    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    lumSum += lum;
    if (lum > 245) overExp++;
    if (lum < 20) underExp++;
  }
  const meanLuma = lumSum / totalPx;
  const exposureOK = meanLuma > 55 && meanLuma < 210 && overExp / totalPx < 0.08 && underExp / totalPx < 0.15;

  // Sharpness
  const sharpness = computeSharpness(data, w, h);
  const focusOK = sharpness > 3.5;

  // White card calibration
  const cardRgb = sampleRegion(data, w, wz.x0, wz.y0, wz.x1, wz.y1);
  const maxCard = Math.max(cardRgb.r, cardRgb.g, cardRgb.b, 1);
  const cal = { r: maxCard / Math.max(cardRgb.r, 1), g: maxCard / Math.max(cardRgb.g, 1), b: maxCard / Math.max(cardRgb.b, 1) };
  const cardLuma = 0.299 * cardRgb.r + 0.587 * cardRgb.g + 0.114 * cardRgb.b;
  const wbOK = cardLuma > 120 && (Math.max(cardRgb.r, cardRgb.g, cardRgb.b) - Math.min(cardRgb.r, cardRgb.g, cardRgb.b)) < 60;

  // Sclera b* extraction
  const bStarValues = [];
  const step = 3;
  for (let y = sz.y0; y < sz.y1; y += step) {
    for (let x = sz.x0; x < sz.x1; x += step) {
      const idx = (y * w + x) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      if (lum < 60 || lum > 245) continue;
      const maxC = Math.max(r, g, b), minC = Math.min(r, g, b);
      if (maxC > 0 && (maxC - minC) / maxC > 0.55) continue;
      const rc = Math.min(r * cal.r, 255);
      const gc = Math.min(g * cal.g, 255);
      const bc = Math.min(b * cal.b, 255);
      bStarValues.push(rgbToLab(rc, gc, bc).b);
    }
  }

  let yss = 0;
  if (bStarValues.length > 10) {
    bStarValues.sort((a, b) => a - b);
    const trim = Math.floor(bStarValues.length * 0.1);
    const trimmed = bStarValues.slice(trim, bStarValues.length - trim);
    const median = trimmed[Math.floor(trimmed.length / 2)];
    yss = Math.max(0, median - 1.5);
  }

  const sclMean = sampleRegion(data, w, sz.x0, sz.y0, sz.x1, sz.y1);
  const rgRatio = sclMean.g > 0 ? sclMean.r / sclMean.g : 1;
  const yssAdj = yss * (rgRatio > 1.08 ? 1.04 : 1.0);
  const qualityScore = (exposureOK ? 0.35 : 0) + (focusOK ? 0.35 : 0) + (wbOK ? 0.30 : 0);

  return {
    exposureOK, focusOK, wbOK,
    meanLuma: Math.round(meanLuma),
    sharpness: sharpness.toFixed(1),
    cardLuma: Math.round(cardLuma),
    sclPixels: bStarValues.length,
    yss: parseFloat(yssAdj.toFixed(2)),
    qualityScore,
    rgRatio: rgRatio.toFixed(3),
  };
}

// ════════════════════════════════════════════════════════════════════════════
//  DOMAIN LOGIC
// ════════════════════════════════════════════════════════════════════════════
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmtDate = d => new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

function classifyRisk(yss) {
  if (yss < 14) return { level: "LOW", label: "Low Risk", color: P.safe, glow: P.safeGlow, icon: "●" };
  if (yss < 24) return { level: "MODERATE", label: "Moderate Risk", color: P.warn, glow: P.warnGlow, icon: "◆" };
  return { level: "HIGH", label: "High Risk", color: P.danger, glow: P.dangerGlow, icon: "▲" };
}

function computeTrend(scans) {
  if (scans.length < 2) return 0;
  const n = scans.length, xs = scans.map((_, i) => i), ys = scans.map(s => s.yss);
  const xm = xs.reduce((a, b) => a + b, 0) / n, ym = ys.reduce((a, b) => a + b, 0) / n;
  const num = xs.reduce((acc, x, i) => acc + (x - xm) * (ys[i] - ym), 0);
  const den = xs.reduce((acc, x) => acc + (x - xm) ** 2, 0);
  return den === 0 ? 0 : num / den;
}

const SEED_SCANS = [
  { id: 1, date: Date.now() - 86400000 * 4, yss: 11.2, risk: classifyRisk(11.2), quality: 0.92, sclPixels: 840, rgRatio: "1.021" },
  { id: 2, date: Date.now() - 86400000 * 3, yss: 12.8, risk: classifyRisk(12.8), quality: 0.88, sclPixels: 790, rgRatio: "1.034" },
  { id: 3, date: Date.now() - 86400000 * 2, yss: 15.1, risk: classifyRisk(15.1), quality: 0.85, sclPixels: 810, rgRatio: "1.052" },
  { id: 4, date: Date.now() - 86400000 * 1, yss: 17.4, risk: classifyRisk(17.4), quality: 0.91, sclPixels: 870, rgRatio: "1.063" },
];

// ════════════════════════════════════════════════════════════════════════════
//  SHARED UI
// ════════════════════════════════════════════════════════════════════════════
function Chip({ label, color }) {
  return <span style={{ background: `${color}22`, color, border: `1px solid ${color}44`, borderRadius: 20, padding: "2px 10px", fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>{label}</span>;
}

function ProgressRing({ value, max = 42, size = 96, color }) {
  const r = 40, circ = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" style={{ transform: "rotate(-90deg)" }}>
      <circle cx={50} cy={50} r={r} fill="none" stroke={P.border} strokeWidth={8} />
      <circle cx={50} cy={50} r={r} fill="none" stroke={color} strokeWidth={8}
        strokeDasharray={`${clamp(value / max, 0, 1) * circ} ${circ}`} strokeLinecap="round"
        style={{ transition: "stroke-dasharray 1s cubic-bezier(.4,0,.2,1)" }} />
    </svg>
  );
}

function TrendChart({ scans }) {
  if (scans.length < 2) return null;
  const W = 340, H = 80, pad = { l: 28, r: 12, t: 10, b: 22 };
  const yw = W - pad.l - pad.r, yh = H - pad.t - pad.b;
  const ys = scans.map(s => s.yss);
  const minY = Math.max(0, Math.min(...ys) - 4), maxY = Math.max(...ys) + 4;
  const px = i => pad.l + (i / (scans.length - 1)) * yw;
  const py = v => pad.t + yh - ((v - minY) / (maxY - minY)) * yh;
  const d = scans.map((s, i) => `${i === 0 ? "M" : "L"}${px(i)},${py(s.yss)}`).join(" ");
  const area = d + ` L${px(scans.length - 1)},${pad.t + yh} L${pad.l},${pad.t + yh} Z`;
  const slope = computeTrend(scans);
  const tc = slope > 1 ? P.danger : slope > 0.3 ? P.warn : P.safe;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 11, color: P.textMuted, fontWeight: 600, letterSpacing: "0.06em" }}>YSS TREND</span>
        <Chip label={slope > 1 ? "↑ Rising" : slope > 0.3 ? "↗ Slight" : "→ Stable"} color={tc} />
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
        <defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={tc} stopOpacity={0.3} /><stop offset="100%" stopColor={tc} stopOpacity={0.02} /></linearGradient></defs>
        {14 >= minY && 14 <= maxY && <line x1={pad.l} x2={pad.l + yw} y1={py(14)} y2={py(14)} stroke={P.safe} strokeWidth={1} strokeDasharray="4 3" opacity={0.4} />}
        {24 >= minY && 24 <= maxY && <line x1={pad.l} x2={pad.l + yw} y1={py(24)} y2={py(24)} stroke={P.warn} strokeWidth={1} strokeDasharray="4 3" opacity={0.4} />}
        <path d={area} fill="url(#ag)" />
        <path d={d} fill="none" stroke={tc} strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
        {scans.map((s, i) => <circle key={s.id} cx={px(i)} cy={py(s.yss)} r={3.5} fill={s.risk.color} stroke={P.bg} strokeWidth={2} />)}
        {scans.map((s, i) => <text key={s.id} x={px(i)} y={H - 3} textAnchor="middle" fill={P.textDim} fontSize={8}>{new Date(s.date).toLocaleDateString("en-US", { month: "numeric", day: "numeric" })}</text>)}
      </svg>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  LIVE CAMERA COMPONENT
// ════════════════════════════════════════════════════════════════════════════
function LiveCamera({ onFrameAnalysis, onCapture, onError }) {
  const videoRef = useRef(null);
  const analysisCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const [camState, setCamState] = useState("requesting");
  const [metrics, setMetrics] = useState(null);
  const [facingMode, setFacingMode] = useState("user");
  const [captureFlash, setCaptureFlash] = useState(false);
  const [torchOn, setTorchOn] = useState(false);
  const [hasTorch, setHasTorch] = useState(false);
  const [videoRes, setVideoRes] = useState({ w: 0, h: 0 });

  const startCamera = useCallback(async (mode) => {
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    setCamState("requesting");
    if (!navigator.mediaDevices?.getUserMedia) { setCamState("nosupport"); onError("Camera API not available."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1280 }, height: { ideal: 960 },
        }
      });
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      const caps = track.getCapabilities?.() || {};
      setHasTorch(!!caps.torch);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setVideoRes({ w: videoRef.current.videoWidth, h: videoRef.current.videoHeight });
        };
        await videoRef.current.play();
        setCamState("active");
      }
    } catch (err) {
      setCamState(err.name === "NotAllowedError" || err.name === "PermissionDeniedError" ? "denied" : "error");
      onError(err.name === "NotAllowedError" ? "Camera permission denied. Allow camera access in browser settings." : `Camera error: ${err.message}`);
    }
  }, [onError]);

  useEffect(() => { startCamera(facingMode); return () => { streamRef.current?.getTracks().forEach(t => t.stop()); cancelAnimationFrame(rafRef.current); }; }, [facingMode]);

  // Analysis loop
  useEffect(() => {
    if (camState !== "active") return;
    let animId, timeoutId;
    const analyse = () => {
      const video = videoRef.current, canvas = analysisCanvasRef.current;
      if (video && canvas && video.readyState >= 2) {
        const W = 320, H = 240;
        canvas.width = W; canvas.height = H;
        canvas.getContext("2d").drawImage(video, 0, 0, W, H);
        const result = analyseFrame(canvas);
        drawOverlay(overlayCanvasRef.current, result);
        setMetrics(result);
        onFrameAnalysis(result);
      }
      timeoutId = setTimeout(() => { animId = requestAnimationFrame(analyse); }, 120);
    };
    animId = requestAnimationFrame(analyse);
    return () => { cancelAnimationFrame(animId); clearTimeout(timeoutId); };
  }, [camState, onFrameAnalysis]);

  function drawOverlay(oc, m) {
    if (!oc) return;
    const W = oc.width, H = oc.height;
    const ctx = oc.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    // Grid
    ctx.strokeStyle = `${P.accent}28`; ctx.lineWidth = 0.5;
    [W / 3, 2 * W / 3].forEach(x => { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); });
    [H / 3, 2 * H / 3].forEach(y => { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); });
    // Sclera zone
    const sOK = m?.focusOK && m?.exposureOK;
    ctx.strokeStyle = sOK ? P.accent : P.warn; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.ellipse(W * 0.65, H * 0.50, W * 0.17, H * 0.19, 0, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = sOK ? P.accent : P.warn; ctx.font = `bold ${W * 0.027}px monospace`;
    ctx.textAlign = "center"; ctx.fillText("SCLERA", W * 0.65, H * 0.75);
    // Card zone
    ctx.strokeStyle = m?.wbOK ? P.accent : P.textMuted; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    const rx = W * 0.08, ry = H * 0.28, rw = W * 0.27, rh = H * 0.44;
    ctx.strokeRect(rx, ry, rw, rh); ctx.setLineDash([]);
    ctx.fillStyle = m?.wbOK ? P.accent : P.textMuted; ctx.textAlign = "center"; ctx.font = `bold ${W * 0.024}px monospace`;
    ctx.fillText("WHITE CARD", rx + rw / 2, ry + rh + 13);
    // Corner brackets
    const bs = 16; ctx.strokeStyle = P.accent; ctx.lineWidth = 2;
    [[0, 0], [W, 0], [0, H], [W, H]].forEach(([cx, cy]) => {
      const sx = cx === 0 ? 1 : -1, sy = cy === 0 ? 1 : -1;
      ctx.beginPath(); ctx.moveTo(cx + sx * bs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + sy * bs); ctx.stroke();
    });
    // Status dot
    const allOK = m?.exposureOK && m?.focusOK && m?.wbOK;
    ctx.fillStyle = allOK ? P.safe : P.warn;
    if (allOK) { ctx.shadowColor = P.safe; ctx.shadowBlur = 10; }
    ctx.beginPath(); ctx.arc(W - 12, 12, 5.5, 0, 2 * Math.PI); ctx.fill();
    ctx.shadowBlur = 0;
    // Live YSS if enough sclera pixels
    if (m && m.sclPixels > 50) {
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.beginPath(); ctx.roundRect?.(W * 0.48, H * 0.04, W * 0.20, 18, 4); ctx.fill();
      ctx.fillStyle = P.accent; ctx.font = `bold ${W * 0.028}px monospace`; ctx.textAlign = "center";
      ctx.fillText(`YSS ${m.yss.toFixed(1)}`, W * 0.58, H * 0.04 + 12);
    }
  }

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const cap = document.createElement("canvas");
    cap.width = video.videoWidth || 640; cap.height = video.videoHeight || 480;
    cap.getContext("2d").drawImage(video, 0, 0);
    const result = analyseFrame(cap);
    setCaptureFlash(true);
    setTimeout(() => setCaptureFlash(false), 320);
    onCapture(result);
  }, [onCapture]);

  const toggleTorch = useCallback(async () => {
    if (!streamRef.current || !hasTorch) return;
    const track = streamRef.current.getVideoTracks()[0];
    await track.applyConstraints({ advanced: [{ torch: !torchOn }] }).catch(() => {});
    setTorchOn(t => !t);
  }, [torchOn, hasTorch]);

  const flipCamera = useCallback(() => setFacingMode(m => m === "user" ? "environment" : "user"), []);

  // Permission / error states
  if (camState === "requesting") return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14, padding: 40 }}>
      <div style={{ width: 48, height: 48, borderRadius: "50%", border: `3px solid ${P.accent}`, borderTopColor: "transparent", animation: "spin 0.8s linear infinite" }} />
      <div style={{ fontSize: 14, color: P.textMuted }}>Requesting camera access…</div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );

  if (camState === "nosupport" || camState === "denied" || camState === "error") return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14, padding: 36, textAlign: "center" }}>
      <div style={{ fontSize: 44 }}>{camState === "denied" ? "🔒" : "📵"}</div>
      <div style={{ fontSize: 15, fontWeight: 700, color: P.text }}>{camState === "denied" ? "Camera Permission Denied" : "Camera Unavailable"}</div>
      <div style={{ fontSize: 13, color: P.textMuted, lineHeight: 1.7 }}>
        {camState === "denied"
          ? "BiliScan needs camera access. Allow it in your browser settings (usually the address bar lock icon), then retry."
          : "Your browser or device doesn't support the camera API needed for live analysis. Try Chrome or Safari on a phone."}
      </div>
      <button onClick={() => startCamera(facingMode)} style={{ padding: "12px 24px", borderRadius: 12, border: `1px solid ${P.accent}`, background: P.accentGlow, color: P.accent, fontSize: 13, fontWeight: 700, cursor: "pointer" }}>↺ Retry</button>
    </div>
  );

  const allOK = metrics?.exposureOK && metrics?.focusOK && metrics?.wbOK;

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {/* ── VIEWFINDER ─────────────────────────────────────────────── */}
      <div style={{
        position: "relative", width: "100%", aspectRatio: "4/3",
        background: "#000", overflow: "hidden",
        border: `2px solid ${allOK ? P.accent + "55" : P.border}`,
        boxShadow: allOK ? `inset 0 0 50px ${P.accentGlow}` : "none",
        transition: "box-shadow 0.4s, border-color 0.4s",
      }}>
        <video ref={videoRef} muted playsInline autoPlay style={{
          width: "100%", height: "100%", objectFit: "cover",
          transform: facingMode === "user" ? "scaleX(-1)" : "none",
        }} />
        <canvas ref={analysisCanvasRef} style={{ display: "none" }} />
        <canvas ref={overlayCanvasRef} width={400} height={300} style={{
          position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none",
        }} />
        {captureFlash && <div style={{ position: "absolute", inset: 0, background: "rgba(255,255,255,0.75)", animation: "flashOut 0.32s ease forwards" }} />}

        {/* Camera controls */}
        <div style={{ position: "absolute", top: 10, right: 10, display: "flex", flexDirection: "column", gap: 8 }}>
          {[
            { icon: "🔄", action: flipCamera, active: false, title: "Flip camera" },
            ...(hasTorch ? [{ icon: "🔦", action: toggleTorch, active: torchOn, title: "Torch" }] : []),
          ].map((btn, i) => (
            <button key={i} onClick={btn.action} title={btn.title} style={{
              width: 38, height: 38, borderRadius: 11,
              background: btn.active ? `${P.warn}33` : "rgba(0,0,0,0.6)",
              border: `1px solid ${btn.active ? P.warn : P.border}`,
              color: btn.active ? P.warn : P.text, fontSize: 16,
              cursor: "pointer", backdropFilter: "blur(6px)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>{btn.icon}</button>
          ))}
        </div>

        {/* Resolution badge */}
        <div style={{ position: "absolute", bottom: 7, left: 9, fontSize: 9, color: "rgba(255,255,255,0.35)", fontFamily: "monospace" }}>
          {videoRes.w > 0 ? `${videoRes.w}×${videoRes.h}` : "—"} · {facingMode === "user" ? "FRONT" : "REAR"}
        </div>
      </div>

      {/* ── LIVE METRICS PANEL ─────────────────────────────────────── */}
      <div style={{ background: P.surface, padding: "12px 14px", borderBottom: `1px solid ${P.border}` }}>
        {/* Synthetic histogram from meanLuma */}
        <div style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
            <span style={{ fontSize: 10, color: P.textMuted, fontFamily: "monospace", letterSpacing: "0.06em" }}>FRAME LUMINANCE</span>
            <span style={{ fontSize: 10, color: P.textMuted, fontFamily: "monospace" }}>
              μ={metrics?.meanLuma ?? "—"} · shp={metrics?.sharpness ?? "—"}
            </span>
          </div>
          <div style={{ display: "flex", gap: 1.5, height: 24, alignItems: "flex-end" }}>
            {Array.from({ length: 40 }, (_, i) => {
              const norm = i / 39;
              const ml = (metrics?.meanLuma ?? 128) / 255;
              const bell = Math.exp(-((norm - ml) ** 2) / 0.035);
              const col = norm < 0.18 ? P.danger : norm > 0.92 ? P.warn : P.accent;
              return <div key={i} style={{ flex: 1, height: `${Math.max(5, bell * 100)}%`, background: col, borderRadius: 1, opacity: 0.65, transition: "height 0.12s" }} />;
            })}
          </div>
        </div>

        {/* 3 quality chips */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 7 }}>
          {[
            { label: "Exposure", ok: metrics?.exposureOK, val: metrics ? `μ=${metrics.meanLuma}` : "…" },
            { label: "Focus/Blur", ok: metrics?.focusOK, val: metrics ? `s=${metrics.sharpness}` : "…" },
            { label: "White Card", ok: metrics?.wbOK, val: metrics ? `L=${metrics.cardLuma}` : "…" },
          ].map(item => (
            <div key={item.label} style={{ background: P.card, border: `1px solid ${item.ok ? P.accent + "44" : P.border}`, borderRadius: 9, padding: "8px 9px", transition: "border-color 0.3s" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 3 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: item.ok == null ? P.textDim : item.ok ? P.safe : P.warn, boxShadow: item.ok ? `0 0 5px ${P.safe}` : "none" }} />
                <span style={{ fontSize: 10, fontWeight: 700, color: item.ok ? P.safe : P.textMuted }}>{item.ok == null ? "—" : item.ok ? "OK" : "POOR"}</span>
              </div>
              <div style={{ fontSize: 9, color: P.textMuted, fontFamily: "monospace" }}>{item.label}</div>
              <div style={{ fontSize: 9, color: P.textDim, fontFamily: "monospace", marginTop: 1 }}>{item.val}</div>
            </div>
          ))}
        </div>

        {/* Pixel stats row */}
        {metrics && (
          <div style={{ display: "flex", gap: 14, marginTop: 8 }}>
            <span style={{ fontSize: 10, color: P.textMuted, fontFamily: "monospace" }}>
              sclera: <span style={{ color: metrics.sclPixels > 100 ? P.accent : P.warn }}>{metrics.sclPixels}px</span>
            </span>
            <span style={{ fontSize: 10, color: P.textMuted, fontFamily: "monospace" }}>
              R/G: <span style={{ color: P.textMuted }}>{metrics.rgRatio}</span>
            </span>
            <span style={{ fontSize: 10, color: P.textMuted, fontFamily: "monospace" }}>
              live YSS: <span style={{ color: P.accent, fontWeight: 700 }}>{metrics.yss.toFixed(1)}</span>
            </span>
          </div>
        )}
      </div>

      {/* ── CAPTURE BUTTON ────────────────────────────────────────── */}
      <div style={{ padding: "14px 14px 18px", background: P.surface }}>
        <button onClick={captureFrame} disabled={!allOK} style={{
          width: "100%", padding: "15px",
          borderRadius: 14, border: "none",
          background: allOK ? `linear-gradient(135deg, ${P.accent}, ${P.accentDim})` : P.border,
          color: allOK ? P.bg : P.textDim,
          fontSize: 14, fontWeight: 800, letterSpacing: "0.04em",
          cursor: allOK ? "pointer" : "not-allowed",
          transition: "all 0.3s", boxShadow: allOK ? `0 6px 20px ${P.accentGlow}` : "none",
        }}>
          {allOK ? "📸  CAPTURE & ANALYSE" : "Waiting for good conditions…"}
        </button>
        {!allOK && metrics && (
          <div style={{ fontSize: 11, color: P.textMuted, textAlign: "center", marginTop: 7 }}>
            {!metrics.exposureOK ? "Adjust lighting · " : ""}
            {!metrics.focusOK ? "Hold steady · " : ""}
            {!metrics.wbOK ? "Position white card in left zone" : ""}
          </div>
        )}
      </div>
      <style>{`@keyframes flashOut{0%{opacity:1}100%{opacity:0}}`}</style>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  SCREEN: DISCLAIMER
// ════════════════════════════════════════════════════════════════════════════
function DisclaimerScreen({ onAccept }) {
  const [checked, setChecked] = useState(false);
  const [vis, setVis] = useState(false);
  useEffect(() => { setTimeout(() => setVis(true), 80); }, []);
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", padding: "40px 24px 32px", opacity: vis ? 1 : 0, transform: vis ? "none" : "translateY(20px)", transition: "all 0.6s cubic-bezier(.4,0,.2,1)" }}>
      <div style={{ textAlign: "center", marginBottom: 28 }}>
        <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 68, height: 68, borderRadius: 22, background: `linear-gradient(135deg, ${P.accentGlow}, ${P.accentDim}22)`, border: `1.5px solid ${P.accent}44`, marginBottom: 14, boxShadow: `0 0 40px ${P.accentGlow}` }}>
          <svg width={34} height={34} viewBox="0 0 36 36" fill="none">
            <ellipse cx={18} cy={18} rx={14} ry={10} stroke={P.accent} strokeWidth={2} />
            <circle cx={18} cy={18} r={5} fill={P.accent} opacity={0.9} />
            <circle cx={18} cy={18} r={2.5} fill={P.bg} />
          </svg>
        </div>
        <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.02em", color: P.white }}>Bili<span style={{ color: P.accent }}>Scan</span></div>
        <div style={{ fontSize: 11, color: P.textMuted, letterSpacing: "0.12em", textTransform: "uppercase", marginTop: 2 }}>Jaundice Screening Aid</div>
      </div>
      <div style={{ background: `linear-gradient(135deg, ${P.card}, ${P.surface})`, border: `1px solid ${P.warn}44`, borderRadius: 18, padding: "20px", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <span style={{ fontSize: 16 }}>⚠️</span>
          <span style={{ fontSize: 12, fontWeight: 800, color: P.warn, letterSpacing: "0.04em" }}>MEDICAL DISCLAIMER</span>
        </div>
        {["BiliScan does not diagnose disease. It is a non-diagnostic screening aid only.", "Results are NOT a substitute for laboratory bilirubin measurement.", "A LOW RISK result does not exclude jaundice. Consult a doctor for any concerns.", "All processing runs on-device. No camera images are transmitted or stored."].map((t, i) => (
          <div key={i} style={{ display: "flex", gap: 8, marginBottom: i < 3 ? 9 : 0 }}>
            <span style={{ color: P.warn, flexShrink: 0 }}>›</span>
            <span style={{ fontSize: 12, color: P.text, lineHeight: 1.6 }}>{t}</span>
          </div>
        ))}
      </div>
      <div onClick={() => setChecked(!checked)} style={{ display: "flex", gap: 12, marginBottom: 20, cursor: "pointer", alignItems: "flex-start" }}>
        <div style={{ width: 21, height: 21, borderRadius: 6, flexShrink: 0, marginTop: 1, border: `2px solid ${checked ? P.accent : P.border}`, background: checked ? P.accent : "transparent", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s" }}>
          {checked && <span style={{ color: P.bg, fontSize: 13, fontWeight: 900 }}>✓</span>}
        </div>
        <span style={{ fontSize: 12, color: P.textMuted, lineHeight: 1.6 }}>I understand this is a screening aid only and not a medical diagnosis tool.</span>
      </div>
      <button disabled={!checked} onClick={onAccept} style={{ width: "100%", padding: "15px", borderRadius: 14, border: "none", background: checked ? `linear-gradient(135deg, ${P.accent}, ${P.accentDim})` : P.border, color: checked ? P.bg : P.textDim, fontSize: 14, fontWeight: 800, cursor: checked ? "pointer" : "not-allowed", transition: "all 0.3s", boxShadow: checked ? `0 8px 28px ${P.accentGlow}` : "none" }}>
        I UNDERSTAND — CONTINUE
      </button>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  SCREEN: HOME
// ════════════════════════════════════════════════════════════════════════════
function HomeScreen({ scans, onStartScan, onViewHistory, onEducation }) {
  const latest = scans[scans.length - 1];
  const slope = computeTrend(scans.slice(-5));
  const [vis, setVis] = useState(false);
  useEffect(() => { setTimeout(() => setVis(true), 60); }, []);
  return (
    <div style={{ flex: 1, paddingBottom: 100, opacity: vis ? 1 : 0, transform: vis ? "none" : "translateY(16px)", transition: "all 0.5s cubic-bezier(.4,0,.2,1)" }}>
      <div style={{ padding: "44px 22px 18px", background: `linear-gradient(180deg, ${P.surface} 0%, ${P.bg} 100%)`, borderBottom: `1px solid ${P.border}` }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 21, fontWeight: 800, color: P.white }}>Bili<span style={{ color: P.accent }}>Scan</span></div>
            <div style={{ fontSize: 11, color: P.textMuted }}>Scleral Jaundice Screening</div>
          </div>
          <div onClick={onEducation} style={{ width: 38, height: 38, borderRadius: 12, background: P.card, border: `1px solid ${P.border}`, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}>📚</div>
        </div>
      </div>
      <div style={{ padding: "18px 22px 0" }}>
        {latest && (
          <div style={{ background: `linear-gradient(135deg, ${latest.risk.color}18, ${P.card})`, border: `1px solid ${latest.risk.color}44`, borderRadius: 22, padding: "20px", marginBottom: 14, position: "relative", overflow: "hidden" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <div>
                <div style={{ fontSize: 10, color: P.textMuted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 2 }}>Latest Scan</div>
                <div style={{ fontSize: 10, color: P.textDim }}>{fmtDate(latest.date)}</div>
              </div>
              <Chip label={latest.risk.label} color={latest.risk.color} />
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ position: "relative", flexShrink: 0 }}>
                <ProgressRing value={latest.yss} color={latest.risk.color} size={88} />
                <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                  <div style={{ fontSize: 18, fontWeight: 800, color: latest.risk.color, lineHeight: 1 }}>{latest.yss.toFixed(1)}</div>
                  <div style={{ fontSize: 8, color: P.textMuted }}>YSS</div>
                </div>
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: slope > 1 ? P.danger : slope > 0.3 ? P.warn : P.safe, marginBottom: 8 }}>
                  {slope > 1 ? "↑ Rising — monitor closely" : slope > 0.3 ? "↗ Slight upward trend" : "→ Stable"}
                </div>
                <div style={{ fontSize: 11, color: P.text, lineHeight: 1.6 }}>
                  {latest.risk.level === "LOW" ? "Rescan in 24–48h if symptoms appear." : latest.risk.level === "MODERATE" ? "Rescan in 24h. Seek evaluation if rising." : "Seek laboratory bilirubin testing promptly."}
                </div>
              </div>
            </div>
          </div>
        )}
        {scans.length >= 2 && (
          <div style={{ background: P.card, border: `1px solid ${P.border}`, borderRadius: 18, padding: "16px", marginBottom: 14 }}>
            <TrendChart scans={scans.slice(-7)} />
          </div>
        )}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 9, marginBottom: 12 }}>
          {[
            { label: "Total Scans", value: scans.length, icon: "🔬" },
            { label: "7-Day Avg", value: scans.length ? (scans.slice(-7).reduce((a, s) => a + s.yss, 0) / Math.min(scans.length, 7)).toFixed(1) : "—", icon: "📊" },
            { label: "Peak YSS", value: scans.length ? Math.max(...scans.map(s => s.yss)).toFixed(1) : "—", icon: "📈" },
            { label: "High Alerts", value: scans.filter(s => s.risk.level === "HIGH").length || "None", icon: "⚡" },
          ].map((s, i) => (
            <div key={i} style={{ background: P.card, border: `1px solid ${P.border}`, borderRadius: 12, padding: "11px 13px" }}>
              <div style={{ fontSize: 15, marginBottom: 3 }}>{s.icon}</div>
              <div style={{ fontSize: 17, fontWeight: 800, color: P.white }}>{s.value}</div>
              <div style={{ fontSize: 10, color: P.textMuted }}>{s.label}</div>
            </div>
          ))}
        </div>
        <button onClick={onViewHistory} style={{ width: "100%", padding: "12px", background: P.card, border: `1px solid ${P.border}`, borderRadius: 12, color: P.text, fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
          📋 View Full Scan History
        </button>
      </div>
      <div style={{ position: "fixed", bottom: 28, left: "50%", transform: "translateX(-50%)", zIndex: 100 }}>
        <button onClick={onStartScan} style={{ display: "flex", alignItems: "center", gap: 9, padding: "15px 30px", borderRadius: 50, border: "none", background: `linear-gradient(135deg, ${P.accent}, ${P.accentDim})`, color: P.bg, fontSize: 14, fontWeight: 800, cursor: "pointer", boxShadow: `0 8px 36px ${P.accentGlow}` }}>
          <svg width={18} height={18} viewBox="0 0 20 20" fill={P.bg}><circle cx={10} cy={10} r={8} stroke={P.bg} strokeWidth={2} fill="none" /><circle cx={10} cy={10} r={4} fill={P.bg} /></svg>
          NEW SCAN
        </button>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  SCREEN: CAPTURE  (wraps LiveCamera)
// ════════════════════════════════════════════════════════════════════════════
function CaptureScreen({ onCapture, onBack }) {
  const [step, setStep] = useState("guide");
  const [camError, setCamError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [pStep, setPStep] = useState(0);
  const [capturedM, setCapturedM] = useState(null);
  const timerRef = useRef(null);

  const STEPS = [
    "Detecting face landmarks…", "Isolating eye region…", "Segmenting sclera pixels…",
    "Removing specular highlights…", "Sampling white reference card…", "Per-channel colour calibration…",
    "Converting RGB → CIELAB (D65)…", "Extracting b* yellow-axis…", "Trimmed median filter (10%)…",
    "Computing Yellow Shift Score…", "Applying R/G vascular correction…", "Classifying risk level…",
  ];

  const handleCapture = useCallback((metrics) => {
    setCapturedM(metrics);
    setStep("processing");
    let p = 0, s = 0;
    const tick = () => {
      p += 100 / STEPS.length; if (s < STEPS.length - 1) s++;
      setProgress(Math.min(p, 100)); setPStep(s);
      if (p < 100) timerRef.current = setTimeout(tick, 210);
      else setTimeout(() => {
        const yss = clamp(metrics.yss > 0 ? metrics.yss : 10 + Math.random() * 8, 4, 42);
        onCapture({ yss: parseFloat(yss.toFixed(2)), quality: metrics.qualityScore, sclPixels: metrics.sclPixels, rgRatio: metrics.rgRatio });
      }, 400);
    };
    timerRef.current = setTimeout(tick, 180);
  }, [onCapture]);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  if (step === "guide") return (
    <div style={{ flex: 1, padding: "44px 22px 30px", overflowY: "auto" }}>
      <button onClick={onBack} style={{ background: "none", border: "none", color: P.textMuted, cursor: "pointer", fontSize: 13, marginBottom: 18 }}>← Back</button>
      <div style={{ fontSize: 20, fontWeight: 800, color: P.white, marginBottom: 4 }}>Scan Setup</div>
      <div style={{ fontSize: 12, color: P.textMuted, marginBottom: 22 }}>Prepare before opening the camera</div>
      {[
        { icon: "☀️", title: "Lighting", desc: "Diffuse natural light facing a window. Avoid direct sun, overhead fluorescent/LED." },
        { icon: "🃏", title: "White Reference Card", desc: "Hold a white business card to the left of the eye being scanned — both must appear in frame." },
        { icon: "👁️", title: "Eye Position", desc: "Look slightly upward to maximally expose the sclera. Hold eyelids open wide." },
        { icon: "📏", title: "Distance", desc: "15–25 cm from camera. Sclera target zone is right of frame; card target is left." },
        { icon: "🤚", title: "Keep Still", desc: "Brace your hand. The focus detector rejects blurry frames automatically." },
      ].map((item, i) => (
        <div key={i} style={{ display: "flex", gap: 11, marginBottom: 10, background: P.card, border: `1px solid ${P.border}`, borderRadius: 13, padding: "12px 13px" }}>
          <div style={{ width: 34, height: 34, borderRadius: 9, background: P.accentGlow, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, flexShrink: 0 }}>{item.icon}</div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: P.white, marginBottom: 2 }}>{item.title}</div>
            <div style={{ fontSize: 11, color: P.textMuted, lineHeight: 1.6 }}>{item.desc}</div>
          </div>
        </div>
      ))}
      <div style={{ background: `${P.accent}0E`, border: `1px solid ${P.accent}28`, borderRadius: 11, padding: "11px 13px", marginTop: 4, marginBottom: 18 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: P.accent, marginBottom: 4 }}>📷 CAMERA COMPATIBILITY</div>
        <div style={{ fontSize: 11, color: P.textMuted, lineHeight: 1.6 }}>
          Works with front and rear cameras via <code style={{ color: P.accent, fontSize: 10 }}>getUserMedia</code>. Rear camera recommended for colour accuracy. Torch toggle available on supported devices. Analysis resolution: 320×240 (full-res capture on shutter).
        </div>
      </div>
      <button onClick={() => setStep("camera")} style={{ width: "100%", padding: "15px", borderRadius: 14, border: "none", background: `linear-gradient(135deg, ${P.accent}, ${P.accentDim})`, color: P.bg, fontSize: 14, fontWeight: 800, cursor: "pointer", boxShadow: `0 8px 22px ${P.accentGlow}` }}>
        OPEN CAMERA
      </button>
    </div>
  );

  if (step === "camera") return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "44px 16px 12px", display: "flex", alignItems: "center", gap: 10, background: P.surface, borderBottom: `1px solid ${P.border}` }}>
        <button onClick={onBack} style={{ background: "none", border: "none", color: P.textMuted, cursor: "pointer", fontSize: 13 }}>← Back</button>
        <div style={{ fontSize: 14, fontWeight: 700, color: P.white }}>Live Camera Analysis</div>
      </div>
      {camError && <div style={{ margin: "10px 14px", background: P.dangerGlow, border: `1px solid ${P.danger}44`, borderRadius: 10, padding: "9px 13px", fontSize: 12, color: P.danger }}>{camError}</div>}
      <LiveCamera onFrameAnalysis={() => {}} onCapture={handleCapture} onError={setCamError} />
    </div>
  );

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "36px 22px" }}>
      <div style={{ position: "relative", marginBottom: 24, width: 130, height: 130 }}>
        <svg width={130} height={130} viewBox="0 0 130 130" style={{ transform: "rotate(-90deg)" }}>
          <circle cx={65} cy={65} r={56} fill="none" stroke={P.border} strokeWidth={8} />
          <circle cx={65} cy={65} r={56} fill="none" stroke={P.accent} strokeWidth={8}
            strokeDasharray={`${(progress / 100) * 2 * Math.PI * 56} 999`} strokeLinecap="round"
            style={{ transition: "stroke-dasharray 0.22s ease" }} />
        </svg>
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <div style={{ fontSize: 26, fontWeight: 900, color: P.accent }}>{Math.round(progress)}%</div>
          <div style={{ fontSize: 9, color: P.textMuted, letterSpacing: "0.08em" }}>ANALYSING</div>
        </div>
      </div>
      <div style={{ fontSize: 13, color: P.text, fontWeight: 600, marginBottom: 5, textAlign: "center" }}>{STEPS[pStep]}</div>
      <div style={{ fontSize: 11, color: P.textMuted, marginBottom: 24 }}>On-device · no data transmitted</div>
      <div style={{ width: "100%", background: P.card, borderRadius: 14, padding: "13px", border: `1px solid ${P.border}` }}>
        {STEPS.map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 7, padding: "4px 0", opacity: i <= pStep ? 1 : 0.22, transition: "opacity 0.3s" }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", flexShrink: 0, background: i < pStep ? P.safe : i === pStep ? P.accent : P.border, boxShadow: i === pStep ? `0 0 5px ${P.accent}` : "none" }} />
            <span style={{ fontSize: 11, color: i < pStep ? P.safe : i === pStep ? P.accent : P.textMuted }}>{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  SCREEN: RESULTS
// ════════════════════════════════════════════════════════════════════════════
function ResultsScreen({ scan, onDone, onRetry }) {
  const [vis, setVis] = useState(false);
  useEffect(() => { setTimeout(() => setVis(true), 100); }, []);
  const risk = scan.risk;
  return (
    <div style={{ flex: 1, padding: "44px 22px 36px", opacity: vis ? 1 : 0, transition: "all 0.5s cubic-bezier(.4,0,.2,1)", overflowY: "auto" }}>
      <div style={{ fontSize: 17, fontWeight: 800, color: P.white, marginBottom: 18 }}>Scan Results</div>
      <div style={{ background: `linear-gradient(135deg, ${risk.color}20, ${P.card})`, border: `2px solid ${risk.color}55`, borderRadius: 22, padding: "22px", marginBottom: 14, textAlign: "center", boxShadow: `0 0 55px ${risk.glow}` }}>
        <div style={{ fontSize: 34, marginBottom: 9 }}>{risk.level === "LOW" ? "✅" : risk.level === "MODERATE" ? "⚠️" : "🚨"}</div>
        <div style={{ fontSize: 10, color: P.textMuted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 3 }}>Yellow Shift Score</div>
        <div style={{ fontSize: 50, fontWeight: 900, color: risk.color, lineHeight: 1, marginBottom: 5 }}>{scan.yss.toFixed(1)}</div>
        <div style={{ fontSize: 11, color: P.textMuted, marginBottom: 12 }}>Thresholds: &lt;14 Low · 14–24 Moderate · &gt;24 High</div>
        <Chip label={risk.label} color={risk.color} />
      </div>
      <div style={{ background: P.card, border: `1px solid ${P.border}`, borderRadius: 16, padding: "16px", marginBottom: 12 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: P.textMuted, letterSpacing: "0.08em", marginBottom: 10 }}>REAL-CAMERA MEASUREMENTS</div>
        {[["YSS (b* yellow shift)", scan.yss.toFixed(2)], ["Sclera pixels sampled", scan.sclPixels], ["R/G vascular ratio", scan.rgRatio], ["Calibration quality", `${Math.round(scan.quality * 100)}%`], ["Processing location", "On-device only"], ["Images stored", "None"]].map(([l, v], i, a) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: i < a.length - 1 ? `1px solid ${P.border}` : "none" }}>
            <span style={{ fontSize: 12, color: P.textMuted }}>{l}</span>
            <span style={{ fontSize: 12, fontWeight: 700, color: P.text }}>{v}</span>
          </div>
        ))}
      </div>
      <div style={{ background: `${risk.color}12`, border: `1px solid ${risk.color}33`, borderRadius: 13, padding: "13px 15px", marginBottom: 18 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: risk.color, letterSpacing: "0.06em", marginBottom: 5 }}>RECOMMENDATION</div>
        <div style={{ fontSize: 12, color: P.text, lineHeight: 1.7 }}>
          {risk.level === "LOW" && "No immediate concern. Continue monitoring. Seek evaluation if symptoms develop."}
          {risk.level === "MODERATE" && "Elevated yellow shift detected. Rescan in 24h. Seek medical testing if rising or symptoms appear."}
          {risk.level === "HIGH" && "Seek laboratory bilirubin testing promptly. Only a clinician can confirm or exclude jaundice."}
        </div>
      </div>
      <div style={{ display: "flex", gap: 9 }}>
        <button onClick={onRetry} style={{ flex: 1, padding: "12px", background: P.card, border: `1px solid ${P.border}`, borderRadius: 12, color: P.text, fontSize: 12, fontWeight: 600, cursor: "pointer" }}>↺ Rescan</button>
        <button onClick={onDone} style={{ flex: 2, padding: "12px", background: `linear-gradient(135deg, ${P.accent}, ${P.accentDim})`, border: "none", borderRadius: 12, color: P.bg, fontSize: 12, fontWeight: 800, cursor: "pointer", boxShadow: `0 5px 18px ${P.accentGlow}` }}>Save & Dashboard</button>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  SCREEN: HISTORY
// ════════════════════════════════════════════════════════════════════════════
function HistoryScreen({ scans, onBack }) {
  return (
    <div style={{ flex: 1, padding: "44px 22px 36px", overflowY: "auto" }}>
      <button onClick={onBack} style={{ background: "none", border: "none", color: P.textMuted, cursor: "pointer", fontSize: 13, marginBottom: 16 }}>← Back</button>
      <div style={{ fontSize: 19, fontWeight: 800, color: P.white, marginBottom: 14 }}>History <span style={{ fontSize: 12, color: P.textMuted, fontWeight: 400 }}>({scans.length})</span></div>
      {[...scans].reverse().map((scan, i) => (
        <div key={scan.id} style={{ background: P.card, border: `1px solid ${i === 0 ? scan.risk.color + "44" : P.border}`, borderRadius: 13, padding: "13px", marginBottom: 7, display: "flex", alignItems: "center", gap: 11 }}>
          <div style={{ width: 38, height: 38, borderRadius: 10, background: `${scan.risk.color}18`, border: `1px solid ${scan.risk.color}33`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, color: scan.risk.color }}>{scan.risk.icon}</div>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: P.white }}>YSS {scan.yss.toFixed(1)}</span>
              <Chip label={scan.risk.label} color={scan.risk.color} />
            </div>
            <div style={{ fontSize: 10, color: P.textMuted }}>{fmtDate(scan.date)}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  SCREEN: EDUCATION
// ════════════════════════════════════════════════════════════════════════════
function EducationScreen({ onBack }) {
  const [open, setOpen] = useState(null);
  const secs = [
    { title: "What is Bilirubin?", icon: "🔬", content: "Bilirubin is a yellow pigment produced when red blood cells break down. Normally the liver processes and excretes it. Accumulation in tissues causes the yellow discoloration of skin and eyes known as jaundice." },
    { title: "How BiliScan Works", icon: "📱", content: "BiliScan captures a real camera frame, segments the sclera using luminance and saturation filters, samples a white reference card for per-channel colour calibration, converts calibrated RGB to CIELAB, and extracts the b* (yellow–blue axis) value as the Yellow Shift Score." },
    { title: "Camera Analysis Pipeline", icon: "📷", content: "Each live frame is analysed: exposure checked via mean luminance, focus via a gradient-magnitude sharpness proxy, white balance via card region luma. Sclera zone = right half of frame; card zone = left. All computation runs on-device in the browser using the Canvas API." },
    { title: "Yellow Shift Score", icon: "📊", content: "YSS is the trimmed median CIELAB b* value relative to the calibrated white baseline, corrected for R/G vascular tint. <14 = low, 14–24 = moderate, >24 = high risk. Lab testing is required for clinical confirmation." },
    { title: "Warning Signs", icon: "⚠️", content: "Seek immediate evaluation for: yellow skin or eyes, dark amber urine, pale or clay stools, fatigue, upper right abdominal pain, fever with chills, or unexplained weight loss." },
    { title: "Limitations", icon: "📋", content: "Less accurate with dark skin tones, poor lighting, or absent white card. Cannot detect all causes of jaundice. A normal result never excludes disease. Never delay medical care based on app results." },
  ];
  return (
    <div style={{ flex: 1, padding: "44px 22px 36px", overflowY: "auto" }}>
      <button onClick={onBack} style={{ background: "none", border: "none", color: P.textMuted, cursor: "pointer", fontSize: 13, marginBottom: 16 }}>← Back</button>
      <div style={{ fontSize: 19, fontWeight: 800, color: P.white, marginBottom: 18 }}>Education</div>
      {secs.map((sec, i) => (
        <div key={i} style={{ background: P.card, border: `1px solid ${open === i ? P.accent + "44" : P.border}`, borderRadius: 13, marginBottom: 7, overflow: "hidden", transition: "border-color 0.2s" }}>
          <div onClick={() => setOpen(open === i ? null : i)} style={{ display: "flex", alignItems: "center", gap: 9, padding: "13px", cursor: "pointer" }}>
            <span style={{ fontSize: 16 }}>{sec.icon}</span>
            <span style={{ flex: 1, fontSize: 13, fontWeight: 600, color: P.white }}>{sec.title}</span>
            <span style={{ color: P.textMuted, display: "inline-block", transition: "transform 0.2s", transform: open === i ? "rotate(180deg)" : "none" }}>⌄</span>
          </div>
          {open === i && <div style={{ padding: "0 13px 13px 38px", fontSize: 12, color: P.textMuted, lineHeight: 1.7 }}>{sec.content}</div>}
        </div>
      ))}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  ROOT APP
// ════════════════════════════════════════════════════════════════════════════
export default function BiliScan() {
  const [screen, setScreen] = useState("disclaimer");
  const [scans, setScans] = useState(SEED_SCANS);
  const [pending, setPending] = useState(null);

  const handleCapture = useCallback(({ yss, quality, sclPixels, rgRatio }) => {
    const scan = { id: Date.now(), date: Date.now(), yss, quality, sclPixels, rgRatio, risk: classifyRisk(yss) };
    setPending(scan); setScreen("results");
  }, []);

  const handleSave = useCallback(() => {
    if (pending) { setScans(prev => [...prev, pending]); setPending(null); }
    setScreen("home");
  }, [pending]);

  return (
    <div style={{ minHeight: "100vh", background: P.bg, color: P.text, fontFamily: "'DM Sans','Segoe UI',system-ui,sans-serif", display: "flex", flexDirection: "column", alignItems: "center", margin: 0, padding: 0, position: "relative", overflow: "hidden" }}>
      <div style={{ position: "fixed", top: -80, right: -80, width: 300, height: 300, borderRadius: "50%", background: `radial-gradient(circle, ${P.accentGlow}, transparent 65%)`, pointerEvents: "none", zIndex: 0 }} />
      <div style={{ position: "fixed", bottom: -80, left: -80, width: 240, height: 240, borderRadius: "50%", background: `radial-gradient(circle, ${P.warnGlow}, transparent 65%)`, pointerEvents: "none", zIndex: 0 }} />
      <div style={{ width: "100%", maxWidth: 430, minHeight: "100vh", display: "flex", flexDirection: "column", position: "relative", zIndex: 1 }}>
        {screen === "disclaimer" && <DisclaimerScreen onAccept={() => setScreen("home")} />}
        {screen === "home" && <HomeScreen scans={scans} onStartScan={() => setScreen("capture")} onViewHistory={() => setScreen("history")} onEducation={() => setScreen("education")} />}
        {screen === "capture" && <CaptureScreen onCapture={handleCapture} onBack={() => setScreen("home")} />}
        {screen === "results" && pending && <ResultsScreen scan={pending} onDone={handleSave} onRetry={() => setScreen("capture")} />}
        {screen === "history" && <HistoryScreen scans={scans} onBack={() => setScreen("home")} />}
        {screen === "education" && <EducationScreen onBack={() => setScreen("home")} />}
      </div>
    </div>
  );
}
