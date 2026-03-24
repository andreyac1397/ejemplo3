document.addEventListener("DOMContentLoaded", () => {
  // =========================
  // 1) FORM + RESULT UI
  // =========================
  const form = document.getElementById("formulario");
  const resultBox = document.getElementById("resultado");
  const errorBox = document.getElementById("error");

  function showResult(html, risk) {
    errorBox.classList.add("hidden");
    resultBox.classList.remove("hidden");
    resultBox.classList.remove("risk-low", "risk-high");
    resultBox.classList.add(risk === "Alto" ? "risk-high" : "risk-low");
    resultBox.innerHTML = html;
  }

  function showError(msg) {
    resultBox.classList.add("hidden");
    errorBox.classList.remove("hidden");
    errorBox.textContent = msg;
  }

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const fd = new FormData(form);
      const payload = Object.fromEntries(fd.entries());

      ["age", "credit_amount", "month_duration", "payment_to_income_ratio"].forEach((k) => {
        payload[k] = Number(payload[k]);
      });

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await res.json();
        if (!res.ok) {
          showError(data.error || "Error en la predicción.");
          return;
        }

        const prob = Math.round(data.probabilidad_default * 10000) / 100;
        const html = `
          <div class="result-title">Resultado: <strong>${data.riesgo} Riesgo</strong></div>
          <div class="result-sub">Probabilidad de default (bad): <strong>${prob}%</strong></div>
          <div class="result-sub">${data.mensaje}</div>
        `;
        showResult(html, data.riesgo);
      } catch {
        showError("No se pudo conectar con la API. ¿Está corriendo uvicorn?");
      }
    });
  }

  // =========================
  // 2) DASHBOARD CHARTS
  // =========================
  const dataEl = document.getElementById("dashboard-data");
  let dash = null;

  try {
    dash = dataEl ? JSON.parse((dataEl.textContent || "").trim()) : null;
  } catch {
    dash = null;
  }

  if (!dash) return;

  // Espera un frame para asegurar tamaños de canvas
  requestAnimationFrame(() => {
    const donutCanvas = document.getElementById("goodBadChart");
    if (donutCanvas) drawDonutGoodBad(donutCanvas, dash.dataset.good, dash.dataset.bad);

    const metricsCanvas = document.getElementById("modelMetricsChart");
    if (metricsCanvas) drawBarsMetrics(metricsCanvas, dash.model);

    const heatmapContainer = document.getElementById("confusionMatrixHeatmap");
    if (heatmapContainer) drawConfusionHeatmap(heatmapContainer, dash.model.confusion_matrix);
  });

  function clamp01(x) {
    const n = Number(x);
    if (Number.isNaN(n)) return 0;
    return Math.max(0, Math.min(1, n));
  }

  function drawDonutGoodBad(canvas, good, bad) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = (canvas.width = (canvas.clientWidth || 520));
    const H = (canvas.height = 140);

    ctx.clearRect(0, 0, W, H);

    const total = Math.max(1, good + bad);
    const goodPct = good / total;
    const badPct = bad / total;

    const cx = Math.round(W * 0.22);
    const cy = Math.round(H * 0.55);
    const rOuter = Math.min(W, H) * 0.33;
    const rInner = rOuter * 0.62;

    ctx.lineWidth = rOuter - rInner;

    // base ring
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.arc(cx, cy, (rOuter + rInner) / 2, 0, Math.PI * 2);
    ctx.stroke();

    let start = -Math.PI / 2;

    // good
    ctx.strokeStyle = "rgba(110,231,183,0.85)";
    ctx.beginPath();
    ctx.arc(cx, cy, (rOuter + rInner) / 2, start, start + Math.PI * 2 * goodPct);
    ctx.stroke();

    start += Math.PI * 2 * goodPct;

    // bad
    ctx.strokeStyle = "rgba(251,113,133,0.85)";
    ctx.beginPath();
    ctx.arc(cx, cy, (rOuter + rInner) / 2, start, start + Math.PI * 2 * badPct);
    ctx.stroke();

    // center text
    ctx.fillStyle = "rgba(238,242,255,0.95)";
    ctx.font = "bold 14px system-ui, -apple-system, Segoe UI, Roboto, Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${Math.round(badPct * 1000) / 10}% Bad`, cx, cy);

    // legend
    const lx = Math.round(W * 0.48);
    const ly = Math.round(H * 0.30);
    ctx.textAlign = "left";
    ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, Arial";

    legend(ctx, lx, ly, "Good", good, goodPct, "rgba(110,231,183,0.85)");
    legend(ctx, lx, ly + 26, "Bad", bad, badPct, "rgba(251,113,133,0.85)");
  }

  function legend(ctx, x, y, label, count, pct, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, 10, 10);
    ctx.fillStyle = "rgba(238,242,255,0.95)";
    ctx.fillText(`${label}: ${count} (${Math.round(pct * 1000) / 10}%)`, x + 16, y + 9);
  }

  function drawBarsMetrics(canvas, m) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = (canvas.width = (canvas.clientWidth || 520));
    const H = (canvas.height = 180);

    ctx.clearRect(0, 0, W, H);

    const items = [
      { label: "Accuracy", value: clamp01(m.accuracy) },
      { label: "Precision", value: clamp01(m.precision) },
      { label: "Recall", value: clamp01(m.recall) },
      { label: "F1", value: clamp01(m.f1) },
    ];

    const padding = 16;
    const chartH = H - 50;
    const baseY = H - 28;
    const barW = Math.floor((W - padding * 2) / items.length) - 12;

    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.beginPath();
    ctx.moveTo(padding, baseY + 0.5);
    ctx.lineTo(W - padding, baseY + 0.5);
    ctx.stroke();

    ctx.textAlign = "center";
    ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, Arial";

    items.forEach((it, i) => {
      const x = padding + i * (barW + 12) + 6;
      const h = Math.round(it.value * chartH);
      const y = baseY - h;

      ctx.fillStyle = "rgba(110,231,183,0.45)";
      ctx.fillRect(x, y, barW, h);

      ctx.fillStyle = "rgba(238,242,255,0.95)";
      ctx.fillText(String(Math.round(it.value * 1000) / 1000), x + barW / 2, y - 8);

      ctx.fillStyle = "rgba(170,179,207,0.95)";
      ctx.fillText(it.label, x + barW / 2, baseY + 16);
    });
  }

  function drawConfusionHeatmap(container, cm) {
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    const max = Math.max(1, tn, fp, fn, tp);

    container.innerHTML = `
      <div class="hm">
        <div class="hm-head"></div>
        <div class="hm-head">Pred Good</div>
        <div class="hm-head">Pred Bad</div>

        <div class="hm-head">Real Good</div>
        ${hmCell(tn, max, "TN")}
        ${hmCell(fp, max, "FP")}

        <div class="hm-head">Real Bad</div>
        ${hmCell(fn, max, "FN")}
        ${hmCell(tp, max, "TP")}
      </div>
    `;
  }

  function hmCell(value, max, tag) {
    const intensity = value / max;
    const bg = `rgba(110,231,183,${0.08 + intensity * 0.35})`;
    return `
      <div class="hm-cell" style="background:${bg}">
        <div class="hm-tag">${tag}</div>
        <div class="hm-val">${value}</div>
      </div>
    `;
  }
});