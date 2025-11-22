let selectedSymbol = null;
let selectedCompany = "";
let availableDates = [];

const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const resultsEl = document.getElementById("results");
const statusEl = document.getElementById("status");
const selectedSymbolEl = document.getElementById("selected-symbol");
const datesSelect = document.getElementById("dates-select");
const datesHint = document.getElementById("dates-hint");
const analyzeBtn = document.getElementById("analyze-btn");
const metaSymbol = document.getElementById("meta-symbol");
const metaFiscal = document.getElementById("meta-fiscal");
const metaDate = document.getElementById("meta-date");
const agenticContent = document.getElementById("agentic-content");
const debugJson = document.getElementById("debug-json");

function setStatus(message, tone = "muted") {
  statusEl.textContent = message;
  statusEl.style.color = tone === "error" ? "#f87171" : "#9ca3af";
}

function renderResults(items) {
  resultsEl.innerHTML = "";
  if (!items || items.length === 0) {
    resultsEl.innerHTML = '<p class="muted small">找不到符合的公司，試試其他關鍵字。</p>';
    return;
  }

  items.forEach((item) => {
    const container = document.createElement("div");
    container.className = "result-item";

    const info = document.createElement("div");
    info.className = "info";
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = `${item.symbol} — ${item.name || "Unnamed"}`;
    const meta = document.createElement("div");
    meta.className = "muted small";
    meta.textContent = `${item.exchange || "N/A"} · ${item.currency || ""}`;

    info.appendChild(title);
    info.appendChild(meta);

    const selectBtn = document.createElement("button");
    selectBtn.textContent = "選取";
    selectBtn.onclick = () => onSelectSymbol(item);

    container.appendChild(info);
    container.appendChild(selectBtn);
    resultsEl.appendChild(container);
  });
}

async function searchSymbols() {
  const query = searchInput.value.trim();
  if (!query) {
    setStatus("請輸入要搜尋的公司名稱或代號");
    return;
  }
  setStatus("搜尋中...");
  renderResults([]);

  try {
    const res = await fetch(`/api/symbols?q=${encodeURIComponent(query)}`);
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const data = await res.json();
    renderResults(data);
    setStatus(`找到 ${data.length} 筆結果`);
  } catch (err) {
    console.error(err);
    setStatus(`搜尋失敗：${err.message}`, "error");
  }
}

function onSelectSymbol(item) {
  selectedSymbol = item.symbol;
  selectedCompany = item.name || item.symbol;
  selectedSymbolEl.textContent = `已選擇：${item.symbol} — ${selectedCompany}`;
  metaSymbol.textContent = item.symbol;
  metaFiscal.textContent = "-";
  metaDate.textContent = "-";
  agenticContent.innerHTML = '<p class="muted">尚未執行分析。</p>';
  debugJson.textContent = "{}";
  fetchTranscriptDates(item.symbol);
}

async function fetchTranscriptDates(symbol) {
  datesSelect.innerHTML = '<option value="">載入中...</option>';
  analyzeBtn.disabled = true;
  datesHint.textContent = "";
  try {
    const res = await fetch(`/api/transcript-dates?symbol=${encodeURIComponent(symbol)}`);
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const data = await res.json();
    availableDates = (data || []).filter(
      (d) => Number.isInteger(d.year) && Number.isInteger(d.quarter)
    );

    if (availableDates.length === 0) {
      datesSelect.innerHTML = '<option value="">沒有可用的季度</option>';
      datesHint.textContent = "此公司尚無可用的逐字稿清單。";
      return;
    }

    datesSelect.innerHTML = "";
    availableDates.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = `${d.year}|${d.quarter}`;
      const labelDate = d.date ? ` · ${d.date}` : "";
      opt.textContent = `FY${d.year} Q${d.quarter}${labelDate}`;
      datesSelect.appendChild(opt);
    });
    datesHint.textContent = `共 ${availableDates.length} 筆季度資料。`;
    analyzeBtn.disabled = false;
  } catch (err) {
    console.error(err);
    datesHint.textContent = "";
    datesSelect.innerHTML = '<option value="">載入失敗</option>';
    setStatus(`載入季度失敗：${err.message}`, "error");
  }
}

function renderAgentic(result) {
  if (!result) {
    agenticContent.innerHTML = '<p class="muted">尚未執行分析。</p>';
    return;
  }
  const { prediction, confidence, summary, reasons, next_steps, metadata } = result;
  const engineLabel = (metadata && metadata.engine) || "Agentic RAG";
  const reasonsMarkup =
    reasons && reasons.length
      ? `<h4>理由</h4><ul>${reasons.map((r) => `<li>${r}</li>`).join("")}</ul>`
      : "";
  const nextStepsMarkup =
    next_steps && next_steps.length
      ? `<h4>後續動作</h4><ul>${next_steps.map((r) => `<li>${r}</li>`).join("")}</ul>`
      : "";
  const confidenceBadge = confidence != null ? `<span class="badge">信心度 ${(confidence * 100).toFixed(0)}%</span>` : "";

  agenticContent.innerHTML = `
    <div class="muted small">${engineLabel}</div>
    <h4>${prediction || "N/A"} ${confidenceBadge}</h4>
    <p>${summary || ""}</p>
    ${reasonsMarkup}
    ${nextStepsMarkup}
  `;
}

async function runAnalysis() {
  if (!selectedSymbol) {
    setStatus("請先選擇公司", "error");
    return;
  }
  if (!datesSelect.value) {
    setStatus("請先選擇年度與季度", "error");
    return;
  }
  const [yearStr, quarterStr] = datesSelect.value.split("|");
  const payload = {
    symbol: selectedSymbol,
    year: Number(yearStr),
    quarter: Number(quarterStr),
  };

  analyzeBtn.disabled = true;
  const originalLabel = analyzeBtn.textContent;
  analyzeBtn.textContent = "分析中...";
  setStatus("正在呼叫後端分析...");

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const data = await res.json();
    metaSymbol.textContent = data.symbol || selectedSymbol;
    metaFiscal.textContent = `FY${data.year} Q${data.quarter}`;
    metaDate.textContent = data.transcript_date || "-";
    renderAgentic(data.agentic_result);
    debugJson.textContent = JSON.stringify(data, null, 2);
    setStatus("分析完成");
  } catch (err) {
    console.error(err);
    setStatus(`分析失敗：${err.message}`, "error");
  } finally {
    analyzeBtn.disabled = availableDates.length === 0;
    analyzeBtn.textContent = originalLabel;
  }
}

searchBtn.addEventListener("click", searchSymbols);
searchInput.addEventListener("keyup", (e) => {
  if (e.key === "Enter") searchSymbols();
});
analyzeBtn.addEventListener("click", runAnalysis);

// 初始狀態
setStatus("輸入公司名稱或代號開始搜尋");
