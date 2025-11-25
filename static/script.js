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
const metaCalendar = document.getElementById("meta-calendar");
const metaDate = document.getElementById("meta-date");
const metaCompany = document.getElementById("meta-company");
const metaSector = document.getElementById("meta-sector");
const agenticContent = document.getElementById("agentic-content");
const debugJson = document.getElementById("debug-json");
const kpiPred = document.getElementById("kpi-pred");
const kpiConf = document.getElementById("kpi-conf");
const kpiReturn = document.getElementById("kpi-return");
const kpiCost = document.getElementById("kpi-cost");
const detailCmp = document.getElementById("detail-cmp");
const detailHist = document.getElementById("detail-hist");
const detailPerf = document.getElementById("detail-perf");
const detailBaseline = document.getElementById("detail-baseline");
const detailTokens = document.getElementById("detail-tokens");
const detailCmpSummary = document.getElementById("detail-cmp-summary");
const detailHistSummary = document.getElementById("detail-hist-summary");
const detailPerfSummary = document.getElementById("detail-perf-summary");
const detailBaselineSummary = document.getElementById("detail-baseline-summary");
const detailTokensSummary = document.getElementById("detail-tokens-summary");

function setStatus(message, tone = "muted") {
  statusEl.textContent = message;
  statusEl.style.color = tone === "error" ? "#f87171" : "#9ca3af";
}

function renderResultsSkeleton() {
  const placeholder = Array.from({ length: 3 })
    .map(
      () => `<div class="skeleton-card">
        <div class="skeleton-line wide"></div>
        <div class="skeleton-line mid"></div>
      </div>`
    )
    .join("");
  resultsEl.innerHTML = placeholder;
}

async function fetchDatesForSymbol(symbol) {
  try {
    const res = await fetch(`/api/transcript-dates?symbol=${encodeURIComponent(symbol)}`);
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const data = await res.json();
    return (data || []).filter((d) => Number.isInteger(d.year) && Number.isInteger(d.quarter));
  } catch (err) {
    console.error("load dates failed for", symbol, err);
    return [];
  }
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
    title.textContent = `${item.symbol} - ${item.name || "Unnamed"}`;
    const meta = document.createElement("div");
    meta.className = "muted small";
    const avail = item.dates ? ` · ${item.dates.length} 筆季度` : "";
    meta.textContent = `${item.exchange || "N/A"} - ${item.currency || ""}${avail}`;

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
  renderResultsSkeleton();
  setStatus("搜尋中...");

  try {
    const res = await fetch(`/api/symbols?q=${encodeURIComponent(query)}`);
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const data = await res.json();

    // 只保留有季度資料的公司
    const enriched = await Promise.all(
      data.map(async (item) => {
        const dates = await fetchDatesForSymbol(item.symbol);
        return { ...item, dates };
      })
    );
    const filtered = enriched.filter((x) => x.dates && x.dates.length > 0);
    renderResults(filtered);
    setStatus(`找到 ${filtered.length} 筆有季度資料的結果`);
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
  metaCalendar.textContent = "-";
  metaDate.textContent = "-";
  agenticContent.innerHTML = '<p class="muted">尚未執行分析。</p>';
  debugJson.textContent = "{}";
  // 直接用已載入的季度資料，若沒有再補拉
  if (item.dates && item.dates.length) {
    availableDates = item.dates;
    populateDatesSelect();
  } else {
    fetchTranscriptDates(item.symbol);
  }
  // 收起其他結果
  resultsEl.innerHTML = "";
}

async function fetchTranscriptDates(symbol) {
  datesSelect.innerHTML = '<option value="">載入中...</option>';
  analyzeBtn.disabled = true;
  datesHint.textContent = "";
  try {
    availableDates = await fetchDatesForSymbol(symbol);

    if (availableDates.length === 0) {
      datesSelect.innerHTML = '<option value="">沒有可用的季度</option>';
      datesHint.textContent = "此公司尚無可用的逐字稿清單。";
      return;
    }

    populateDatesSelect();
  } catch (err) {
    console.error(err);
    datesHint.textContent = "";
    datesSelect.innerHTML = '<option value="">載入失敗</option>';
    setStatus(`載入季度失敗：${err.message}`, "error");
  }
}

function populateDatesSelect() {
  datesSelect.innerHTML = "";
  availableDates.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = `${d.year}|${d.quarter}`;
    const labelDate = d.date ? ` · ${d.date}` : "";
    const cal =
      d.calendar_year && d.calendar_quarter
        ? `CY${d.calendar_year} Q${d.calendar_quarter}`
        : `CY?`;
    opt.textContent = `${cal}${labelDate ? " · " + labelDate : ""}`;
    datesSelect.appendChild(opt);
  });
  datesHint.textContent = `共 ${availableDates.length} 筆季度資料。`;
  analyzeBtn.disabled = false;
}

function renderAgentic(result) {
  if (!result) {
    agenticContent.innerHTML = '<p class="muted">尚未執行分析。</p>';
    return;
  }
  const { prediction, confidence, summary, reasons, next_steps, metadata } = result;
  const engineLabel = "";
  const tokenUsage = (result.raw && result.raw.token_usage) || metadata?.token_usage || {};
  const formatReasonBody = (text) => {
    if (!text) return "<p>-</p>";
    const parts = String(text)
      .split(/\n{2,}/)
      .map((p) => p.trim())
      .filter(Boolean);
    return parts
      .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
      .join("") || `<p>${text}</p>`;
  };
  const reasonsMarkup =
    reasons && reasons.length
      ? `<h4>理由</h4><div class="control-row"><button class="btn-ghost" id="reasons-expand-all">展開全部</button><button class="btn-ghost" id="reasons-collapse-all">收合全部</button><button class="btn-ghost" id="reasons-copy">複製理由</button></div><div class="accordion">${reasons
          .map((r, idx) => {
            const summary = typeof r === "string" ? r.split(" ").slice(0, 8).join(" ") + (r.split(" ").length > 8 ? "..." : "") : "理由";
            return `
            <div class="accordion-item">
              <button class="accordion-header" type="button" aria-expanded="false" data-target="reason-${idx}">
                <span>${summary}</span>
                <span class="chevron">▼</span>
              </button>
              <div id="reason-${idx}" class="accordion-body" hidden>
                ${formatReasonBody(r)}
              </div>
            </div>`;
          })
          .join("")}</div>`
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

  // Wire up accordion toggles for reasons
  const reasonHeaders = agenticContent.querySelectorAll(".accordion-header");
  reasonHeaders.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const body = agenticContent.querySelector(`#${targetId}`);
      const expanded = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", String(!expanded));
      if (body) {
        if (expanded) {
          body.hidden = true;
          btn.querySelector(".chevron").style.transform = "rotate(-90deg)";
        } else {
          body.hidden = false;
          btn.querySelector(".chevron").style.transform = "rotate(0deg)";
        }
      }
    });
  });

  const expandAll = agenticContent.querySelector("#reasons-expand-all");
  const collapseAll = agenticContent.querySelector("#reasons-collapse-all");
  const copyBtn = agenticContent.querySelector("#reasons-copy");
  if (expandAll) {
    expandAll.addEventListener("click", () => {
      reasonHeaders.forEach((btn) => {
        const targetId = btn.getAttribute("data-target");
        const body = agenticContent.querySelector(`#${targetId}`);
        btn.setAttribute("aria-expanded", "true");
        if (body) {
          body.hidden = false;
          btn.querySelector(".chevron").style.transform = "rotate(0deg)";
        }
      });
    });
  }
  if (collapseAll) {
    collapseAll.addEventListener("click", () => {
      reasonHeaders.forEach((btn) => {
        const targetId = btn.getAttribute("data-target");
        const body = agenticContent.querySelector(`#${targetId}`);
        btn.setAttribute("aria-expanded", "false");
        if (body) {
          body.hidden = true;
          btn.querySelector(".chevron").style.transform = "rotate(-90deg)";
        }
      });
    });
  }
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const bodies = agenticContent.querySelectorAll(".accordion-body");
      const text = Array.from(bodies)
        .map((b) => b.textContent.trim())
        .filter(Boolean)
        .join("\n\n");
      try {
        await navigator.clipboard.writeText(text || "");
        setStatus("理由已複製");
      } catch (e) {
        setStatus("無法複製理由", "error");
      }
    });
  }
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
  agenticContent.innerHTML = `<div class="control-row"><span class="spinner"></span><span class="muted small">分析中...</span></div>`;
  debugJson.textContent = "{}";
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
    if (data.calendar_year && data.calendar_quarter) {
      metaCalendar.textContent = `CY${data.calendar_year} Q${data.calendar_quarter}`;
    } else {
      metaCalendar.textContent = "-";
    }
    metaDate.textContent = data.transcript_date || "-";
    metaCompany.textContent = data.context?.company || data.company || "-";
    metaSector.textContent = data.context?.sector || data.sector || "-";
    renderAgentic(data.agentic_result);
    debugJson.textContent = JSON.stringify(data, null, 2);

    // KPI + 詳細資訊
    const agent = data.agentic_result || {};
    const rawNotes = (agent.raw && agent.raw.notes) || {};
    const tokenUsage = data.token_usage || agent.raw?.token_usage || {};
    const postReturn = data.post_return_meta?.return ?? data.post_earnings_return;

    kpiPred.textContent = agent.prediction || "N/A";
    kpiConf.textContent =
      agent.confidence != null ? `${Math.round((agent.confidence || 0) * 100)}%` : "-";
    kpiReturn.textContent = postReturn != null ? `${(postReturn * 100).toFixed(2)}%` : "未計算";
    const cost = tokenUsage.cost_usd != null ? `$${tokenUsage.cost_usd.toFixed(4)}` : "N/A";
    kpiCost.textContent = `${cost}`;
    kpiReturn.classList.remove("pos", "neg");
    if (postReturn != null) {
      if (postReturn > 0) kpiReturn.classList.add("pos");
      else if (postReturn < 0) kpiReturn.classList.add("neg");
    }

    const clean = (val) => {
      if (!val) return null;
      const s = String(val).trim();
      return ["n/a", "na", "none"].includes(s.toLowerCase()) ? null : s;
    };
    detailCmp.textContent = clean(rawNotes.peers) || "尚未產生（可能缺少 Neo4j 資料）";
    detailHist.textContent = clean(rawNotes.past) || "尚未產生（可能缺少 Neo4j 資料）";
    detailPerf.textContent = clean(rawNotes.financials) || "尚未產生（可能缺少 Neo4j 資料）";
    detailBaseline.textContent = "Baseline / Sentiment 預留";
    const tokensToShow =
      tokenUsage.cost_usd != null ? { cost_usd: tokenUsage.cost_usd, ...tokenUsage } : tokenUsage;
    detailTokens.textContent = Object.keys(tokensToShow || {}).length
      ? JSON.stringify(tokensToShow, null, 2)
      : "N/A";

    const summarize = (txt) => {
      if (!txt) return "尚無資料";
      const words = String(txt).split(/\s+/);
      if (words.length <= 12) return txt;
      return words.slice(0, 12).join(" ") + "...";
    };
    detailCmpSummary.textContent = summarize(detailCmp.textContent);
    detailHistSummary.textContent = summarize(detailHist.textContent);
    detailPerfSummary.textContent = summarize(detailPerf.textContent);
    detailBaselineSummary.textContent = summarize(detailBaseline.textContent);
    detailTokensSummary.textContent = summarize(detailTokens.textContent);
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

// Collapsible debug JSON
const debugToggle = document.querySelector(".collapse-toggle");
const debugBody = document.getElementById("debug-json-wrap");
if (debugToggle && debugBody) {
  debugToggle.addEventListener("click", () => {
    const expanded = debugToggle.getAttribute("aria-expanded") === "true";
    debugToggle.setAttribute("aria-expanded", String(!expanded));
    const chevron = debugToggle.querySelector(".chevron");
    if (!expanded) {
      debugBody.hidden = false;
      chevron.style.transform = "rotate(0deg)";
    } else {
      debugBody.hidden = true;
      chevron.style.transform = "rotate(-90deg)";
    }
  });
  // start collapsed
  debugBody.hidden = true;
  const chevron = debugToggle.querySelector(".chevron");
  if (chevron) chevron.style.transform = "rotate(-90deg)";
}

// Init detail toggles
wireDetailToggles();

// 初始狀態
setStatus("輸入公司名稱或代號開始搜尋");
// Toggle detail sections
function wireDetailToggles() {
  document.querySelectorAll(".detail-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const body = document.getElementById(targetId);
      const expanded = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", String(!expanded));
      const chevron = btn.querySelector(".chevron");
      if (body) body.hidden = expanded;
      if (chevron) chevron.style.transform = expanded ? "rotate(-90deg)" : "rotate(0deg)";
    });
    // start collapsed
    btn.setAttribute("aria-expanded", "false");
    const chevron = btn.querySelector(".chevron");
    if (chevron) chevron.style.transform = "rotate(-90deg)";
    const targetId = btn.getAttribute("data-target");
    const body = document.getElementById(targetId);
    if (body) body.hidden = true;
  });
}
