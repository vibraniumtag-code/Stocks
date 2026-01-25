/* docs/app.js — prettier dashboard + lock EXECUTE after click */
const VERSION = "2026-01-25-DASHBOARD-NICE-V1";

const WORKFLOW_FILE = "apply_move.yml";
const BRANCH = "main";
const PLAN_CSV = "portfolio_plan.csv";

const els = {
  owner: document.getElementById("owner"),
  repo: document.getElementById("repo"),
  token: document.getElementById("token"),
  saveBtn: document.getElementById("saveBtn"),
  loadBtn: document.getElementById("loadBtn"),
  resetBtn: document.getElementById("resetBtn"),
  planBody: document.getElementById("planBody"),
  status: document.getElementById("status"),
  log: document.getElementById("log"),
};

const EXEC_STORE_KEY = "pm_executed_moves_v1";

function log(msg) {
  if (!els.log) return;
  els.log.textContent = `${new Date().toISOString()}  ${msg}\n` + els.log.textContent;
}
function setStatus(msg) {
  if (!els.status) return;
  els.status.textContent = msg;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

/** ---------- local persistence of executed moves ---------- */
function loadExecutedMap() {
  try {
    return JSON.parse(localStorage.getItem(EXEC_STORE_KEY) || "{}") || {};
  } catch {
    return {};
  }
}
function saveExecutedMap(map) {
  localStorage.setItem(EXEC_STORE_KEY, JSON.stringify(map || {}));
}
function markExecuted(moveId) {
  const m = loadExecutedMap();
  m[moveId] = { ts: Date.now() };
  saveExecutedMap(m);
}
function isExecuted(moveId) {
  const m = loadExecutedMap();
  return !!m[moveId];
}
function resetExecuted() {
  localStorage.removeItem(EXEC_STORE_KEY);
}

/** ---------- settings ---------- */
function loadSettings() {
  if (els.owner) els.owner.value = localStorage.getItem("pm_owner") || "";
  if (els.repo) els.repo.value = localStorage.getItem("pm_repo") || "";
  if (els.token) els.token.value = localStorage.getItem("pm_token") || "";
}
function saveSettings() {
  const o = (els.owner?.value || "").trim();
  const r = (els.repo?.value || "").trim();
  const t = (els.token?.value || "");
  localStorage.setItem("pm_owner", o);
  localStorage.setItem("pm_repo", r);
  localStorage.setItem("pm_token", t);
  setStatus("Saved ✅");
  log(`Saved owner/repo/token. owner=${o} repo=${r}`);
}

/** Robust CSV parsing */
function parseCSV(text) {
  const rows = [];
  let row = [], field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const n = text[i + 1];

    if (inQuotes) {
      if (c === '"' && n === '"') { field += '"'; i++; }
      else if (c === '"') inQuotes = false;
      else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n") {
        row.push(field); field = "";
        if (row.some(x => x.trim() !== "")) rows.push(row);
        row = [];
      } else if (c === "\r") { /* ignore */ }
      else field += c;
    }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function toObjects(csvText) {
  const rows = parseCSV(csvText);
  if (!rows.length) return [];
  const header = rows[0].map(h => (h || "").trim());
  const out = [];
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const obj = {};
    for (let j = 0; j < header.length; j++) obj[header[j]] = (r[j] ?? "").trim();
    if (Object.values(obj).some(v => (v || "").trim() !== "")) out.push(obj);
  }
  return out;
}

function pill(type) {
  const t = (type || "").toUpperCase();
  if (t === "SELL") return `<span class="pill sell">SELL</span>`;
  if (t === "BUY") return `<span class="pill buy">BUY</span>`;
  if (t === "ADD") return `<span class="pill add">ADD</span>`;
  if (t === "HOLD") return `<span class="pill hold">HOLD</span>`;
  return `<span class="pill">${escapeHtml(type || "")}</span>`;
}

function getMoveTypeFromRow(row) {
  return (row["Type"] || "").trim().toUpperCase();
}

function buildMoveId(row, idx) {
  const t = (row["Ticker"] || "").trim().toUpperCase();
  const ty = (row["Type"] || "").trim().toUpperCase();
  const opt = (row["Option"] || "").trim();
  const sym = (row["OptionSymbol"] || "").trim();
  return `${idx + 1}-${ty}-${t}-${(sym || opt || "NA")}`.replaceAll(" ", "_").slice(0, 120);
}

function askInt(label, defaultVal) {
  const raw = prompt(`${label}\n(Default: ${defaultVal})`, String(defaultVal ?? "").trim());
  if (raw === null) return null;
  const v = Number(String(raw).trim());
  if (!Number.isFinite(v) || v < 0) return NaN;
  return Math.floor(v);
}
function askFloat(label) {
  const raw = prompt(label, "");
  if (raw === null) return null;
  const s = String(raw).trim().replace("$", "");
  const v = Number(s);
  if (!Number.isFinite(v) || v <= 0) return NaN;
  return v;
}

async function loadPlan() {
  setStatus("Loading plan…");
  if (els.planBody) els.planBody.innerHTML = `<tr><td colspan="12" style="padding:14px;">Loading…</td></tr>`;

  try {
    const url = `${PLAN_CSV}?ts=${Date.now()}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to load plan: ${res.status}`);
    const text = await res.text();
    const rows = toObjects(text);
    setStatus(`Loaded rows: ${rows.length}`);
    log(`Loaded plan rows=${rows.length} from docs/${PLAN_CSV}`);
    renderTable(rows);
  } catch (e) {
    setStatus("Load failed ❌");
    if (els.planBody) {
      els.planBody.innerHTML =
        `<tr><td colspan="12" style="padding:14px;color:#fecaca;">${escapeHtml(e.message)}</td></tr>`;
    }
    log(`ERROR: ${e.message}`);
  }
}

function renderTable(rows) {
  if (!els.planBody) return;

  if (!rows.length) {
    els.planBody.innerHTML = `<tr><td colspan="12" style="padding:14px;">Plan is empty.</td></tr>`;
    return;
  }

  // Put actionable rows first
  rows.sort((a, b) => {
    const ta = getMoveTypeFromRow(a);
    const tb = getMoveTypeFromRow(b);
    const pa = (ta === "HOLD" || ta === "") ? 1 : 0;
    const pb = (tb === "HOLD" || tb === "") ? 1 : 0;
    if (pa !== pb) return pa - pb;
    return String((a["Ticker"] || "")).localeCompare(String((b["Ticker"] || "")));
  });

  const html = rows.map((r, idx) => {
    const type = getMoveTypeFromRow(r);
    const ticker = (r["Ticker"] || "").trim().toUpperCase();
    const optionName = (r["Option"] || "").trim();
    const strategy = (r["Strategy"] || "").trim();
    const expiry = (r["Expiry"] || "").trim();
    const optionSymbol = (r["OptionSymbol"] || "").trim();
    const reason = (r["Reason"] || "").trim();

    const moveId = buildMoveId(r, idx);
    const done = isExecuted(moveId);

    const sellC = (r["SellContracts"] || "0").trim();
    const addC  = (r["AddContracts"] || "0").trim();
    const buyC  = (r["BuyContracts"] || "0").trim();

    const executeDisabled = done || type === "HOLD" || type === "";
    const execLabel = done ? `Executed ✅` : `EXECUTE · Update CSV`;

    return `
      <tr class="${done ? "row-done" : ""}"
          data-moveid="${escapeHtml(moveId)}"
          data-type="${escapeHtml(type)}"
          data-ticker="${escapeHtml(ticker)}"
          data-option="${escapeHtml(optionName)}"
          data-sell="${escapeHtml(sellC)}"
          data-add="${escapeHtml(addC)}"
          data-buy="${escapeHtml(buyC)}"
          data-strategy="${escapeHtml(strategy)}"
          data-expiry="${escapeHtml(expiry)}"
          data-symbol="${escapeHtml(optionSymbol)}"
      >
        <td>${pill(type)}</td>
        <td class="mono">${escapeHtml(ticker)}</td>
        <td class="mono wraptext">${escapeHtml(optionName || "-")}</td>
        <td class="num">${escapeHtml(r["ContractsHeld"] || "")}</td>
        <td class="num">${escapeHtml(sellC)}</td>
        <td class="num">${escapeHtml(addC)}</td>
        <td class="num">${escapeHtml(buyC)}</td>
        <td>${escapeHtml(strategy || "-")}</td>
        <td class="mono">${escapeHtml(expiry || "-")}</td>
        <td class="mono">${escapeHtml(optionSymbol || "-")}</td>
        <td class="muted wraptext">${escapeHtml(reason || "-")}</td>
        <td>
          <div class="act">
            <button class="btnmini btn-ack" data-action="ack">ACK</button>
            <button class="btnmini btn-exec" data-action="execute" ${executeDisabled ? "disabled" : ""}>
              ${done ? "Executed ✅" : "EXECUTE · Update CSV"}
            </button>
            ${done ? `<span class="doneTag">Locked</span>` : ``}
          </div>
        </td>
      </tr>
    `;
  }).join("");

  els.planBody.innerHTML = html;

  els.planBody.querySelectorAll("button[data-action]").forEach(btn => {
    btn.addEventListener("click", onActionClick);
  });
}

async function onActionClick(e) {
  const btn = e.currentTarget;
  const action = btn.getAttribute("data-action");
  const tr = btn.closest("tr");
  if (!tr) return;

  const moveId = tr.getAttribute("data-moveid") || `move_${Date.now()}`;
  const type = (tr.getAttribute("data-type") || "").toUpperCase();
  const ticker = (tr.getAttribute("data-ticker") || "").toUpperCase();
  let optionName = tr.getAttribute("data-option") || "";

  let sellContracts = tr.getAttribute("data-sell") || "0";
  let addContracts  = tr.getAttribute("data-add") || "0";
  let buyContracts  = tr.getAttribute("data-buy") || "0";

  const strategy = tr.getAttribute("data-strategy") || "";
  const expiry = tr.getAttribute("data-expiry") || "";
  const optionSymbol = tr.getAttribute("data-symbol") || "";

  if (action === "ack") {
    log(`✅ ACK: ${type} ${ticker} ${optionName || ""}`);
    alert(`ACK only (no repo changes).\n\n${type} ${ticker}\n${optionName || ""}`);
    return;
  }

  // EXECUTE
  const owner = (els.owner?.value || "").trim();
  const repo = (els.repo?.value || "").trim();
  const token = els.token?.value || "";

  if (!owner || !repo) return alert("Please set owner + repo.");
  if (!token) return alert("No token saved. Paste your PAT and click Save Token first.");

  // Contract overrides (better real-life UX)
  if (type === "SELL") {
    const v = askInt(`SELL ${ticker}\nHow many contracts did you actually SELL?`, sellContracts || "0");
    if (v === null) return;
    if (Number.isNaN(v) || v <= 0) return alert("SELL contracts must be > 0");
    sellContracts = String(v);
  } else if (type === "ADD") {
    const v = askInt(`ADD ${ticker}\nHow many contracts did you actually ADD?`, addContracts || "0");
    if (v === null) return;
    if (Number.isNaN(v) || v <= 0) return alert("ADD contracts must be > 0");
    addContracts = String(v);
  } else if (type === "BUY") {
    const v = askInt(`BUY ${ticker}\nHow many contracts did you actually BUY?`, buyContracts || "0");
    if (v === null) return;
    if (Number.isNaN(v) || v <= 0) return alert("BUY contracts must be > 0");
    buyContracts = String(v);
  }

  // SELL/ADD requires option_name
  if ((type === "SELL" || type === "ADD") && (!optionName || optionName.trim() === "")) {
    optionName = prompt("Missing option_name.\nPaste exactly as in positions.csv\nExample: NFLX 2026-02-20 P 88", "");
    if (!optionName || optionName.trim() === "") return alert("SELL/ADD requires option_name.");
  }

  // BUY needs actual fills
  let optionEntryPrice = "";
  let underlyingEntryPrice = "";
  if (type === "BUY") {
    const op = askFloat("Enter OPTION fill price (avg fill). Example: 4.52");
    if (op === null) return;
    if (Number.isNaN(op)) return alert("BUY requires a valid option fill price > 0");
    optionEntryPrice = String(op);

    const up = askFloat("Enter UNDERLYING price at execution. Example: 135.12");
    if (up === null) return;
    if (Number.isNaN(up)) return alert("BUY requires a valid underlying price > 0");
    underlyingEntryPrice = String(up);
  }

  const ok = confirm(
    `EXECUTE will dispatch GitHub Action to update positions.csv.\n\n` +
    `${type} ${ticker}\n` +
    `${optionName || "(will be built from option_symbol)"}\n\n` +
    `Counts: SELL=${sellContracts} ADD=${addContracts} BUY=${buyContracts}\n` +
    (type === "BUY" ? `Prices: option=${optionEntryPrice} underlying=${underlyingEntryPrice}\n\n` : "\n") +
    `Proceed?`
  );
  if (!ok) return;

  // UI: lock + spinner immediately
  const oldHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>&nbsp;Dispatching…`;

  try {
    const inputs = {
      move_id: moveId,
      move_type: type,
      ticker: ticker,
      option_name: optionName || "",
      sell_contracts: sellContracts || "0",
      add_contracts: addContracts || "0",
      buy_contracts: buyContracts || "0",
      strategy: strategy || "",
      expiry: expiry || "",
      option_symbol: optionSymbol || "",
      option_entry_price: optionEntryPrice || "",
      underlying_entry_price: underlyingEntryPrice || "",
    };

    log(`DISPATCH inputs=${JSON.stringify(inputs)}`);
    await dispatchWorkflow({ owner, repo, token, inputs });

    // Mark executed locally (dims row + locks)
    markExecuted(moveId);
    tr.classList.add("row-done");

    btn.innerHTML = "Executed ✅";
    btn.disabled = true;

    setStatus(`Dispatched ✅ ${type} ${ticker}`);
    log(`🚀 Dispatch OK: ${type} ${ticker} move_id=${moveId}`);
    alert("✅ Dispatched. Check Actions tab for status. If successful, positions.csv will be committed.");
  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = oldHtml;
    setStatus("Dispatch failed ❌");
    log(`❌ Dispatch failed: ${err.message}`);
    alert(`❌ Dispatch failed:\n${err.message}`);
  }
}

async function dispatchWorkflow({ owner, repo, token, inputs }) {
  const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${WORKFLOW_FILE}/dispatches`;
  const body = { ref: BRANCH, inputs };

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": `token ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      msg = `${res.status}: ${JSON.stringify(j)}`;
    } catch {}
    throw new Error(msg);
  }
}

/** wiring */
if (els.saveBtn) els.saveBtn.addEventListener("click", saveSettings);
if (els.loadBtn) els.loadBtn.addEventListener("click", loadPlan);
if (els.resetBtn) els.resetBtn.addEventListener("click", () => {
  if (!confirm("Reset all local Executed/Locked flags on this device?")) return;
  resetExecuted();
  loadPlan();
  setStatus("Reset executed flags ✅");
  log("Reset executed flags (localStorage).");
});

loadSettings();
loadPlan();
log(`Dashboard ready. VERSION=${VERSION}`);