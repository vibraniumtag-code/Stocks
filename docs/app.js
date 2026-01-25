/* docs/app.js
   Portfolio Manager Dashboard
   - Loads docs/portfolio_plan.csv
   - Shows moves with ACK + EXECUTE
   - EXECUTE dispatches GitHub Action workflow_dispatch to apply_move.yml
   - BUY execution prompts for option_entry_price + underlying_entry_price (required)

   IMPORTANT:
   - Set WORKFLOW_FILE to match your workflow filename in .github/workflows/
   - This stores token in Safari localStorage (device only)
*/

const WORKFLOW_FILE = "apply_move.yml"; // .github/workflows/apply_move.yml
const BRANCH = "main";
const PLAN_CSV = "portfolio_plan.csv"; // served from /docs

const els = {
  owner: document.getElementById("owner"),
  repo: document.getElementById("repo"),
  token: document.getElementById("token"),
  saveBtn: document.getElementById("saveBtn"),
  loadBtn: document.getElementById("loadBtn"),
  planBody: document.getElementById("planBody"),
  status: document.getElementById("status"),
  log: document.getElementById("log"),
};

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function log(msg) {
  els.log.textContent = `${new Date().toISOString()}  ${msg}\n` + els.log.textContent;
}

function setStatus(msg) {
  els.status.textContent = msg;
}

function saveSettings() {
  localStorage.setItem("pm_owner", els.owner.value.trim());
  localStorage.setItem("pm_repo", els.repo.value.trim());
  localStorage.setItem("pm_token", els.token.value || "");
  setStatus("Saved ✅");
  log("Saved owner/repo/token in this browser (localStorage).");
}

function loadSettings() {
  els.owner.value = localStorage.getItem("pm_owner") || "";
  els.repo.value = localStorage.getItem("pm_repo") || "";
  els.token.value = localStorage.getItem("pm_token") || "";
}

/** Robust CSV parser (handles quotes + commas) */
function parseCSV(text) {
  const rows = [];
  let row = [], field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const n = text[i + 1];

    if (inQuotes) {
      if (c === '"' && n === '"') { field += '"'; i++; }
      else if (c === '"') { inQuotes = false; }
      else { field += c; }
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n") {
        row.push(field); field = "";
        if (row.some(x => x.trim() !== "")) rows.push(row);
        row = [];
      } else if (c === "\r") {
        // ignore
      } else field += c;
    }
  }
  if (field.length > 0 || row.length) { row.push(field); rows.push(row); }
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
  if (t === "SELL") return `<span class="pill pill-sell">SELL</span>`;
  if (t === "BUY") return `<span class="pill pill-buy">BUY</span>`;
  if (t === "ADD") return `<span class="pill pill-add">ADD</span>`;
  if (t === "HOLD") return `<span class="pill pill-hold">HOLD</span>`;
  return `<span class="pill">${escapeHtml(type || "")}</span>`;
}

function numCell(v) {
  const x = (v ?? "").toString().trim();
  return `<td class="num">${escapeHtml(x)}</td>`;
}

function textCell(v, mono=false) {
  const cls = mono ? "mono" : "";
  return `<td class="${cls}">${escapeHtml((v ?? "").toString())}</td>`;
}

function getMoveTypeFromRow(row) {
  return (row["Type"] || "").trim().toUpperCase();
}

function buildMoveId(row, idx) {
  const t = (row["Ticker"] || "").trim().toUpperCase();
  const ty = (row["Type"] || "").trim().toUpperCase();
  const opt = (row["Option"] || "").trim();
  return `${Date.now()}_${idx}_${ty}_${t}_${opt}`.replaceAll(" ", "_").slice(0, 140);
}

async function loadPlan() {
  setStatus("Loading plan…");
  els.planBody.innerHTML = `<tr><td colspan="12" style="padding:14px;">Loading…</td></tr>`;

  try {
    const url = `${PLAN_CSV}?ts=${Date.now()}`; // cache busting
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to load plan: ${res.status}`);
    const text = await res.text();
    const rows = toObjects(text);

    setStatus(`Loaded rows: ${rows.length}`);
    log(`Loaded plan rows=${rows.length} from docs/${PLAN_CSV}`);
    renderTable(rows);
  } catch (e) {
    setStatus("Load failed ❌");
    els.planBody.innerHTML =
      `<tr><td colspan="12" style="padding:14px;color:#991b1b;">${escapeHtml(e.message)}</td></tr>`;
    log(`ERROR: ${e.message}`);
  }
}

function renderTable(rows) {
  if (!rows.length) {
    els.planBody.innerHTML = `<tr><td colspan="12" style="padding:14px;">Plan is empty.</td></tr>`;
    return;
  }

  const html = rows.map((r, idx) => {
    const type = getMoveTypeFromRow(r);
    const ticker = (r["Ticker"] || "").trim().toUpperCase();

    // IMPORTANT: for BUY rows, Option may be blank in your plan file
    const optionName = (r["Option"] || "").trim();

    const sellC = (r["SellContracts"] || "").trim();
    const addC = (r["AddContracts"] || "").trim();
    const buyC = (r["BuyContracts"] || "").trim();

    const strategy = (r["Strategy"] || "").trim();
    const expiry = (r["Expiry"] || "").trim();
    const optionSymbol = (r["OptionSymbol"] || "").trim();
    const reason = (r["Reason"] || "").trim();

    const moveId = buildMoveId(r, idx);

    const ackBtn = `<button class="ok" data-action="ack" data-moveid="${escapeHtml(moveId)}">ACK</button>`;

    const executeDisabled = (type === "HOLD" || type === "");
    const execBtn = executeDisabled
      ? `<button disabled>EXECUTE</button>`
      : `<button class="warn" data-action="execute" data-moveid="${escapeHtml(moveId)}">EXECUTE · Update CSV</button>`;

    return `
      <tr data-idx="${idx}"
          data-type="${escapeHtml(type)}"
          data-ticker="${escapeHtml(ticker)}"
          data-option="${escapeHtml(optionName)}"
          data-sell="${escapeHtml(sellC)}"
          data-add="${escapeHtml(addC)}"
          data-buy="${escapeHtml(buyC)}"
          data-strategy="${escapeHtml(strategy)}"
          data-expiry="${escapeHtml(expiry)}"
          data-symbol="${escapeHtml(optionSymbol)}">
        <td>${pill(type)}</td>
        ${textCell(ticker, true)}
        ${textCell(optionName || "-", true)}
        ${numCell(r["ContractsHeld"] || "")}
        ${numCell(sellC || "0")}
        ${numCell(addC || "0")}
        ${numCell(buyC || "0")}
        ${textCell(strategy || "-")}
        ${textCell(expiry || "-", true)}
        ${textCell(optionSymbol || "-", true)}
        ${textCell(reason || "-")}
        <td>${ackBtn} ${execBtn}</td>
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

  const moveId = btn.getAttribute("data-moveid") || `move_${Date.now()}`;
  const type = (tr.getAttribute("data-type") || "").toUpperCase();
  const ticker = tr.getAttribute("data-ticker") || "";
  let optionName = tr.getAttribute("data-option") || "";

  const sellContracts = tr.getAttribute("data-sell") || "0";
  const addContracts = tr.getAttribute("data-add") || "0";
  const buyContracts = tr.getAttribute("data-buy") || "0";

  const strategy = tr.getAttribute("data-strategy") || "";
  const expiry = tr.getAttribute("data-expiry") || "";
  const optionSymbol = tr.getAttribute("data-symbol") || "";

  if (action === "ack") {
    log(`✅ ACK: ${type} ${ticker} ${optionName || ""}`);
    alert(`ACK only (no repo changes).\n\n${type} ${ticker}\n${optionName || ""}`);
    return;
  }

  // EXECUTE
  const owner = els.owner.value.trim();
  const repo = els.repo.value.trim();
  const token = els.token.value;

  if (!owner || !repo) {
    alert("Please set owner + repo.");
    return;
  }
  if (!token) {
    alert("No token saved. Paste your PAT and click Save Token first.");
    return;
  }

  // If BUY and Option is blank (common in your plan), that's OK for workflow (it can build),
  // BUT we still want to preserve option_name when possible. If blank, let workflow build it.
  // For SELL/ADD, option_name MUST be present.
  if ((type === "SELL" || type === "ADD") && (!optionName || optionName.trim() === "")) {
    optionName = prompt("Missing option_name. Paste it exactly as in positions.csv (e.g., NFLX 2026-02-20 P 88):", "");
    if (!optionName || optionName.trim() === "") {
      alert("SELL/ADD requires option_name.");
      return;
    }
  }

  // BUY requires entry prices (actual fills)
  let optionEntryPrice = "";
  let underlyingEntryPrice = "";

  if (type === "BUY") {
    optionEntryPrice = prompt("Enter OPTION fill price (avg fill). مثال: 4.52", "");
    if (optionEntryPrice === null) return;
    optionEntryPrice = optionEntryPrice.trim().replace("$", "");
    if (!optionEntryPrice || isNaN(Number(optionEntryPrice)) || Number(optionEntryPrice) <= 0) {
      alert("BUY requires a valid option fill price > 0");
      return;
    }

    underlyingEntryPrice = prompt("Enter UNDERLYING price at execution. مثال: 135.12", "");
    if (underlyingEntryPrice === null) return;
    underlyingEntryPrice = underlyingEntryPrice.trim().replace("$", "");
    if (!underlyingEntryPrice || isNaN(Number(underlyingEntryPrice)) || Number(underlyingEntryPrice) <= 0) {
      alert("BUY requires a valid underlying price > 0");
      return;
    }
  }

  const ok = confirm(
    `EXECUTE will update positions.csv via GitHub Actions.\n\n` +
    `${type} ${ticker}\n${optionName || "(option_name will be built from symbol)"}\n\nProceed?`
  );
  if (!ok) return;

  btn.disabled = true;

  try {
    await dispatchWorkflow({
      owner, repo, token,
      inputs: {
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

        // BUY required fields
        option_entry_price: optionEntryPrice || "",
        underlying_entry_price: underlyingEntryPrice || "",
      }
    });

    log(`🚀 Dispatch OK: ${type} ${ticker} move_id=${moveId}`);
    alert("✅ Dispatched. Check Actions tab for status. If successful, positions.csv will be committed.");
  } catch (err) {
    log(`❌ Dispatch failed: ${err.message}`);
    alert(`❌ Dispatch failed:\n${err.message}`);
  } finally {
    btn.disabled = false;
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

// Wire up
els.saveBtn.addEventListener("click", saveSettings);
els.loadBtn.addEventListener("click", loadPlan);

loadSettings();
loadPlan();
log("Dashboard ready.");