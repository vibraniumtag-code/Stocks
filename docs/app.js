/* docs/app.js
   Dashboard loader for portfolio_plan.csv (in SAME folder as index.html)
   Fixes: 0 rows issue by normalizing headers + handling mixed row shapes (SELL/HOLD/BUY)

   ✅ Put portfolio_plan.csv in /docs next to index.html
   ✅ Serve via GitHub Pages OR locally via:  cd docs && python -m http.server 8000
*/

// ====== CONFIG: set these ======
const OWNER = "vibraniumtag-code";     // e.g. "johnsmith" or "my-org"
const REPO  = "Stocks";        // e.g. "turtle-trader"
const BRANCH = "main";
const WORKFLOW_FILE = "apply_move.yml";  // .github/workflows/apply_move.yml
const PLAN_PATH = "./portfolio_plan.csv"; // IMPORTANT: same folder as index.html
// ===============================

/* If your apply_move.yml expects different input names,
   update buildDispatchInputs() below accordingly.
*/

// ---------------- Helpers ----------------
const $ = (sel) => document.querySelector(sel);

function normKey(k) {
  return String(k || "").trim().toLowerCase();
}
function clean(v) {
  return (v ?? "").toString().trim();
}
function toNum(v) {
  const n = Number(String(v ?? "").trim());
  return Number.isFinite(n) ? n : null;
}
function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function parseCSV(text) {
  // Simple CSV parser that supports quoted fields with commas.
  // Assumes first row is header.
  const rows = [];
  let i = 0, field = "", row = [];
  let inQuotes = false;

  const pushField = () => { row.push(field); field = ""; };
  const pushRow = () => {
    // ignore empty trailing row
    if (row.length === 1 && row[0] === "") { row = []; return; }
    rows.push(row);
    row = [];
  };

  while (i < text.length) {
    const c = text[i];

    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuotes = false; i++; continue;
      }
      field += c; i++; continue;
    }

    if (c === '"') { inQuotes = true; i++; continue; }
    if (c === ",") { pushField(); i++; continue; }
    if (c === "\r") { i++; continue; }
    if (c === "\n") { pushField(); pushRow(); i++; continue; }

    field += c; i++;
  }
  pushField();
  pushRow();

  if (!rows.length) return [];

  const header = rows[0].map((h) => normKey(h));
  const out = [];

  for (let r = 1; r < rows.length; r++) {
    const vals = rows[r];
    const obj = {};
    for (let c = 0; c < header.length; c++) {
      obj[header[c]] = clean(vals[c] ?? "");
    }
    out.push(obj);
  }
  return out;
}

function normalizePlanRows(rows) {
  return rows
    .map((r, idx) => {
      // Normalize type + ticker names (your CSV uses Type/Ticker -> type/ticker after parsing)
      const type = (r.type || "").toUpperCase();
      const ticker = (r.ticker || "").toUpperCase();

      // Two row shapes:
      // SELL/HOLD/ADD: has option, contractsheld, sellcontracts, addcontracts...
      // BUY: has strategy, expiry, optionsymbol, optionlast, buycontracts...
      const option = r.option || ""; // blank for BUY in your file
      const optionSymbol = r.optionsymbol || "";
      const expiry = r.expiry || "";
      const strategy = r.strategy || "";
      const reco = (r.recommendation || "").toUpperCase();

      // Numbers often come as "4.0" — keep as numeric-friendly
      const contractsHeld = toNum(r.contractsheld);
      const sellContracts = toNum(r.sellcontracts);
      const addContracts  = toNum(r.addcontracts);
      const buyContracts  = toNum(r.buycontracts);

      // Create stable move id
      const moveId = `${idx}-${type}-${ticker}-${option || optionSymbol || ""}`;

      return {
        _idx: idx,
        moveId,
        type,
        ticker,
        recommendation: reco,
        option,
        strategy,
        expiry,
        optionSymbol,
        contractsHeld,
        sellContracts,
        addContracts,
        buyContracts,
        optionLast: r.optionlast || "",
        estCostTotal: r.estcosttotal || "",
        positionValue: r.positionvalue || "",
        optionMark: r.optionmark || "",
        reason: r.reason || "",
        pyramidReason: r.pyramidreason || "",
      };
    })
    .filter((r) => r.ticker && r.type); // IMPORTANT: do NOT require option (BUY rows have blank option)
}

function getToken() {
  return localStorage.getItem("gh_pat") || "";
}
function setToken(tok) {
  localStorage.setItem("gh_pat", tok);
}
function clearToken() {
  localStorage.removeItem("gh_pat");
}

function fmtNum(n) {
  if (n === null || n === undefined) return "";
  if (!Number.isFinite(n)) return "";
  return String(n);
}

function badge(html, kind = "neutral") {
  const styles = {
    ok: "background:#ecfdf5;border:1px solid #d1fae5;color:#065f46;",
    warn: "background:#fffbeb;border:1px solid #fde68a;color:#92400e;",
    sell: "background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;",
    danger: "background:#fef2f2;border:1px solid #fecaca;color:#991b1b;",
    neutral: "background:#f3f4f6;border:1px solid #e5e7eb;color:#374151;",
  }[kind] || styles.neutral;

  return `<span style="display:inline-block;padding:2px 8px;border-radius:999px;font-weight:800;font-size:11px;${styles}">${html}</span>`;
}

function buildDispatchInputs(r) {
  // Map plan row -> workflow_dispatch inputs
  // Adjust keys if your apply_move.yml uses different input names
  return {
    move_id: r.moveId,
    move_type: r.type,            // SELL / HOLD / ADD / BUY
    ticker: r.ticker,

    // For existing positions actions:
    option_name: r.option || "",
    sell_contracts: r.sellContracts !== null ? String(r.sellContracts) : "0",
    add_contracts:  r.addContracts  !== null ? String(r.addContracts)  : "0",

    // For buy actions (optional):
    strategy: r.strategy || "",
    expiry: r.expiry || "",
    option_symbol: r.optionSymbol || "",
    buy_contracts: r.buyContracts !== null ? String(r.buyContracts) : "0",
  };
}

async function dispatchMove(r) {
  const token = getToken();
  if (!token) throw new Error("Missing token. Paste your fine-grained PAT and click Save Token.");

  const url = `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${encodeURIComponent(WORKFLOW_FILE)}/dispatches`;

  const payload = {
    ref: BRANCH,
    inputs: buildDispatchInputs(r),
  };

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`Dispatch failed (${res.status}): ${txt || res.statusText}`);
  }
}

// ---------------- UI Rendering ----------------
function ensureBasicUI() {
  // If your index.html already has these, we won't overwrite.
  if (!$("#app")) {
    document.body.innerHTML = `
      <div style="max-width:1100px;margin:0 auto;padding:16px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;">
        <div id="app"></div>
      </div>`;
  }
}

function render(rows) {
  const total = rows.length;
  const actions = rows.filter(r => r.recommendation && r.recommendation !== "HOLD").length;

  const sells = rows.filter(r => r.type === "SELL").length;
  const holds = rows.filter(r => r.type === "HOLD").length;
  const buys  = rows.filter(r => r.type === "BUY").length;
  const adds  = rows.filter(r => r.type === "ADD").length;

  const tokenSet = !!getToken();

  // Sort: actionable first, then by type/ticker
  const sorted = [...rows].sort((a,b) => {
    const aAct = (a.recommendation && a.recommendation !== "HOLD") ? 0 : 1;
    const bAct = (b.recommendation && b.recommendation !== "HOLD") ? 0 : 1;
    if (aAct !== bAct) return aAct - bAct;
    if (a.type !== b.type) return a.type.localeCompare(b.type);
    return a.ticker.localeCompare(b.ticker);
  });

  const header = `
    <div style="background:#0b1220;color:#fff;padding:14px 16px;border-radius:14px;">
      <div style="font-size:18px;font-weight:900;">📊 Portfolio Plan Dashboard</div>
      <div style="opacity:.9;font-size:12px;margin-top:4px;">
        Loaded rows: <b>${total}</b> · Action rows: <b>${actions}</b>
        · SELL ${sells} · HOLD ${holds} · ADD ${adds} · BUY ${buys}
      </div>
    </div>

    <div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;">
      <div style="flex:1;min-width:260px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:12px;">
        <div style="font-size:12px;color:#6b7280;font-weight:800;">GitHub Token (fine-grained PAT)</div>
        <div style="display:flex;gap:8px;margin-top:8px;">
          <input id="token" type="password" placeholder="${tokenSet ? "Token saved" : "Paste token"}"
            style="flex:1;padding:10px;border:1px solid #e5e7eb;border-radius:10px;"/>
          <button id="saveTok" style="padding:10px 12px;border-radius:10px;border:1px solid #111827;background:#111827;color:#fff;font-weight:800;cursor:pointer;">Save</button>
          <button id="clearTok" style="padding:10px 12px;border-radius:10px;border:1px solid #e5e7eb;background:#fff;font-weight:800;cursor:pointer;">Clear</button>
        </div>
        <div style="font-size:12px;color:#6b7280;margin-top:8px;">
          Token is stored only in your browser localStorage.
        </div>
      </div>

      <div style="flex:1;min-width:260px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:12px;">
        <div style="font-size:12px;color:#6b7280;font-weight:800;">Data Source</div>
        <div style="margin-top:8px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:10px;">
          ${escapeHtml(new URL(PLAN_PATH, window.location.href).toString())}
        </div>
      </div>
    </div>
  `;

  const tableHead = `
    <thead>
      <tr>
        <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Ticker</th>
        <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Type</th>
        <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Recommendation</th>
        <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Option / Symbol</th>
        <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Held</th>
        <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Sell</th>
        <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Add</th>
        <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Buy</th>
        <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Reason</th>
        <th style="text-align:center;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;">Execute</th>
      </tr>
    </thead>
  `;

  const tableRows = sorted.map(r => {
    const reco = r.recommendation || "—";
    const recoKind =
      reco.startsWith("SELL") ? (reco === "SELL_ALL" ? "danger" : "sell") :
      reco.startsWith("ADD") ? "ok" :
      reco === "HOLD" ? "ok" : "neutral";

    const optOrSym = r.option || r.optionSymbol || "";
    const reason = r.type === "BUY"
      ? (r.reason || "New entry")
      : (r.pyramidReason && r.pyramidReason !== "No action" ? r.pyramidReason : (r.reason || ""));

    return `
      <tr>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace;">${escapeHtml(r.ticker)}</td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;">${badge(escapeHtml(r.type), r.type === "SELL" ? "sell" : r.type === "BUY" ? "warn" : "neutral")}</td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;">${badge(escapeHtml(reco), recoKind)}</td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;max-width:360px;word-break:break-word;">
          ${escapeHtml(optOrSym)}
          ${r.expiry ? `<div style="color:#6b7280;font-size:11px;margin-top:3px;">Exp: ${escapeHtml(r.expiry)} ${r.strategy ? `· ${escapeHtml(r.strategy)}` : ""}</div>` : ""}
        </td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:right;font-variant-numeric:tabular-nums;">${fmtNum(r.contractsHeld) || ""}</td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:right;font-variant-numeric:tabular-nums;">${fmtNum(r.sellContracts) || ""}</td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:right;font-variant-numeric:tabular-nums;">${fmtNum(r.addContracts) || ""}</td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:right;font-variant-numeric:tabular-nums;">${fmtNum(r.buyContracts) || ""}</td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;max-width:320px;word-break:break-word;color:#374151;">
          ${escapeHtml(reason)}
        </td>
        <td style="padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:center;">
          <button class="execBtn"
            data-moveid="${escapeHtml(r.moveId)}"
            style="padding:8px 10px;border-radius:10px;border:1px solid #111827;background:#111827;color:#fff;font-weight:800;cursor:pointer;">
            Execute
          </button>
          <div class="status" data-status="${escapeHtml(r.moveId)}" style="margin-top:6px;font-size:11px;color:#6b7280;"></div>
        </td>
      </tr>
    `;
  }).join("");

  const table = `
    <div style="border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;background:#fff;">
      <div style="overflow:auto;">
        <table style="width:100%;border-collapse:collapse;">
          ${tableHead}
          <tbody>${tableRows || `<tr><td colspan="10" style="padding:12px;">No rows</td></tr>`}</tbody>
        </table>
      </div>
    </div>
  `;

  $("#app").innerHTML = header + table;

  // Token buttons
  $("#saveTok").onclick = () => {
    const v = $("#token").value.trim();
    if (!v) return alert("Paste a token first.");
    setToken(v);
    $("#token").value = "";
    alert("Token saved in browser.");
  };
  $("#clearTok").onclick = () => {
    clearToken();
    alert("Token cleared.");
  };

  // Execute buttons
  document.querySelectorAll(".execBtn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const moveId = btn.getAttribute("data-moveid");
      const row = sorted.find(x => x.moveId === moveId);
      const st = document.querySelector(`.status[data-status="${CSS.escape(moveId)}"]`);
      try {
        btn.disabled = true;
        btn.style.opacity = "0.6";
        if (st) st.textContent = "Dispatching…";

        await dispatchMove(row);

        if (st) st.textContent = "✅ Dispatched. Check Actions tab.";
      } catch (e) {
        if (st) st.textContent = `❌ ${e.message}`;
        btn.disabled = false;
        btn.style.opacity = "1";
      }
    });
  });
}

// ---------------- Main ----------------
async function loadPlan() {
  ensureBasicUI();

  try {
    // Helpful debug
    console.log("PLAN_PATH =", PLAN_PATH);
    console.log("Fetching plan from:", new URL(PLAN_PATH, window.location.href).toString());

    const res = await fetch(PLAN_PATH, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status} loading ${PLAN_PATH}`);

    const text = await res.text();
    const raw = parseCSV(text);
    const rows = normalizePlanRows(raw);

    console.log("Parsed raw rows:", raw.slice(0, 3));
    console.log("Normalized rows:", rows.slice(0, 3));
    console.log("Loaded rows:", rows.length);

    render(rows);
  } catch (e) {
    $("#app").innerHTML = `
      <div style="max-width:980px;margin:0 auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;">
        <div style="background:#fef2f2;border:1px solid #fecaca;color:#991b1b;padding:14px;border-radius:12px;">
          <div style="font-weight:900;">Failed to load plan</div>
          <div style="margin-top:6px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;white-space:pre-wrap;">
${escapeHtml(e.message)}
          </div>
          <div style="margin-top:10px;color:#7f1d1d;font-size:12px;">
            Confirm <b>portfolio_plan.csv</b> is inside <b>/docs</b> next to index.html, and you are serving from /docs.
          </div>
        </div>
      </div>
    `;
    console.error(e);
  }
}

document.addEventListener("DOMContentLoaded", loadPlan);