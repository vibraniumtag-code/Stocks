// Positions Editor (docs/positions_app.js)
//
// Requires:
// - docs/positions.csv exists (mirrored by workflow)
// - workflow apply_position.yml supports move_type: SELL, ADD, BUY, DELETE
//
// Saves settings + token to localStorage.

const POS_CSV_URL = "./positions.csv"; // served from docs/

const LS_KEY = "pos_editor_settings_v1";
const LS_TOKEN = "pos_editor_token_v1";

function $(id){ return document.getElementById(id); }

function loadSettings(){
  const raw = localStorage.getItem(LS_KEY);
  const s = raw ? JSON.parse(raw) : {};
  $("owner").value = s.owner || "";
  $("repo").value = s.repo || "";
  $("workflow").value = s.workflow || "apply_position.yml";
  $("branch").value = s.branch || "main";
  $("token").value = localStorage.getItem(LS_TOKEN) || "";
}

function saveSettings(){
  const s = {
    owner: $("owner").value.trim(),
    repo: $("repo").value.trim(),
    workflow: $("workflow").value.trim(),
    branch: $("branch").value.trim(),
  };
  localStorage.setItem(LS_KEY, JSON.stringify(s));
  const tok = $("token").value.trim();
  if (tok) localStorage.setItem(LS_TOKEN, tok);
  setGlobalStatus("✅ Settings saved.", "good");
}

function clearToken(){
  localStorage.removeItem(LS_TOKEN);
  $("token").value = "";
  setGlobalStatus("🧹 Token cleared.", "warn");
}

function getCfg(){
  const raw = localStorage.getItem(LS_KEY);
  const s = raw ? JSON.parse(raw) : {};
  const token = localStorage.getItem(LS_TOKEN) || $("token").value.trim();

  return {
    owner: (s.owner || $("owner").value).trim(),
    repo: (s.repo || $("repo").value).trim(),
    workflow: (s.workflow || $("workflow").value).trim() || "apply_position.yml",
    branch: (s.branch || $("branch").value).trim() || "main",
    token
  };
}

function setGlobalStatus(msg, kind=""){
  const el = $("globalStatus");
  el.textContent = msg;
  el.classList.remove("mono");
  if (kind === "good") el.innerHTML = `<span class="pill good">${escapeHtml(msg)}</span>`;
  else if (kind === "bad") el.innerHTML = `<span class="pill bad">${escapeHtml(msg)}</span>`;
  else if (kind === "warn") el.innerHTML = `<span class="pill warn">${escapeHtml(msg)}</span>`;
  else el.textContent = msg;
}

function escapeHtml(s){
  return (s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

async function fetchNoCache(url){
  const bust = (url.includes("?") ? "&" : "?") + "t=" + Date.now();
  const res = await fetch(url + bust, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return { text: await res.text(), lastMod: res.headers.get("last-modified") };
}

function parseCSV(text){
  // Simple CSV parser that handles quoted commas
  const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
  if (!lines.length) return { header: [], rows: [] };

  const parseLine = (line) => {
    const out = [];
    let cur = "";
    let inQ = false;
    for (let i=0;i<line.length;i++){
      const ch = line[i];
      if (ch === '"' ){
        if (inQ && line[i+1] === '"'){ cur += '"'; i++; }
        else inQ = !inQ;
      } else if (ch === "," && !inQ){
        out.push(cur);
        cur = "";
      } else {
        cur += ch;
      }
    }
    out.push(cur);
    return out;
  };

  const header = parseLine(lines[0]).map(h => h.trim());
  const rows = lines.slice(1).map(l => {
    const vals = parseLine(l);
    const obj = {};
    header.forEach((h, idx) => obj[h] = (vals[idx] ?? "").trim());
    return obj;
  });
  return { header, rows };
}

function buildMoveId(prefix, row){
  // stable-ish id
  const t = (row.ticker||"").toUpperCase();
  const opt = (row.option_name||"").slice(0,40).replace(/\s+/g,"_");
  return `${Date.now()}-${prefix}-${t}-${opt}`;
}

async function dispatchWorkflow(inputs){
  const cfg = getCfg();
  if (!cfg.owner || !cfg.repo){
    throw new Error("Missing owner/repo in Settings.");
  }
  if (!cfg.token){
    throw new Error("No token saved. Paste your PAT and click Save Settings.");
  }

  const url = `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/actions/workflows/${cfg.workflow}/dispatches`;
  const body = {
    ref: cfg.branch,
    inputs
  };

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": `Bearer ${cfg.token}`,
      "X-GitHub-Api-Version": "2022-11-28"
    },
    body: JSON.stringify(body)
  });

  if (!res.ok){
    let msg = `${res.status}`;
    try { msg += `: ${JSON.stringify(await res.json())}`; }
    catch { /* ignore */ }
    throw new Error(`Dispatch failed (${msg})`);
  }
}

function rowStatusCell(kind, text){
  if (kind === "good") return `<span class="pill good">${escapeHtml(text)}</span>`;
  if (kind === "bad") return `<span class="pill bad">${escapeHtml(text)}</span>`;
  if (kind === "warn") return `<span class="pill warn">${escapeHtml(text)}</span>`;
  return `<span class="pill">${escapeHtml(text)}</span>`;
}

function renderTable(rows){
  const tbody = $("posTbody");
  if (!rows.length){
    tbody.innerHTML = `<tr><td colspan="10" class="muted">No rows found.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((r, idx) => {
    const ticker = escapeHtml(r.ticker || "");
    const opt = escapeHtml(r.option_name || "");
    const eopt = escapeHtml(r.option_entry_price || "");
    const edate = escapeHtml(r.entry_date || "");
    const eund = escapeHtml(r.underlying_entry_price || "");
    const contracts = escapeHtml(r.contracts || "0");

    return `
      <tr data-idx="${idx}">
        <td class="mono">${ticker}</td>
        <td class="mono" style="max-width:420px; word-break:break-word;">${opt}</td>
        <td class="num mono">${eopt}</td>
        <td class="mono">${edate}</td>
        <td class="num mono">${eund}</td>
        <td class="num mono">${contracts}</td>

        <td class="num">
          <input class="input small-input" type="number" min="0" step="1" value="${contracts}" data-role="setContracts">
        </td>

        <td>
          <button class="btn btn-primary" data-role="apply">✅ Apply</button>
        </td>

        <td>
          <button class="btn btn-danger" data-role="delete">🗑 Delete</button>
        </td>

        <td data-role="status">${rowStatusCell("warn","Ready")}</td>
      </tr>
    `;
  }).join("");

  // Wire events
  tbody.querySelectorAll("button[data-role='apply']").forEach(btn => {
    btn.addEventListener("click", async (e) => onApplyClick(e, rows));
  });
  tbody.querySelectorAll("button[data-role='delete']").forEach(btn => {
    btn.addEventListener("click", async (e) => onDeleteClick(e, rows));
  });
}

function setRowStatus(tr, kind, text){
  const cell = tr.querySelector("[data-role='status']");
  cell.innerHTML = rowStatusCell(kind, text);
}

function disableRowButtons(tr, disabled){
  tr.querySelectorAll("button").forEach(b => b.disabled = disabled);
  tr.querySelectorAll("input").forEach(i => i.disabled = disabled);
}

async function onApplyClick(e, rows){
  const tr = e.target.closest("tr");
  const idx = parseInt(tr.getAttribute("data-idx"), 10);
  const row = rows[idx];

  const cur = parseInt((row.contracts || "0"), 10) || 0;
  const desired = parseInt(tr.querySelector("input[data-role='setContracts']").value, 10);
  const want = isFinite(desired) ? Math.max(desired, 0) : cur;

  if (want === cur){
    setRowStatus(tr, "warn", "No change");
    return;
  }

  // Compute delta => SELL or ADD
  let move_type = "";
  let sell_contracts = "0";
  let add_contracts = "0";

  if (want < cur){
    move_type = "SELL";
    sell_contracts = String(cur - want);
  } else {
    move_type = "ADD";
    add_contracts = String(want - cur);
  }

  const option_name = (row.option_name || "").trim();
  const ticker = (row.ticker || "").trim();

  if (!ticker || !option_name){
    setRowStatus(tr, "bad", "Missing ticker/option");
    return;
  }

  const inputs = {
    move_id: buildMoveId(move_type, row),
    move_type,
    ticker,
    option_name,
    sell_contracts,
    add_contracts,
    buy_contracts: "0",
    strategy: "",
    expiry: "",
    option_symbol: "",
    option_entry_price: "",
    underlying_entry_price: ""
  };

  try{
    disableRowButtons(tr, true);
    setRowStatus(tr, "warn", "Dispatching…");
    await dispatchWorkflow(inputs);
    setRowStatus(tr, "good", "Queued ✅");
    // keep disabled so user doesn’t double-submit
    tr.style.opacity = "0.65";
  }catch(err){
    disableRowButtons(tr, false);
    setRowStatus(tr, "bad", String(err.message || err));
  }
}

async function onDeleteClick(e, rows){
  const tr = e.target.closest("tr");
  const idx = parseInt(tr.getAttribute("data-idx"), 10);
  const row = rows[idx];

  const option_name = (row.option_name || "").trim();
  const ticker = (row.ticker || "").trim();
  if (!ticker || !option_name){
    setRowStatus(tr, "bad", "Missing ticker/option");
    return;
  }

  // lightweight confirm (iOS-friendly)
  const ok = confirm(`Delete this row?\n\n${ticker}\n${option_name}`);
  if (!ok) return;

  const inputs = {
    move_id: buildMoveId("DELETE", row),
    move_type: "DELETE",
    ticker,
    option_name,
    sell_contracts: "0",
    add_contracts: "0",
    buy_contracts: "0",
    strategy: "",
    expiry: "",
    option_symbol: "",
    option_entry_price: "",
    underlying_entry_price: ""
  };

  try{
    disableRowButtons(tr, true);
    setRowStatus(tr, "warn", "Deleting…");
    await dispatchWorkflow(inputs);
    setRowStatus(tr, "good", "Queued ✅");
    tr.style.opacity = "0.55";
  }catch(err){
    disableRowButtons(tr, false);
    setRowStatus(tr, "bad", String(err.message || err));
  }
}

async function loadPositions(){
  try{
    setGlobalStatus("Loading positions…");
    const { text, lastMod } = await fetchNoCache(POS_CSV_URL);
    $("lastUpdated").textContent = lastMod ? `Last updated: ${lastMod}` : "Last updated: (unknown)";
    const parsed = parseCSV(text);

    // Expect your schema; but be tolerant
    const rows = parsed.rows.map(r => ({
      ticker: r.ticker || r.Ticker || "",
      option_name: r.option_name || r.Option || "",
      option_entry_price: r.option_entry_price || "",
      entry_date: r.entry_date || "",
      underlying_entry_price: r.underlying_entry_price || "",
      contracts: r.contracts || "0"
    }));

    renderTable(rows);
    setGlobalStatus(`✅ Loaded ${rows.length} rows.`, "good");
  }catch(err){
    $("posTbody").innerHTML = `<tr><td colspan="10" class="muted">❌ ${escapeHtml(String(err.message || err))}</td></tr>`;
    setGlobalStatus(`❌ ${String(err.message || err)}`, "bad");
  }
}

function promptToken(){
  const tok = prompt("Paste your GitHub PAT (classic):");
  if (!tok) return;
  $("token").value = tok.trim();
  saveSettings();
}

window.addEventListener("load", () => {
  loadSettings();
  loadPositions();

  $("btnRefresh").addEventListener("click", loadPositions);
  $("btnSaveSettings").addEventListener("click", saveSettings);
  $("btnClearToken").addEventListener("click", clearToken);
  $("btnToken").addEventListener("click", promptToken);
});