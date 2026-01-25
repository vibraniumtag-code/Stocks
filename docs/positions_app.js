/* Positions Editor — full page JS
   Loads from Pages root: ./positions.csv
   Saves by dispatching your "Apply Move" workflow with an UPSERT move.
*/

const $ = (id) => document.getElementById(id);

// IMPORTANT: On GitHub Pages (docs as root), this must be relative root
const POS_CSV_URL = "./positions.csv";

const CFG_KEY = "stocks_cfg_v1";
const DEFAULT_CFG = {
  owner: "vibraniumtag-code",
  repo: "Stocks",
  branch: "main",
  workflow: "apply_move.yml", // file name inside .github/workflows/
  token: ""
};

// --------- UI state
let state = {
  rows: [],          // current working rows (editable)
  original: [],      // last loaded rows (for dirty compare)
  deletedKeys: new Set(),
  dirty: false,
  saving: false,
};

// --------- Helpers
function escapeHtml(s){
  return String(s ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function setStatus(text, kind="info"){
  const el = $("status");
  el.textContent = text;
  el.className = `status status-${kind}`;
}

function setDirty(on){
  state.dirty = !!on;
  $("dirty").style.display = on ? "inline-flex" : "none";
}

function setSaving(on){
  state.saving = !!on;
  $("btnSave").disabled = on;
  $("btnAdd").disabled = on;
  $("btnReload").disabled = on;
  if (on) setStatus("Saving…", "info");
}

function rowKey(r){
  // Unique key (ticker + option_name). Matches how your positions.csv identifies a position.
  return `${(r.ticker||"").toUpperCase().trim()}|${(r.option_name||"").trim()}`;
}

function normalizeRow(r){
  return {
    ticker: (r.ticker||"").toUpperCase().trim(),
    option_name: (r.option_name||"").trim(),
    option_entry_price: (r.option_entry_price||"").trim(),
    entry_date: (r.entry_date||"").trim(),
    underlying_entry_price: (r.underlying_entry_price||"").trim(),
    contracts: (r.contracts||"").trim(),
  };
}

function deepEqualRows(a,b){
  if (a.length !== b.length) return false;
  for (let i=0;i<a.length;i++){
    const x=a[i], y=b[i];
    for (const k of ["ticker","option_name","option_entry_price","entry_date","underlying_entry_price","contracts"]){
      if (String(x[k]??"") !== String(y[k]??"")) return false;
    }
  }
  return true;
}

async function fetchNoCache(url){
  const bust = url.includes("?") ? "&" : "?";
  const res = await fetch(url + bust + "t=" + Date.now(), { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed ${res.status} for ${url}`);
  const text = await res.text();
  const lastMod = res.headers.get("last-modified") || "";
  return { text, lastMod };
}

// Minimal CSV parser (handles quoted fields)
function parseCSV(text){
  const lines = (text || "").trim().split(/\r?\n/);
  if (!lines.length) return { headers: [], rows: [] };

  const parseLine = (line) => {
    const out = [];
    let cur = "", inQ=false;
    for (let i=0;i<line.length;i++){
      const ch=line[i];
      if (ch === '"' ){
        if (inQ && line[i+1] === '"'){ cur += '"'; i++; }
        else inQ = !inQ;
      } else if (ch === "," && !inQ){
        out.push(cur); cur="";
      } else cur += ch;
    }
    out.push(cur);
    return out;
  };

  const headers = parseLine(lines[0]).map(h => h.trim());
  const rows = [];
  for (let i=1;i<lines.length;i++){
    if (!lines[i].trim()) continue;
    const vals = parseLine(lines[i]);
    const obj = {};
    headers.forEach((h, idx) => obj[h] = (vals[idx] ?? ""));
    rows.push(obj);
  }
  return { headers, rows };
}

function toCSV(rows){
  const headers = ["ticker","option_name","option_entry_price","entry_date","underlying_entry_price","contracts"];
  const esc = (v) => {
    const s = String(v ?? "");
    if (/[",\n]/.test(s)) return `"${s.replaceAll('"','""')}"`;
    return s;
  };
  const out = [];
  out.push(headers.join(","));
  rows.forEach(r=>{
    out.push(headers.map(h=>esc(r[h] ?? "")).join(","));
  });
  return out.join("\n") + "\n";
}

// --------- Config storage
function getCfg(){
  try{
    const raw = localStorage.getItem(CFG_KEY);
    if (!raw) return { ...DEFAULT_CFG };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_CFG, ...parsed };
  }catch{
    return { ...DEFAULT_CFG };
  }
}

function setCfg(cfg){
  localStorage.setItem(CFG_KEY, JSON.stringify(cfg));
}

function populateCfgUI(){
  const cfg = getCfg();
  $("cfgOwner").value = cfg.owner || "";
  $("cfgRepo").value = cfg.repo || "";
  $("cfgBranch").value = cfg.branch || "main";
  $("cfgWorkflow").value = cfg.workflow || "apply_move.yml";
  $("cfgToken").value = cfg.token || "";
}

// --------- Table rendering
function render(){
  const tbody = $("posTbody");
  const rows = state.rows.filter(r => !state.deletedKeys.has(rowKey(r)));

  // sort by ticker
  rows.sort((a,b)=> (a.ticker||"").localeCompare(b.ticker||""));

  if (!rows.length){
    tbody.innerHTML = `<tr><td colspan="7" class="muted">No rows.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((r, idx)=>{
    const key = rowKey(r);
    return `
      <tr data-key="${escapeHtml(key)}">
        <td><input class="in" value="${escapeHtml(r.ticker)}" data-k="ticker" /></td>
        <td><input class="in" value="${escapeHtml(r.option_name)}" data-k="option_name" /></td>
        <td><input class="in in-num" value="${escapeHtml(r.option_entry_price)}" data-k="option_entry_price" /></td>
        <td><input class="in" value="${escapeHtml(r.entry_date)}" data-k="entry_date" placeholder="YYYY-MM-DD" /></td>
        <td><input class="in in-num" value="${escapeHtml(r.underlying_entry_price)}" data-k="underlying_entry_price" /></td>
        <td><input class="in in-num" value="${escapeHtml(r.contracts)}" data-k="contracts" /></td>
        <td class="center">
          <button class="btn btn-small btn-danger" data-del="1">Delete</button>
        </td>
      </tr>
    `;
  }).join("");

  // bind events
  tbody.querySelectorAll("input.in").forEach(inp=>{
    inp.addEventListener("input", ()=>{
      const tr = inp.closest("tr");
      const key = tr.getAttribute("data-key");
      const k = inp.getAttribute("data-k");
      const newVal = inp.value;

      // find row in state.rows by key (key might change if ticker/option_name edited)
      // We'll locate by DOM position: get current display rows list index
      // Simpler: rebuild using rowKey matching before edit; store original key in dataset
      let idx = state.rows.findIndex(rr => rowKey(rr) === key);
      if (idx === -1){
        // fallback: try by DOM order
        const all = [...$("posTbody").querySelectorAll("tr")];
        const pos = all.indexOf(tr);
        const liveRows = state.rows.filter(r=>!state.deletedKeys.has(rowKey(r))).sort((a,b)=>(a.ticker||"").localeCompare(b.ticker||""));
        const tgt = liveRows[pos];
        idx = state.rows.findIndex(rr => rr === tgt);
      }
      if (idx >= 0){
        state.rows[idx][k] = newVal;
        // key might have changed if ticker/option_name edited:
        tr.setAttribute("data-key", escapeHtml(rowKey(state.rows[idx])));
        setDirty(true);
      }
    });
  });

  tbody.querySelectorAll("button[data-del]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const tr = btn.closest("tr");
      const key = tr.getAttribute("data-key");
      state.deletedKeys.add(key);
      setDirty(true);
      render();
    });
  });
}

// --------- Loading
async function loadPositions(){
  try{
    setStatus("Loading…", "info");
    $("fileInfo").textContent = "File: positions.csv";

    const { text, lastMod } = await fetchNoCache(POS_CSV_URL);
    $("lastUpdated").textContent = lastMod ? `Last updated: ${lastMod}` : "Last updated: (unknown)";

    const parsed = parseCSV(text);
    const rows = parsed.rows.map(obj => normalizeRow({
      ticker: obj.ticker || obj.Ticker || "",
      option_name: obj.option_name || obj.Option || "",
      option_entry_price: obj.option_entry_price || "",
      entry_date: obj.entry_date || "",
      underlying_entry_price: obj.underlying_entry_price || "",
      contracts: obj.contracts || "0",
    }));

    state.rows = rows;
    state.original = rows.map(r=>({ ...r }));
    state.deletedKeys = new Set();
    setDirty(false);

    render();
    setStatus(`✅ Loaded ${rows.length} rows`, "good");
  }catch(e){
    $("posTbody").innerHTML = `<tr><td colspan="7" class="muted">❌ ${escapeHtml(e.message || String(e))}</td></tr>`;
    setStatus("❌ Failed to load", "bad");
  }
}

// --------- Workflow dispatch save
async function dispatchApplyMove(payload){
  const cfg = getCfg();
  if (!cfg.token) throw new Error("No token saved. Paste your PAT and click Save Settings first.");
  if (!cfg.owner || !cfg.repo) throw new Error("Missing owner/repo in settings.");
  const workflow = cfg.workflow || "apply_move.yml";

  const url = `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/actions/workflows/${workflow}/dispatches`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": `token ${cfg.token}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok){
    let txt = "";
    try{ txt = await res.text(); }catch{}
    throw new Error(`Dispatch failed (${res.status}): ${txt}`);
  }
}

// We will save by sending ONE workflow dispatch with a special move_type=UPSERT_BULK
// that your apply_move.yml should handle (or you can implement a simpler workflow_dispatch that accepts csv_blob).
async function saveAllRows(){
  const cfg = getCfg();
  if (!cfg.token) throw new Error("No token saved. Add PAT in settings.");
  // Validate rows
  const rows = state.rows.filter(r => !state.deletedKeys.has(rowKey(r))).map(normalizeRow);

  // basic sanity
  for (const r of rows){
    if (!r.ticker) throw new Error("One row is missing ticker.");
    if (!r.option_name) throw new Error(`Row ${r.ticker} is missing option_name.`);
    const c = Number(r.contracts);
    if (!Number.isFinite(c) || c < 0) throw new Error(`Row ${r.ticker} has invalid contracts.`);
  }

  // Build CSV blob to write
  const csvBlob = toCSV(rows);

  // Dispatch workflow with inputs that write docs/positions.csv + positions.csv
  await dispatchApplyMove({
    ref: cfg.branch || "main",
    inputs: {
      move_id: `BULK-${Date.now()}`,
      move_type: "UPSERT_BULK",
      ticker: "BULK",
      option_name: "BULK",
      sell_contracts: "0",
      add_contracts: "0",
      buy_contracts: "0",
      strategy: "",
      expiry: "",
      option_symbol: "",
      option_entry_price: "",
      underlying_entry_price: "",
      csv_blob: csvBlob
    }
  });
}

// --------- Add row
function addRow(){
  const newRow = {
    ticker: "",
    option_name: "",
    option_entry_price: "",
    entry_date: "",
    underlying_entry_price: "",
    contracts: "1"
  };
  state.rows.push(newRow);
  setDirty(true);
  render();

  // focus last row ticker
  setTimeout(()=>{
    const trs = [...$("posTbody").querySelectorAll("tr")];
    const tr = trs[trs.length - 1];
    if (tr){
      const inp = tr.querySelector('input[data-k="ticker"]');
      inp?.focus();
      tr.scrollIntoView({ behavior:"smooth", block:"center" });
    }
  }, 50);
}

// --------- Wire up UI
$("btnReload").addEventListener("click", loadPositions);
$("btnAdd").addEventListener("click", addRow);

$("btnSaveCfg").addEventListener("click", ()=>{
  const cfg = {
    owner: $("cfgOwner").value.trim(),
    repo: $("cfgRepo").value.trim(),
    branch: ($("cfgBranch").value.trim() || "main"),
    workflow: ($("cfgWorkflow").value.trim() || "apply_move.yml"),
    token: $("cfgToken").value.trim()
  };
  setCfg(cfg);
  populateCfgUI();
  setStatus("✅ Settings saved", "good");
});

$("btnClearCfg").addEventListener("click", ()=>{
  localStorage.removeItem(CFG_KEY);
  populateCfgUI();
  setStatus("Cleared settings", "info");
});

$("btnSave").addEventListener("click", async ()=>{
  if (!state.dirty){
    setStatus("Nothing to save", "info");
    return;
  }
  try{
    setSaving(true);
    await saveAllRows();
    setStatus("✅ Save requested. Refresh in ~10–30s.", "good");
    setDirty(false);
  }catch(e){
    setStatus(`❌ ${e.message || e}`, "bad");
  }finally{
    setSaving(false);
  }
});

// init
populateCfgUI();
loadPositions();