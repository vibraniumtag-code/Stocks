const WORKFLOW_FILE = "save_positions.yml";
const BRANCH = "main";
const POS_CSV_URL = "positions.csv"; // served from docs/positions.csv
const REQUIRED_COLS = ["ticker","option_name","option_entry_price","entry_date","underlying_entry_price","contracts"];

const elStatus = document.getElementById("status");
const elHead = document.getElementById("thead");
const elBody = document.getElementById("tbody");

const loadBtn = document.getElementById("loadBtn");
const addRowBtn = document.getElementById("addRowBtn");
const downloadBtn = document.getElementById("downloadBtn");
const saveBtn = document.getElementById("saveBtn");

let headers = [];
let data = [];          // array of objects
let originalCSV = "";   // to detect changes

function setStatus(msg){ elStatus.textContent = msg; }
function log(msg){ setStatus(msg); }

function esc(s){
  return String(s ?? "")
    .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

function parseCSV(text){
  const rows = [];
  let row = [], field = "";
  let inQuotes = false;

  for (let i=0;i<text.length;i++){
    const c=text[i], n=text[i+1];
    if(inQuotes){
      if(c === '"' && n === '"'){ field+='"'; i++; }
      else if(c === '"'){ inQuotes=false; }
      else field += c;
    } else {
      if(c === '"') inQuotes=true;
      else if(c === ','){ row.push(field); field=""; }
      else if(c === '\n'){ row.push(field); field=""; if(row.some(x=>x.trim()!=="")) rows.push(row); row=[]; }
      else if(c === '\r'){ /* ignore */ }
      else field += c;
    }
  }
  if(field.length || row.length){ row.push(field); rows.push(row); }
  return rows;
}

function toObjects(text){
  const rows = parseCSV(text);
  if(!rows.length) return [];
  headers = rows[0].map(h => (h||"").trim());
  const out=[];
  for(let i=1;i<rows.length;i++){
    const r=rows[i];
    const obj={};
    for(let j=0;j<headers.length;j++) obj[headers[j]] = (r[j] ?? "").trim();
    if(Object.values(obj).some(v => (v||"").trim() !== "")) out.push(obj);
  }
  return out;
}

function csvEscape(v){
  const s = String(v ?? "");
  if (s.includes('"') || s.includes(",") || s.includes("\n") || s.includes("\r")) {
    return `"${s.replaceAll('"','""')}"`;
  }
  return s;
}

function toCSV(headers, objs){
  const lines = [];
  lines.push(headers.map(csvEscape).join(","));
  for(const o of objs){
    lines.push(headers.map(h => csvEscape(o[h] ?? "")).join(","));
  }
  return lines.join("\n") + "\n";
}

function ensureRequiredColumns(){
  // Add missing required columns to headers, preserving existing order
  for(const c of REQUIRED_COLS){
    if(!headers.includes(c)) headers.push(c);
  }
  // Ensure every row has all headers
  for(const row of data){
    for(const h of headers){
      if(!(h in row)) row[h] = "";
    }
  }
}

function render(){
  // header
  elHead.innerHTML = headers.map(h => `<th>${esc(h)}</th>`).join("");

  // body
  elBody.innerHTML = data.map((row, idx) => {
    const tds = headers.map(h => {
      const v = row[h] ?? "";
      const isNum = (h === "contracts" || h.endsWith("_price"));
      const cls = `cell ${isNum ? "num" : ""}`;
      return `<td>
        <input class="${cls}" data-row="${idx}" data-col="${esc(h)}" value="${esc(v)}"/>
      </td>`;
    }).join("");

    return `<tr>${tds}</tr>`;
  }).join("") || `<tr><td style="padding:14px;">No rows.</td></tr>`;

  // wire inputs
  elBody.querySelectorAll("input.cell").forEach(inp => {
    inp.addEventListener("input", () => {
      inp.classList.add("changed");
      const r = Number(inp.getAttribute("data-row"));
      const c = inp.getAttribute("data-col");
      data[r][c] = inp.value;
      saveBtn.disabled = false;
      setStatus("Edited (not saved).");
    });
  });
}

async function loadPositions(){
  setStatus("Loading positions…");
  elBody.innerHTML = `<tr><td style="padding:14px;">Loading…</td></tr>`;
  try{
    const res = await fetch(`${POS_CSV_URL}?ts=${Date.now()}`, { cache:"no-store" });
    if(!res.ok) throw new Error(`Failed to load positions.csv: ${res.status}`);
    const text = await res.text();
    originalCSV = text;
    data = toObjects(text);
    ensureRequiredColumns();
    render();
    saveBtn.disabled = true;
    setStatus(`Loaded ${data.length} rows ✅`);
  }catch(e){
    setStatus(`Load failed ❌ ${e.message}`);
    elBody.innerHTML = `<tr><td style="padding:14px;color:#fecaca;">${esc(e.message)}</td></tr>`;
  }
}

function addRow(){
  const row = {};
  for(const h of headers) row[h] = "";
  data.push(row);
  render();
  saveBtn.disabled = false;
  setStatus("Row added (not saved).");
}

function downloadCSV(){
  const csv = toCSV(headers, data);
  const blob = new Blob([csv], {type:"text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "positions.csv";
  a.click();
  URL.revokeObjectURL(url);
}

async function dispatchSave(csvText){
  const owner = (localStorage.getItem("pm_owner") || "").trim();
  const repo  = (localStorage.getItem("pm_repo")  || "").trim();
  const token = (localStorage.getItem("pm_token") || "");
  if(!owner || !repo) throw new Error("Owner/repo not set. Go back to index.html and set them.");
  if(!token) throw new Error("Token not set. Go back to index.html and Save Token.");

  const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${WORKFLOW_FILE}/dispatches`;

  // workflow_dispatch inputs limit exists; keep payload smaller:
  // We'll base64 encode and send; workflow will decode.
  const b64 = btoa(unescape(encodeURIComponent(csvText)));

  const body = {
    ref: BRANCH,
    inputs: {
      positions_b64: b64
    }
  };

  const res = await fetch(url, {
    method:"POST",
    headers:{
      "Accept":"application/vnd.github+json",
      "Authorization":`token ${token}`,
      "Content-Type":"application/json"
    },
    body: JSON.stringify(body)
  });

  if(!res.ok){
    let msg = `${res.status} ${res.statusText}`;
    try{ msg = `${res.status}: ${JSON.stringify(await res.json())}`; }catch{}
    throw new Error(msg);
  }
}

async function saveToGitHub(){
  // basic validation
  for(const c of REQUIRED_COLS){
    if(!headers.includes(c)) throw new Error(`Missing required column: ${c}`);
  }

  // Build CSV
  const csv = toCSV(headers, data);

  if(csv.trim() === (originalCSV || "").trim()){
    setStatus("No changes to save.");
    return;
  }

  if(!confirm("This will commit positions.csv to the repo via GitHub Actions. Proceed?")) return;

  saveBtn.disabled = true;
  saveBtn.textContent = "Saving…";

  try{
    await dispatchSave(csv);
    setStatus("Dispatched save ✅ Check Actions tab for completion.");
    originalCSV = csv;
  }catch(e){
    setStatus(`Save failed ❌ ${e.message}`);
    alert(`Save failed:\n${e.message}`);
    saveBtn.disabled = false;
  }finally{
    saveBtn.textContent = "💾 Save to GitHub (commit)";
  }
}

loadBtn.addEventListener("click", loadPositions);
addRowBtn.addEventListener("click", addRow);
downloadBtn.addEventListener("click", downloadCSV);
saveBtn.addEventListener("click", saveToGitHub);

loadPositions();