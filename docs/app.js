// ====== CONFIG: set these ======
const OWNER = "vibraniumtag-code";
const REPO  = "Stocks";
const WORKFLOW_FILE = "apply_move.yml"; // must match .github/workflows/apply_move.yml
const BRANCH = "main";
const PLAN_PATH = "../portfolio_plan.csv"; // file in repo root
// ===============================

const LS_TOKEN_KEY = "pm_exec_pat_v1";

function $(id){ return document.getElementById(id); }

function setStatus(msg, cls="muted"){
  const el = $("status");
  el.className = `${cls} mono`;
  el.textContent = msg;
}

function pill(type){
  const t = (type||"").toUpperCase();
  if (t === "SELL") return `<span class="pill pill-sell">SELL</span>`;
  if (t === "ADD")  return `<span class="pill pill-add">ADD</span>`;
  if (t === "BUY")  return `<span class="pill pill-buy">BUY</span>`;
  return `<span class="pill pill-hold">${t||"?"}</span>`;
}

// minimal CSV parser (handles quoted commas)
function parseCSV(text){
  const rows = [];
  let cur = [], val = "", inQ = false;
  for (let i=0;i<text.length;i++){
    const c = text[i];
    if (c === '"' ){
      if (inQ && text[i+1] === '"'){ val += '"'; i++; }
      else inQ = !inQ;
    } else if (c === ',' && !inQ){
      cur.push(val); val="";
    } else if ((c === '\n' || c === '\r') && !inQ){
      if (val !== "" || cur.length){
        cur.push(val); rows.push(cur);
        cur = []; val="";
      }
      // swallow \r\n
      if (c === '\r' && text[i+1] === '\n') i++;
    } else {
      val += c;
    }
  }
  if (val !== "" || cur.length){ cur.push(val); rows.push(cur); }
  return rows;
}

function toObj(headers, row){
  const o = {};
  headers.forEach((h, i)=> o[h] = (row[i] ?? "").trim());
  return o;
}

function buildMoveId(r){
  const type = (r["Type"] || "").toUpperCase();
  const ticker = (r["Ticker"] || "").toUpperCase();
  if (["SELL","ADD","HOLD"].includes(type)){
    return `${type}|${ticker}|${r["Option"]||""}`;
  }
  if (type === "BUY"){
    return `BUY|${ticker}|${r["OptionSymbol"]||""}|${r["Expiry"]||""}`;
  }
  return `${type}|${ticker}`;
}

async function loadPlan(){
  setStatus("Fetching portfolio_plan.csv …");
  const res = await fetch(PLAN_PATH, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load plan: ${res.status}`);
  const text = await res.text();
  const parsed = parseCSV(text);
  if (parsed.length < 2) return [];
  const headers = parsed[0].map(s=>s.trim());
  const data = parsed.slice(1).filter(r=>r.some(x=>String(x||"").trim() !== ""));
  return data.map(r=>toObj(headers, r));
}

function renderTable(rows){
  const tbody = $("planTable").querySelector("tbody");
  if (!rows.length){
    tbody.innerHTML = `<tr><td colspan="11" class="muted">No rows in portfolio_plan.csv</td></tr>`;
    return;
  }

  // action-first sort
  rows.sort((a,b)=>{
    const ta = (a.Type||"").toUpperCase();
    const tb = (b.Type||"").toUpperCase();
    const aAction = (ta === "SELL" || ta === "ADD" || ta === "BUY") ? 0 : 1;
    const bAction = (tb === "SELL" || tb === "ADD" || tb === "BUY") ? 0 : 1;
    if (aAction !== bAction) return aAction - bAction;
    return (a.Ticker||"").localeCompare(b.Ticker||"");
  });

  tbody.innerHTML = rows.map(r=>{
    const type = (r.Type||"").toUpperCase();
    const moveId = buildMoveId(r);

    const optionOrSymbol = r.Option || r.OptionSymbol || "";
    const held = r.ContractsHeld || "";
    const sell = r.SellContracts || "";
    const add  = r.AddContracts || "";
    const buy  = r.BuyContracts || "";
    const est  = r.EstCostTotal || "";
    const reason = r.PyramidReason || r.Reason || "";

    const canMutate = (type === "SELL" || type === "ADD");
    const execLabel = canMutate ? "✅ Executed (updates CSV)" : "✅ Executed (ack only)";

    return `
      <tr>
        <td class="mono">${escapeHtml(moveId)}</td>
        <td>${pill(type)}</td>
        <td class="mono">${escapeHtml((r.Ticker||"").toUpperCase())}</td>
        <td class="mono">${escapeHtml(optionOrSymbol)}</td>
        <td class="num">${escapeHtml(held)}</td>
        <td class="num">${escapeHtml(sell)}</td>
        <td class="num">${escapeHtml(add)}</td>
        <td class="num">${escapeHtml(buy)}</td>
        <td class="num">${escapeHtml(est)}</td>
        <td style="white-space:normal; max-width:360px;">${escapeHtml(reason)}</td>
        <td>
          <div class="actions">
            <button class="primary" data-act="executed" data-id="${escapeAttr(moveId)}">${execLabel}</button>
            <button class="warn" data-act="skip" data-id="${escapeAttr(moveId)}">↩️ Skip</button>
            <button class="danger" data-act="revert" data-id="${escapeAttr(moveId)}">🧨 Revert (log only)</button>
          </div>
        </td>
      </tr>
    `;
  }).join("");

  // attach click handlers
  tbody.querySelectorAll("button[data-act]").forEach(btn=>{
    btn.addEventListener("click", async ()=>{
      const act = btn.getAttribute("data-act");
      const moveId = btn.getAttribute("data-id");
      await dispatchMove(moveId, act);
    });
  });
}

function escapeHtml(s){ return String(s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"); }
function escapeAttr(s){ return escapeHtml(s).replaceAll('"',"&quot;"); }

function getToken(){
  return localStorage.getItem(LS_TOKEN_KEY) || "";
}

async function dispatchMove(moveId, decision){
  const token = getToken();
  if (!token){
    alert("No token saved. Paste your PAT and click Save Token first.");
    return;
  }

  setStatus(`Dispatching workflow: ${decision} ${moveId} …`);

  const url = `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`;
  const payload = {
    ref: BRANCH,
    inputs: { move_id: moveId, decision }
  };

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Accept": "application/vnd.github+json",
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (res.status === 204){
    setStatus(`✅ Dispatched. GitHub Action will commit updates shortly. (${decision} ${moveId})`, "ok");
    return;
  }

  let msg = `Dispatch failed: HTTP ${res.status}`;
  try {
    const j = await res.json();
    msg += `\n${JSON.stringify(j, null, 2)}`;
  } catch {}
  console.error(msg);
  setStatus(`❌ ${msg}`, "bad");
  alert(msg);
}

async function refresh(){
  try {
    const rows = await loadPlan();
    renderTable(rows);
    setStatus(`Loaded plan rows: ${rows.length}`, "ok");
  } catch (e){
    console.error(e);
    setStatus(`❌ ${e.message}`, "bad");
  }
}

// token controls
$("saveToken").addEventListener("click", ()=>{
  const v = $("token").value.trim();
  if (!v) return alert("Paste token first.");
  localStorage.setItem(LS_TOKEN_KEY, v);
  $("token").value = "";
  setStatus("Token saved in localStorage.", "ok");
});

$("clearToken").addEventListener("click", ()=>{
  localStorage.removeItem(LS_TOKEN_KEY);
  setStatus("Token cleared.", "ok");
});

$("refresh").addEventListener("click", refresh);

// init
refresh();