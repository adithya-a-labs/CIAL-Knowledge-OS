"""Self-contained, dependency-free HTML dashboard for experiment artifacts."""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .evaluation_metrics import aggregate_experiment, rank_experiments
from .evaluation_report import build_recommendations


def _scalar(value: str) -> Any:
    text = value.strip()
    if text.casefold() in {"true", "false"}:
        return text.casefold() == "true"
    if not text:
        return ""
    try:
        return float(text) if any(char in text for char in ".eE") else int(text)
    except ValueError:
        pass
    if text[:1] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: _scalar(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _normalize_row(row: Mapping[str, Any], experiment_id: str) -> dict[str, Any]:
    normalized = dict(row)
    aliases = {
        "generated_answer": ("generated_answer", "answer"),
        "retrieval_top_k": ("retrieval_top_k", "config_retrieval_top_k", "top_k"),
        "total_latency": ("total_latency", "total_latency_seconds"),
        "retrieval_latency": ("retrieval_latency", "retrieval_latency_seconds"),
        "generation_latency": ("generation_latency", "answer_latency_seconds"),
        "estimated_tokens": ("estimated_tokens", "final_context_tokens_estimate"),
        "retrieved_chunk_ids": ("retrieved_chunk_ids", "chunk_ids"),
        "retrieved_pages": ("retrieved_pages", "page_numbers"),
        "similarity_scores": ("similarity_scores", "retrieval_scores"),
    }
    normalized["experiment_id"] = normalized.get("experiment_id") or experiment_id
    for target, sources in aliases.items():
        if normalized.get(target) in (None, ""):
            normalized[target] = next(
                (normalized.get(source) for source in sources if normalized.get(source) not in (None, "")),
                "",
            )
    return normalized


def load_dashboard_data(
    output_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Load canonical artifacts, with legacy root CSVs as a compatibility fallback."""

    root = Path(output_root).expanduser().resolve()
    files = sorted((root / "experiments").glob("*.csv"))
    if not files:
        files = sorted(root.glob("*.csv"))
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(_normalize_row(row, path.stem) for row in _read_csv(path))

    summary_path = root / "summary" / "experiment_summary.csv"
    if summary_path.is_file():
        summaries = _read_csv(summary_path)
    else:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("experiment_id")), []).append(row)
        summaries = rank_experiments(
            aggregate_experiment(group) for group in grouped.values()
        )
    manifest = [str(path.relative_to(root)).replace("\\", "/") for path in files]
    if summary_path.is_file():
        manifest.append(str(summary_path.relative_to(root)).replace("\\", "/"))
    return rows, summaries, manifest


def _safe_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).replace("</", "<\\/")


def generate_dashboard(
    output_root: str | Path,
    *,
    summaries: Iterable[Mapping[str, Any]] | None = None,
    experiment_rows: Iterable[Mapping[str, Any]] | None = None,
    recommendations: Mapping[str, Any] | None = None,
    title: str = "RAG Experiment Evaluation",
) -> Path:
    """Generate ``reports/dashboard.html`` containing all data and code inline."""

    root = Path(output_root).expanduser().resolve()
    loaded_rows, loaded_summaries, manifest = load_dashboard_data(root)
    rows = [dict(row) for row in experiment_rows] if experiment_rows is not None else loaded_rows
    summary_values = (
        [dict(row) for row in summaries] if summaries is not None else loaded_summaries
    )
    recommendation_values = dict(
        recommendations or build_recommendations(summary_values)
    )
    payload = {
        "rows": rows,
        "summaries": summary_values,
        "recommendations": recommendation_values,
        "manifest": manifest,
    }
    document = _DASHBOARD_TEMPLATE.replace("{{TITLE}}", html.escape(title)).replace(
        "{{PAYLOAD}}", _safe_json(payload)
    )
    destination = root / "reports" / "dashboard.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination.resolve()


_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>{{TITLE}}</title>
<style>
:root{--bg:#07111f;--panel:#0e1b2d;--panel2:#12243a;--line:#263b55;--text:#e6edf6;--muted:#91a4ba;--cyan:#2dd4bf;--blue:#60a5fa;--amber:#fbbf24;--red:#fb7185;--green:#34d399}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,Segoe UI,Arial,sans-serif}
header{padding:28px 32px 20px;border-bottom:1px solid var(--line);background:linear-gradient(120deg,#0b1a2e,#10263a)}
h1{font-size:25px;margin:0 0 5px}h2{font-size:18px;margin:0 0 16px}h3{font-size:14px;margin:0 0 10px;color:#c9d7e7}
.sub,.muted{color:var(--muted)}main{max-width:1600px;margin:auto;padding:24px 28px 50px}
section{margin:0 0 28px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0}.stat{grid-column:span 3}.stat strong{display:block;font-size:23px;margin-top:7px;color:#fff}.half{grid-column:span 6}.third{grid-column:span 4}.full{grid-column:span 12}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:#a9bed4;cursor:pointer;position:sticky;top:0;background:var(--panel)}tr:hover td{background:#112238}
.scroll{overflow:auto;max-height:430px}select,input{background:#091626;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px 9px}label{color:var(--muted);margin-right:6px}.toolbar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
canvas{width:100%;height:270px}.heatmap{display:grid;gap:3px;overflow:auto}.heatcell{padding:10px;min-width:62px;text-align:center;border-radius:4px;color:#fff;font-size:11px}.funnel{display:flex;align-items:center;gap:7px;justify-content:center;flex-wrap:wrap}.funnel div{padding:12px 18px;background:#15304b;border:1px solid #315476;border-radius:7px;text-align:center}.arrow{color:var(--cyan)}
.barrow{display:grid;grid-template-columns:150px 1fr 48px;gap:8px;align-items:center;margin:8px 0}.bar{height:9px;border-radius:5px;background:#19304a;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan))}
.recommend{border-left:3px solid var(--cyan)}ul{margin:8px 0;padding-left:20px}.pill{display:inline-block;padding:3px 7px;border-radius:20px;background:#17314a;color:#b9d6ee;margin:2px}.good{color:var(--green)}.bad{color:var(--red)}pre{white-space:pre-wrap;word-break:break-word;background:#071321;padding:12px;border-radius:6px;max-height:260px;overflow:auto;color:#c6d6e7}
.empty{padding:24px;text-align:center;color:var(--muted)}footer{color:var(--muted);padding-top:15px;border-top:1px solid var(--line)}
@media(max-width:900px){.stat,.half,.third{grid-column:span 12}main{padding:18px}.grid{gap:10px}}
@media print{body{background:#fff;color:#111}.card,header{background:#fff;border-color:#ccc}main{max-width:none}.toolbar{display:none}}
</style>
</head>
<body>
<header><h1>{{TITLE}}</h1><div class="sub">Offline engineering dashboard · embedded artifact snapshot · no notebook or network required</div></header>
<main>
<section><h2>Overview</h2><div id="overview" class="grid"></div></section>
<section><h2>Configuration Leaderboard</h2><div class="card"><div class="toolbar"><label>Rank by <select id="rankMetric"></select></label><span id="manifest" class="muted"></span></div><div class="scroll"><table id="leaderboard"></table></div></div></section>
<section><h2>Performance Overview</h2><div class="grid">
<div class="card half"><h3>Accuracy vs Latency</h3><canvas id="accuracyLatency"></canvas></div>
<div class="card half"><h3>Context Size vs Accuracy</h3><canvas id="contextAccuracy"></canvas></div>
<div class="card third"><h3>Retrieval Top-K vs Accuracy</h3><canvas id="topKAccuracy"></canvas></div>
<div class="card third"><h3>Retrieval Top-K vs Latency</h3><canvas id="topKLatency"></canvas></div>
<div class="card third"><h3>Context Budget vs Latency</h3><canvas id="budgetLatency"></canvas></div>
</div></section>
<section><h2>Heatmaps</h2><div class="grid">
<div class="card half"><h3>Top-K × Context Size (Accuracy)</h3><div id="heatTopContext"></div></div>
<div class="card half"><h3>Top-K × Pass Rate</h3><div id="heatTopPass"></div></div>
<div class="card half"><h3>Context Size × Hallucination Rate</h3><div id="heatContextHall"></div></div>
<div class="card half"><h3>Neighbor Window × Accuracy</h3><div id="heatNeighborAccuracy"></div></div>
</div></section>
<section><h2>Retrieval Analytics</h2><div class="grid">
<div class="card full"><h3>Retrieval Funnel</h3><div id="funnel" class="funnel"></div></div>
<div class="card third"><h3>Deduplication Effectiveness</h3><div id="dedup"></div></div>
<div class="card third"><h3>Neighbor Expansion Statistics</h3><div id="neighbor"></div></div>
<div class="card third"><h3>Context Compression Statistics</h3><div id="compression"></div></div>
<div class="card full"><h3>Citation Quality Distribution</h3><canvas id="citationQuality"></canvas></div>
</div></section>
<section><h2>Question Analytics</h2><div class="grid">
<div class="card half"><h3>Pass Rate by Category</h3><div id="categoryBars"></div></div>
<div class="card half"><h3>Pass Rate by Difficulty</h3><div id="difficultyBars"></div></div>
<div class="card half"><h3>Questions Failing Across Every Configuration</h3><div id="alwaysFail" class="scroll"></div></div>
<div class="card half"><h3>Questions Consistently Answered Correctly</h3><div id="alwaysPass" class="scroll"></div></div>
<div class="card half"><h3>Questions Benefiting from Larger Context</h3><div id="contextBenefit" class="scroll"></div></div>
<div class="card half"><h3>Questions Benefiting from Larger Top-K</h3><div id="topKBenefit" class="scroll"></div></div>
</div></section>
<section><h2>Latency Analytics</h2><div class="grid">
<div class="card third"><h3>Total Latency Distribution</h3><canvas id="totalLatency"></canvas></div>
<div class="card third"><h3>Retrieval Latency</h3><canvas id="retrievalLatency"></canvas></div>
<div class="card third"><h3>Context Construction Latency</h3><canvas id="contextLatency"></canvas></div>
<div class="card half"><h3>Generation Latency</h3><canvas id="generationLatency"></canvas></div>
<div class="card half"><h3>Configuration Latency Comparison</h3><canvas id="configLatency"></canvas></div>
</div></section>
<section><h2>Recommendation Panel</h2><div id="recommendations" class="card recommend"></div></section>
<section><h2>Experiment Explorer</h2><div class="card">
<div class="toolbar"><label>Experiment <select id="experimentSelect"></select></label><label>Question <select id="questionSelect"></select></label><label>Status <select id="statusFilter"><option value="">All</option><option value="true">Pass</option><option value="false">Fail</option></select></label></div>
<div id="explorer"></div></div></section>
<footer id="footer"></footer>
</main>
<script>
const DATA={{PAYLOAD}};
const rows=DATA.rows||[], summaries=DATA.summaries||[], rec=DATA.recommendations||{};
const num=(v,d=0)=>{const n=Number(v);return Number.isFinite(n)?n:d};
const bool=v=>v===true||String(v).toLowerCase()==='true'||v===1;
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const avg=(a,key)=>a.length?a.reduce((s,r)=>s+num(r[key]),0)/a.length:0;
const pct=v=>`${(num(v)*100).toFixed(1)}%`, sec=v=>`${num(v).toFixed(3)}s`, compact=v=>num(v).toLocaleString(undefined,{maximumFractionDigits:1});
function best(key,min=false){if(!summaries.length)return null;return [...summaries].sort((a,b)=>(min?1:-1)*(num(a[key])-num(b[key])))[0]}
function card(label,value,detail=''){return `<div class="card stat"><span class="muted">${esc(label)}</span><strong>${esc(value)}</strong><small class="muted">${esc(detail)}</small></div>`}
const totalQuestions=new Set(rows.map(r=>r.question_id||r.question)).size;
const overall=best('overall_score'), fastest=best('average_latency',true), accurate=best('answer_accuracy'), safe=best('hallucination_rate',true), cited=best('citation_quality');
document.querySelector('#overview').innerHTML=[
card('Total experiments',summaries.length),card('Questions evaluated',totalQuestions),
card('Best configuration',overall?.experiment_id||'n/a',overall?`score ${num(overall.overall_score).toFixed(3)}`:''),
card('Fastest configuration',fastest?.experiment_id||'n/a',fastest?sec(fastest.average_latency):''),
card('Highest answer accuracy',accurate?.experiment_id||'n/a',accurate?pct(accurate.answer_accuracy):''),
card('Lowest hallucination rate',safe?.experiment_id||'n/a',safe?pct(safe.hallucination_rate):''),
card('Highest citation quality',cited?.experiment_id||'n/a',cited?pct(cited.citation_quality):''),
card('Evaluated answers',rows.length)
].join('');
if(rows.length&&!rows.some(r=>r.passed_answer_test!==undefined&&r.passed_answer_test!=='')){
document.querySelector('#overview').insertAdjacentHTML('beforeend','<div class="card full"><b>Legacy data notice:</b> these batch exports predate ground-truth evaluation fields. Run the automated sweep to populate accuracy, hallucination, citation-quality, category, and difficulty analytics.</div>');
}
const metrics={overall_score:'Overall score',answer_accuracy:'Answer accuracy',average_latency:'Latency (lowest)',hallucination_rate:'Hallucination (lowest)',citation_quality:'Citation quality',keyword_coverage:'Keyword coverage'};
const metricSelect=document.querySelector('#rankMetric');metricSelect.innerHTML=Object.entries(metrics).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');
function table(el,data,columns){if(!data.length){el.innerHTML='<tbody><tr><td class="empty">No experiment data found.</td></tr></tbody>';return}let sortKey='',asc=true;const draw=()=>{const d=[...data];if(sortKey)d.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];return (typeof x==='number'||!isNaN(Number(x)))?(num(x)-num(y))*(asc?1:-1):String(x).localeCompare(String(y))*(asc?1:-1)});el.innerHTML=`<thead><tr>${columns.map(c=>`<th data-key="${c}">${esc(c.replaceAll('_',' '))}</th>`).join('')}</tr></thead><tbody>${d.map(r=>`<tr>${columns.map(c=>`<td>${formatCell(c,r[c])}</td>`).join('')}</tr>`).join('')}</tbody>`;el.querySelectorAll('th').forEach(th=>th.onclick=()=>{asc=sortKey===th.dataset.key?!asc:true;sortKey=th.dataset.key;draw()})};draw()}
function formatCell(k,v){if(v===null||v===undefined||v==='')return '—';if(/accuracy|rate|coverage|quality/.test(k))return pct(v);if(/latency/.test(k))return sec(v);if(typeof v==='object')return `<span title="${esc(JSON.stringify(v))}">details</span>`;return esc(v)}
function renderLeaderboard(){const key=metricSelect.value||'overall_score',min=/latency|hallucination/.test(key);const ranked=[...summaries].sort((a,b)=>(min?1:-1)*(num(a[key])-num(b[key]))).map((r,i)=>({...r,display_rank:i+1}));const configCols=[...new Set(summaries.flatMap(Object.keys))].filter(k=>k.startsWith('config_')||['retrieval_top_k','max_context_chars','neighbor_window'].includes(k)).slice(0,5);table(document.querySelector('#leaderboard'),ranked,['display_rank','experiment_id',...configCols,'answer_accuracy','average_latency','hallucination_rate','citation_quality','overall_score'])}
metricSelect.onchange=renderLeaderboard;renderLeaderboard();
document.querySelector('#manifest').textContent=`Embedded sources: ${DATA.manifest.length||0}`;
function canvas(id,points,xKey,yKey,xLabel,yLabel,bar=false){const c=document.getElementById(id);if(!c)return;const ratio=devicePixelRatio||1,w=c.clientWidth||500,h=270;c.width=w*ratio;c.height=h*ratio;const ctx=c.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);ctx.strokeStyle='#29415d';ctx.fillStyle='#91a4ba';ctx.font='11px Segoe UI';const pad={l:48,r:14,t:14,b:38};ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();if(!points.length){ctx.fillText('No data',w/2-20,h/2);return}const xs=points.map(p=>num(p[xKey])),ys=points.map(p=>num(p[yKey])),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(0,...ys),ymax=Math.max(...ys,0.001);const X=x=>pad.l+(x-xmin)/(xmax-xmin||1)*(w-pad.l-pad.r),Y=y=>h-pad.b-(y-ymin)/(ymax-ymin||1)*(h-pad.t-pad.b);ctx.fillStyle='#91a4ba';ctx.fillText(xLabel,w/2-30,h-8);ctx.save();ctx.translate(11,h/2+30);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();ctx.fillText(ymax.toFixed(2),4,pad.t+5);ctx.fillText(ymin.toFixed(2),8,h-pad.b);if(bar){const bw=Math.max(3,(w-pad.l-pad.r)/points.length*.65);points.forEach((p,i)=>{ctx.fillStyle='#2dd4bf';ctx.fillRect(X(num(p[xKey]))-bw/2,Y(num(p[yKey])),bw,h-pad.b-Y(num(p[yKey])))})}else{ctx.strokeStyle='#60a5fa';ctx.beginPath();points.sort((a,b)=>num(a[xKey])-num(b[xKey])).forEach((p,i)=>{const x=X(num(p[xKey])),y=Y(num(p[yKey]));if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();points.forEach(p=>{ctx.fillStyle='#2dd4bf';ctx.beginPath();ctx.arc(X(num(p[xKey])),Y(num(p[yKey])),4,0,Math.PI*2);ctx.fill()})}}
function summaryAlias(key,...fallbacks){return summaries.map((r,i)=>({...r,_i:i,[key]:r[key]??fallbacks.map(k=>r[k]).find(v=>v!==undefined)}))}
canvas('accuracyLatency',summaries,'average_latency','answer_accuracy','Latency','Accuracy');canvas('contextAccuracy',summaries,'average_context_size','answer_accuracy','Context chars','Accuracy');
canvas('topKAccuracy',summaryAlias('retrieval_top_k','config_retrieval_top_k','config_top_k'),'retrieval_top_k','answer_accuracy','Top-K','Accuracy');
canvas('topKLatency',summaryAlias('retrieval_top_k','config_retrieval_top_k','config_top_k'),'retrieval_top_k','average_latency','Top-K','Latency');
canvas('budgetLatency',summaryAlias('max_context_chars','config_max_context_chars'),'max_context_chars','average_latency','Budget','Latency');
function heat(id,xKey,yKey,valueKey){const el=document.getElementById(id),xs=[...new Set(summaries.map(r=>r[xKey]??r[`config_${xKey}`]).filter(v=>v!==''&&v!=null))].sort((a,b)=>num(a)-num(b)),ys=[...new Set(summaries.map(r=>r[yKey]??r[`config_${yKey}`]).filter(v=>v!==''&&v!=null))].sort((a,b)=>num(a)-num(b));if(!xs.length){el.innerHTML='<div class="empty">Metric not available</div>';return}const yvals=ys.length?ys:['pass'];el.className='heatmap';el.style.gridTemplateColumns=`90px repeat(${xs.length},minmax(62px,1fr))`;let out='<div></div>'+xs.map(x=>`<div class="muted">${esc(x)}</div>`).join('');yvals.forEach(y=>{out+=`<div class="muted">${esc(y)}</div>`;xs.forEach(x=>{const matched=summaries.filter(r=>(r[xKey]??r[`config_${xKey}`])==x&&(y==='pass'||(r[yKey]??r[`config_${yKey}`])==y));const v=matched.length?avg(matched,valueKey):0;const hue=valueKey.includes('hallucination')?120*(1-v):120*v;out+=`<div class="heatcell" style="background:hsl(${hue} 55% 31%)">${pct(v)}</div>`})});el.innerHTML=out}
heat('heatTopContext','retrieval_top_k','max_context_chars','answer_accuracy');heat('heatTopPass','retrieval_top_k','_none','answer_accuracy');heat('heatContextHall','max_context_chars','_none','hallucination_rate');heat('heatNeighborAccuracy','neighbor_window','_none','answer_accuracy');
const stages=[['Retrieved','average_retrieved_chunks'],['Deduplicated','average_deduplicated_chunks'],['Expanded','average_expanded_chunks'],['Merged','average_merged_sections'],['Compressed','average_context_sections']];
document.querySelector('#funnel').innerHTML=stages.map(([l,k],i)=>`${i?'<span class="arrow">→</span>':''}<div><strong>${compact(avg(summaries,k))}</strong><br><span class="muted">${l}</span></div>`).join('');
function mini(id,items){document.getElementById(id).innerHTML=items.map(([l,v])=>`<div class="barrow"><span>${esc(l)}</span><div class="bar"><i style="width:${Math.max(0,Math.min(100,num(v)*100))}%"></i></div><b>${pct(v)}</b></div>`).join('')}
const retrieved=avg(summaries,'average_retrieved_chunks'),deduped=avg(summaries,'average_deduplicated_chunks'),expanded=avg(summaries,'average_expanded_chunks'),merged=avg(summaries,'average_merged_sections');
mini('dedup',[['Removed',retrieved?1-deduped/retrieved:0]]);mini('neighbor',[['Expansion',deduped?Math.max(0,expanded/deduped-1):0]]);mini('compression',[['Reduced',expanded?1-merged/expanded:0]]);
function histogram(id,key){const values=rows.map(r=>num(r[key],NaN)).filter(Number.isFinite);if(!values.length){canvas(id,[],'x','y','','');return}const lo=Math.min(...values),hi=Math.max(...values),bins=10,width=(hi-lo||1)/bins,points=Array.from({length:bins},(_,i)=>({x:lo+(i+.5)*width,y:0}));values.forEach(v=>points[Math.min(bins-1,Math.floor((v-lo)/(hi-lo||1)*bins))].y++);canvas(id,points,'x','y',key.replaceAll('_',' '),'Count',true)}
histogram('citationQuality','citation_quality');histogram('totalLatency','total_latency');histogram('retrievalLatency','retrieval_latency');histogram('contextLatency','context_construction_latency');histogram('generationLatency','generation_latency');
canvas('configLatency',summaries.map((r,i)=>({...r,index:i+1})),'index','average_latency','Configuration','Latency',true);
function groupedRate(key){const g={};rows.forEach(r=>{const k=r[key]||'unknown';(g[k]??=[]).push(bool(r.passed_answer_test))});return Object.entries(g).map(([k,v])=>[k,v.filter(Boolean).length/v.length])}
mini('categoryBars',groupedRate('category'));mini('difficultyBars',groupedRate('difficulty'));
const byQ={};rows.forEach(r=>(byQ[r.question_id||r.question]??=[]).push(r));
function questionList(predicate){const found=Object.values(byQ).filter(predicate).map(a=>a[0].question);return found.length?`<ul>${found.slice(0,60).map(q=>`<li>${esc(q)}</li>`).join('')}</ul>`:'<div class="empty">None identified</div>'}
document.querySelector('#alwaysFail').innerHTML=questionList(a=>a.length===summaries.length&&a.every(r=>!bool(r.passed_answer_test)));
document.querySelector('#alwaysPass').innerHTML=questionList(a=>a.length===summaries.length&&a.every(r=>bool(r.passed_answer_test)));
function benefits(key){return questionList(a=>{const valid=a.filter(r=>r[key]!==''&&r[key]!=null).sort((x,y)=>num(x[key])-num(y[key]));return valid.length>1&&!bool(valid[0].passed_answer_test)&&bool(valid.at(-1).passed_answer_test)})}
document.querySelector('#contextBenefit').innerHTML=benefits('max_context_chars');document.querySelector('#topKBenefit').innerHTML=benefits('retrieval_top_k');
const recItems=(rec.recommendations||[]).map(x=>`<span class="pill">${esc(x.parameter)} = ${esc(x.value)}</span>`).join('');
document.querySelector('#recommendations').innerHTML=`<h3>Recommended default: <span class="good">${esc(rec.recommended_default_configuration||'n/a')}</span></h3><p>${recItems||'No ranked configurations available.'}</p><h3>Trade-offs</h3><ul>${(rec.tradeoffs||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul><h3>Observed bottlenecks</h3><ul>${(rec.bottlenecks||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul><h3>Suggested improvements for Phase 3</h3><ul>${(rec.phase_3_improvements||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`;
const expSel=document.querySelector('#experimentSelect'),qSel=document.querySelector('#questionSelect'),statusSel=document.querySelector('#statusFilter');const expIds=[...new Set(rows.map(r=>r.experiment_id))];expSel.innerHTML=expIds.map(x=>`<option>${esc(x)}</option>`).join('');
function updateQuestions(){const subset=rows.filter(r=>r.experiment_id===expSel.value&&(statusSel.value===''||String(bool(r.passed_answer_test))===statusSel.value));qSel.innerHTML=subset.map((r,i)=>`<option value="${i}">${esc(r.question_id)} · ${esc(r.question)}</option>`).join('');qSel._rows=subset;showExplorer()}
function showExplorer(){const r=(qSel._rows||[])[num(qSel.value)];if(!r){document.querySelector('#explorer').innerHTML='<div class="empty">No experiment rows available.</div>';return}const config=Object.fromEntries(Object.entries(r).filter(([k])=>k.startsWith('config_')||['retrieval_top_k','max_context_chars','neighbor_window','multi_query_enabled','neighbor_expansion_enabled'].includes(k)));document.querySelector('#explorer').innerHTML=`<div class="grid"><div class="card half"><h3>Question & Evaluation</h3><p>${esc(r.question)}</p><p><b class="${bool(r.passed_answer_test)?'good':'bad'}">${bool(r.passed_answer_test)?'PASS':'FAIL'}</b> · ${esc(r.category)} · ${esc(r.difficulty)}</p><h3>Generated Answer</h3><pre>${esc(r.generated_answer)}</pre><h3>Expected Answer</h3><pre>${esc(r.expected_answer)}</pre></div><div class="card half"><h3>Configuration</h3><pre>${esc(JSON.stringify(config,null,2))}</pre><h3>Citations</h3><pre>${esc(JSON.stringify(r.citations||[],null,2))}</pre><h3>Retrieval Trace</h3><pre>${esc(JSON.stringify(r.retrieval_trace||{},null,2))}</pre><h3>Context & Latency</h3><pre>${esc(JSON.stringify({context_characters:r.final_context_characters,estimated_tokens:r.estimated_tokens,sections:r.final_context_sections,total_latency:r.total_latency,retrieval_latency:r.retrieval_latency,context_construction_latency:r.context_construction_latency,generation_latency:r.generation_latency},null,2))}</pre></div></div>`}
expSel.onchange=updateQuestions;statusSel.onchange=updateQuestions;qSel.onchange=showExplorer;updateQuestions();
document.querySelector('#footer').textContent=`Generated from an embedded snapshot of ${DATA.manifest.join(', ')||'no source artifacts'}. Re-run the experiment sweep to refresh this file.`;
window.addEventListener('resize',()=>{clearTimeout(window._resize);window._resize=setTimeout(()=>location.reload(),250)});
</script>
</body></html>"""
