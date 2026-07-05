(() => {
  const $ = id => document.getElementById(id);
  const names = {
    query_analyzer: "Query Analyzer", response_planner: "Response Planner",
    phase4_retrieval: "Phase 4 Retrieval", evidence_selection: "Evidence Selection",
    prompt_composer: "Prompt Composer", draft_generator: "Draft Generator",
    critic_agent: "Critic", compliance_agent: "Compliance", risk_agent: "Risk",
    evidence_verifier: "Evidence Verifier", consensus_engine: "Consensus Engine",
    finalizer: "Finalizer"
  };
  let state = null;
  const safe = value => value === null || value === undefined || value === "" ? "—" : String(value);
  const esc = value => safe(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pct = value => `${Math.round(Number(value || 0) * (Number(value || 0) <= 1 ? 100 : 1))}%`;
  const duration = seconds => `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
  const bytes = value => {
    let n = Number(value || 0), units = ["B", "KB", "MB", "GB", "TB"], i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
  };
  const metric = (label, value) => `<div class="metric"><span>${label}</span><strong>${esc(value)}</strong></div>`;
  const telemetryCard = (label, value, percentage) => `<div class="telemetry-card"><span>${label}</span><strong title="${esc(value)}">${esc(value)}</strong>${percentage === undefined ? "" : `<div class="gauge"><i style="width:${Math.max(0, Math.min(100, percentage))}%"></i></div>`}</div>`;

  function readiness(metrics) {
    if (metrics.final_status === "unsupported_query") return ["Unsupported", "bad"];
    if (metrics.final_status === "insufficient_evidence") return ["Insufficient evidence", "bad"];
    if (metrics.consensus_decision === "reject") return ["Rejected", "bad"];
    if (metrics.consensus_decision === "accept" && metrics.verification_rate >= .85 && metrics.risk_level !== "high") return ["Ready", "good"];
    if (metrics.consensus_decision) return ["Use with caution", "caution"];
    return ["Waiting", ""];
  }
  function render(s) {
    state = s;
    $("runId").textContent = safe(s.run_id); $("question").textContent = safe(s.question);
    $("stage").textContent = safe(s.current_stage); $("currentAgent").textContent = names[s.current_agent] || safe(s.current_agent);
    $("answerStatus").textContent = safe(s.answer_status); $("elapsed").textContent = duration(s.elapsed_seconds || 0);
    $("runStatus").textContent = s.run_status; $("runStatus").className = `status ${s.run_status}`;
    $("revision").textContent = s.revision?.used ? `Loop ${s.revision.loop} · ${s.revision.status}` : "Not used";
    $("progressBar").style.width = `${s.progress || 0}%`; $("progressLabel").textContent = `${Math.round(s.progress || 0)}%`;
    $("pipeline").innerHTML = Object.entries(names).map(([key, label]) => `<div class="pipeline-node ${s.agents[key]?.status || "pending"}">${label}</div>`).join("");
    $("agents").innerHTML = Object.entries(names).map(([key, label]) => {
      const a = s.agents[key] || {};
      const warnings = [...(a.warnings || []), ...(a.errors || [])].join("; ");
      return `<article class="agent-card ${a.status || "pending"}"><span class="state">${esc(a.status || "pending")}</span><strong>${label}</strong><dl><dt>Model</dt><dd title="${esc(a.model)}">${esc(a.model)}</dd><dt>Latency</dt><dd>${a.latency_ms == null ? "—" : `${Number(a.latency_ms).toFixed(1)} ms`}</dd><dt>Fallback</dt><dd>${a.fallback_used ? "Yes" : "No"}</dd><dt>Notes</dt><dd title="${esc(warnings)}">${esc(warnings || "—")}</dd></dl></article>`;
    }).join("");
    const m = s.metrics || {};
    $("evidenceMetrics").innerHTML = metric("Selected chunks", m.selected_evidence_count) + metric("Sufficiency", pct(m.evidence_sufficiency_score)) + metric("Source diversity", m.source_diversity);
    const modalities = m.modality_mix || {}, max = Math.max(1, ...Object.values(modalities));
    $("modalityMix").innerHTML = Object.entries(modalities).length ? Object.entries(modalities).map(([name, count]) => `<div class="bar-row"><span>${esc(name)}</span><div class="bar-track"><i style="width:${count / max * 100}%"></i></div><b>${count}</b></div>`).join("") : "<span class='eyebrow'>No evidence yet</span>";
    $("qualityMetrics").innerHTML = metric("Verification", pct(m.verification_rate)) + metric("Compliance", m.compliance_passed === undefined ? "Waiting" : m.compliance_passed ? "Passed" : "Failed") + metric("Risk", m.risk_level) + metric("Unsupported", m.unsupported_claim_count) + metric("Citation mismatches", m.citation_mismatch_count) + metric("Critic issues", m.critic_issue_count) + metric("Consensus", m.consensus_decision) + metric("Final status", m.final_status);
    const ready = readiness(m); $("readiness").textContent = ready[0]; $("readiness").className = `readiness ${ready[1]}`;
    const t = s.telemetry || {}, gpu = t.gpu || {};
    $("gpuStatus").textContent = gpu.available ? (gpu.devices?.map(d => d.name).join(", ") || "GPU available") : "GPU unavailable";
    $("telemetry").innerHTML = telemetryCard("CPU", pct(t.cpu_percent), t.cpu_percent) + telemetryCard("RAM", `${pct(t.ram_percent)} · ${bytes(t.ram_used_bytes)}`, t.ram_percent) + telemetryCard("GPU", gpu.available ? pct(gpu.usage_percent) : "Unavailable", gpu.usage_percent) + telemetryCard("VRAM", gpu.available ? `${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB` : "Unavailable", gpu.available && gpu.memory_total_mb ? gpu.memory_used_mb / gpu.memory_total_mb * 100 : undefined) + telemetryCard("Disk", `${pct(t.disk_percent)} · ${bytes(t.disk_used_bytes)}`, t.disk_percent) + telemetryCard("Process RAM", bytes(t.process_memory_bytes)) + telemetryCard("Current model", t.current_model || Object.values(s.agents || {}).find(a => a.status === "running")?.model || "—") + telemetryCard("Model latency", t.model_latency_ms == null ? "—" : `${t.model_latency_ms} ms`) + telemetryCard("Generated tokens", t.tokens_generated) + telemetryCard("Tokens / sec", t.tokens_per_second);
    $("answer").textContent = s.final_answer || s.draft_answer || "Draft and final answer updates will appear here.";
    $("citations").innerHTML = (s.citations || []).map((c, i) => `<div class="citation">[${i + 1}] ${esc(c.source || c.relative_path || c.file_name)}</div>`).join("");
  }
  function logEvent(event) {
    const box = $("log"), level = event.event_type?.includes("failed") ? "error" : (event.data?.warnings?.length ? "warning" : "");
    const message = event.agent || event.stage || event.data?.message || event.event_type;
    box.insertAdjacentHTML("beforeend", `<div class="log-entry ${level}"><time>${new Date(event.timestamp || Date.now()).toLocaleTimeString()}</time><b>${esc(event.event_type)}</b> · ${esc(message)}</div>`);
    box.scrollTop = box.scrollHeight;
  }
  fetch("/api/state").then(r => r.json()).then(render).catch(() => {});
  const stream = new EventSource("/events");
  stream.addEventListener("snapshot", e => render(JSON.parse(e.data)));
  stream.onopen = () => { $("connection").textContent = "Connected"; $("connection").className = "status connected"; };
  stream.onerror = () => { $("connection").textContent = "Reconnecting"; $("connection").className = "status waiting"; };
  const eventTypes = ["run_started","run_completed","run_failed","stage_started","stage_completed","agent_started","agent_completed","agent_failed","phase4_started","phase4_completed","evidence_selected","draft_generated","critic_completed","compliance_completed","risk_completed","verification_completed","consensus_decided","revision_started","revision_completed","telemetry_update"];
  eventTypes.forEach(type => stream.addEventListener(type, e => { const event = JSON.parse(e.data); logEvent(event); fetch("/api/state").then(r => r.json()).then(render); }));
  setInterval(() => { if (state?.run_status === "running") { state.elapsed_seconds = Number(state.elapsed_seconds || 0) + 1; $("elapsed").textContent = duration(state.elapsed_seconds); } }, 1000);
  $("clearLog").onclick = () => $("log").replaceChildren();
  const theme = localStorage.getItem("cial-live-theme") || "dark";
  document.documentElement.dataset.theme = theme; $("theme").textContent = theme === "dark" ? "Light mode" : "Dark mode";
  $("theme").onclick = () => { const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = next; localStorage.setItem("cial-live-theme", next); $("theme").textContent = next === "dark" ? "Light mode" : "Dark mode"; };
})();
