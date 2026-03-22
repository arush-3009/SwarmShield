let state = null;
function connectSSE() {
    const src = new EventSource('/stream');
    src.onmessage = function(ev) {
        const d = JSON.parse(ev.data);
        if (d.heartbeat) return;
        state = d;
        updateDashboard();
    };
    src.onerror = function() { src.close(); setTimeout(connectSSE, 2000); };
}
function updateDashboard() {
    if (!state) return;
    const dmg = state.server_damage, max = state.server_damage_max;
    const remaining = max - dmg, pct = (remaining / max) * 100;
    const bar = document.getElementById('server-health-bar');
    const txt = document.getElementById('server-health-text');
    const label = document.getElementById('server-health-label');
    bar.style.width = pct + '%';
    bar.className = 'health-bar' + (pct < 25 ? ' critical' : pct < 50 ? ' warning' : '');
    txt.textContent = Math.round(remaining) + ' / ' + Math.round(max);
    if (pct < 25) { label.textContent = 'CRITICAL'; label.className = 'health-label critical'; }
    else if (pct < 50) { label.textContent = 'WARNING'; label.className = 'health-label warning'; }
    else { label.textContent = 'SECURE'; label.className = 'health-label'; }
    setText('count-clean', state.counts.clean);
    setText('count-infected', state.counts.infected_total);
    setText('count-blocked', state.counts.blocked_total);
    setText('count-quarantined', state.counts.quarantined_total);
    for (let i = 0; i < 3; i++) {
        const ag = state.agents[i];
        let detail = `Host ${ag.host} (${ag.host_name})`;
        if (ag.in_transit) detail += ` → ${ag.target_name || '?'} (transit)`;
        if (ag.action) detail += '  |  ' + ag.action;
        setText('agent-' + i + '-detail', detail);
    }
    setText('ep-num', state.episode);
    setText('ep-step', state.timestep + ' / ' + state.max_timesteps);
    document.getElementById('ep-progress').style.width = (state.timestep / state.max_timesteps * 100) + '%';
    setText('ep-wins', state.wins);
    setText('ep-losses', state.losses);
    setText('ep-survived', state.survived);
    const banner = document.getElementById('outcome-banner');
    if (state.outcome) {
        banner.className = 'outcome-banner ' + state.outcome;
        banner.textContent = state.outcome === 'win' ? '✔  NETWORK SECURED' :
            state.outcome === 'loss' ? '☠  SERVER BREACHED' : '⏱  TIME EXPIRED';
    } else { banner.className = 'outcome-banner hidden'; }
    buildLog();
    for (let i = 0; i < 3; i++) {
        const el = document.getElementById('reward-' + i);
        const r = state.rewards[i];
        el.textContent = (r >= 0 ? '+' : '') + r.toFixed(2);
        el.className = 'reward-value' + (r > 0.01 ? ' positive' : r < -0.01 ? ' negative' : '');
    }
}
function buildLog() {
    const div = document.getElementById('action-log');
    let html = '';
    if (state.events) {
        for (const e of state.events) {
            if (e.type === 'infection') html += `<div class="log-entry event-infection"><span class="log-time">[t=${state.timestep}]</span> ⚠ NEW INFECTION: ${e.host_name}</div>`;
        }
        const attacks = (state.traffic || []).filter(t => t.type === 'server_attack');
        if (attacks.length > 0) html += `<div class="log-entry event-server"><span class="log-time">[t=${state.timestep}]</span> ⚡ SERVER ATTACK (${attacks.length} conn)</div>`;
    }
    if (state.action_log) {
        const entries = [...state.action_log].reverse();
        for (const e of entries) {
            const cls = `log-agent-${e.agent}`;
            let icon = '○ ', rowCls = '';
            if (e.action.startsWith('QUARANTINE')) { icon = '⬛ '; rowCls = 'event-quarantine'; }
            else if (e.action.startsWith('BLOCK')) { icon = '🛡️ '; rowCls = 'event-block'; }
            else if (e.action.startsWith('MOVE')) { icon = '→ '; }
            else if (e.action.startsWith('UNBLOCK')) { icon = '🔓 '; }
            html += `<div class="log-entry ${rowCls}"><span class="log-time">[t=${e.timestep}]</span> ${icon}<span class="${cls}">${e.agent_name}</span> ${e.action} @ ${e.host_name}</div>`;
        }
    }
    div.innerHTML = html;
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
document.addEventListener('keydown', function(e) {
    const k = e.key;
    if (k === ' ') { e.preventDefault(); post({action:'toggle_pause'}); }
    if (k === 'r' || k === 'R') post({action:'restart'});
    if (k === '+' || k === '=') post({action:'speed_up'});
    if (k === '-' || k === '_') post({action:'speed_down'});
    if (k === 'n' || k === 'N') post({action:'step'});
});
function post(body) { fetch('/control', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }
connectSSE();
