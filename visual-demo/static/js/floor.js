const canvas = document.getElementById("floor-canvas");
const ctx = canvas.getContext("2d");

let state = null;
let animTick = 0;
let eventFx = [];

const COLORS = {
    bgTop: "#1a1410",
    bgBottom: "#0f0c08",
    wall: "#e8e0d0",
    wallStroke: "#b09878",
    carpet: "#c4b49a",
    carpetAlt: "#b8a88e",
    wood: "#b48b57",
    woodDark: "#8b6842",
    desk: "#cfab74",
    deskDark: "#a77d4d",
    text: "#3a2e24",
    softText: "#7a6a5a",
    clean: "#39c96b",
    infected: "#f44336",
    blocked: "#ffad33",
    quarantined: "#8ea0b6",
    falseContain: "#b06ed6",
    normal: "#5fdcff",
    c2: "#ba68ff",
    scan: "#ff964d",
    lateral: "#ff625a",
    attack: "#ff1744",
    agent0: "#ffd54a",
    agent1: "#61a7ff",
    agent2: "#ff7083",
};

// Host positions as percentages of the floor area
const HOSTS = {
    13: { x: 20, y: 14 },   // Pam (reception)
    12: { x: 85, y: 14 },   // Michael (his office)
    0: { x: 38, y: 34 },   // Jim
    1: { x: 48, y: 34 },   // Dwight
    4: { x: 58, y: 34 },   // Andy
    2: { x: 38, y: 48 },   // Stanley
    3: { x: 48, y: 48 },   // Phyllis
    5: { x: 14, y: 40 },   // Angela
    6: { x: 14, y: 50 },   // Oscar
    7: { x: 14, y: 60 },   // Kevin
    8: { x: 80, y: 38 },   // Erin
    9: { x: 80, y: 50 },   // Meredith
    10: { x: 80, y: 62 },   // Creed
    11: { x: 80, y: 74 },   // Darryl
    14: { x: 20, y: 76 },   // Conf A
    15: { x: 28, y: 76 },   // Conf B
    16: { x: 36, y: 76 },   // Conf C
    17: { x: 92, y: 86 },   // File Server
};

const DISPLAY_NAMES = [
    "Jim", "Dwight", "Stanley", "Phyllis", "Andy",
    "Angela", "Oscar", "Kevin",
    "Erin", "Meredith", "Creed", "Darryl",
    "Michael", "Pam", "Conf A", "Conf B", "Conf C", "FILE SERVER"
];

const AGENT_COLORS = [COLORS.agent0, COLORS.agent1, COLORS.agent2];
const AGENT_NAMES = ["David Wallace", "Jan Gould", "Robert California"];
const C2_POS = { x: 50, y: 2 };


function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + "px";
    canvas.style.height = window.innerHeight + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", setupCanvas);
setupCanvas();

function wx(pct) { return 40 + (pct / 100) * (window.innerWidth - 360); }
function wy(pct) { return 40 + (pct / 100) * (window.innerHeight - 80); }

function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function connectSSE() {
    const src = new EventSource("/stream");
    src.onmessage = function (ev) {
        const d = JSON.parse(ev.data);
        if (d.heartbeat) return;
        if (state && d.timestep !== state.timestep) buildEventFx(d);
        state = d;
        updateHUD();
    };
    src.onerror = function () { src.close(); setTimeout(connectSSE, 2000); };
}
connectSSE();

document.addEventListener("keydown", function (e) {
    if (e.key === " ") { e.preventDefault(); post({ action: "toggle_pause" }); }
    if (e.key === "r" || e.key === "R") post({ action: "restart" });
    if (e.key === "+" || e.key === "=") post({ action: "speed_up" });
    if (e.key === "-" || e.key === "_") post({ action: "speed_down" });
    if (e.key === "n" || e.key === "N") post({ action: "step" });
});

function post(body) {
    fetch("/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}


function updateHUD() {
    if (!state) return;

    document.getElementById("hud-infected").textContent = state.counts.infected_total;
    document.getElementById("hud-blocked").textContent = state.counts.blocked_total;
    document.getElementById("hud-quarantined").textContent = state.counts.quarantined_total;
    document.getElementById("hud-step").textContent = state.timestep + " / " + state.max_timesteps;

    var dmg = state.server_damage || 0;
    var mx = state.server_damage_max || 300;
    var pct = Math.max(0, Math.min(1, dmg / mx));
    var fill = document.getElementById("server-bar-fill");
    fill.style.width = (pct * 100) + "%";
    if (pct < 0.35) fill.style.background = "linear-gradient(90deg, #2ecc71, #87d96c)";
    else if (pct < 0.7) fill.style.background = "linear-gradient(90deg, #ffad33, #ffcd5c)";
    else fill.style.background = "linear-gradient(90deg, #f44336, #ff6d6d)";
    document.getElementById("server-bar-text").textContent = Math.round(dmg) + " / " + Math.round(mx) + " damage";

    for (var i = 0; i < 3; i++) {
        var ag = state.agents[i];
        if (!ag) continue;
        var txt = ag.host_name;
        if (ag.in_transit && ag.target_name) txt += " -> " + ag.target_name;
        if (ag.action) txt += " | " + ag.action;
        document.getElementById("agent-" + i + "-detail").textContent = txt;
    }

    var ribbon = document.getElementById("event-ribbon");
    var events = state.events || [];
    if (events.length === 0) {
        if (state.outcome === "win") ribbon.innerHTML = '<span class="event-badge badge-cool">SECURED</span> All infections quarantined!';
        else if (state.outcome === "loss") ribbon.innerHTML = '<span class="event-badge badge-alert">BREACH</span> File server compromised.';
        else ribbon.textContent = "Monitoring branch traffic...";
    } else {
        var html = "";
        var latest = events.slice(-3);
        for (var j = 0; j < latest.length; j++) {
            var e = latest[j];
            if (e.type === "infection") html += '<span class="event-badge badge-alert">INFECTION</span>' + e.host_name + " compromised  ";
            else if (e.type === "block") html += '<span class="event-badge badge-warn">BLOCK</span>' + e.host_name + "  ";
            else if (e.type === "quarantine") html += '<span class="event-badge badge-cool">QUARANTINE</span>' + e.host_name + "  ";
            else if (e.type === "unblock") html += '<span class="event-badge badge-move">UNBLOCK</span>' + e.host_name + "  ";
        }
        ribbon.innerHTML = html;
    }
}


function buildEventFx(nextState) {
    var events = nextState.events || [];
    for (var i = 0; i < events.length; i++) {
        var e = events[i];
        if (e.host != null && HOSTS[e.host]) {
            eventFx.push({
                kind: e.type,
                x: HOSTS[e.host].x,
                y: HOSTS[e.host].y,
                born: performance.now(),
                life: 900
            });
        }
    }
}

function drawEffects() {
    var now = performance.now();
    eventFx = eventFx.filter(function (fx) { return now - fx.born < fx.life; });

    for (var i = 0; i < eventFx.length; i++) {
        var fx = eventFx[i];
        var age = (now - fx.born) / fx.life;
        var x = wx(fx.x);
        var y = wy(fx.y);

        var color = COLORS.clean;
        if (fx.kind === "infection") color = COLORS.infected;
        if (fx.kind === "block") color = COLORS.blocked;
        if (fx.kind === "quarantine") color = COLORS.quarantined;

        ctx.save();
        ctx.globalAlpha = 1 - age;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3 - age * 2;
        ctx.beginPath();
        ctx.arc(x, y, 12 + age * 30, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }
}



function drawBackground() {
    var grad = ctx.createRadialGradient(
        window.innerWidth * 0.3, window.innerHeight * 0.3, 0,
        window.innerWidth * 0.5, window.innerHeight * 0.5, window.innerWidth * 0.8
    );
    grad.addColorStop(0, "#2a2018");
    grad.addColorStop(1, "#0f0c08");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
}

function drawFloor() {
    // Main floor rectangle
    var x = wx(6), y = wy(4);
    var w = wx(98) - wx(6), h = wy(96) - wy(4);

    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.3)";
    ctx.shadowBlur = 30;
    ctx.shadowOffsetY = 10;
    ctx.fillStyle = COLORS.carpet;
    roundRect(x, y, w, h, 12);
    ctx.fill();
    ctx.restore();

    
    ctx.fillStyle = COLORS.carpetAlt;
    for (var i = 0; i < 300; i++) {
        var px = x + ((i * 97) % Math.floor(w));
        var py = y + ((i * 53) % Math.floor(h));
        ctx.fillRect(px, py, 2, 2);
    }
}

function drawRoom(x1, y1, x2, y2, label, fillColor) {
    var x = wx(x1), y = wy(y1), w = wx(x2) - wx(x1), h = wy(y2) - wy(y1);
    ctx.fillStyle = fillColor || "#ddd4c4";
    roundRect(x, y, w, h, 8);
    ctx.fill();
    ctx.strokeStyle = COLORS.wallStroke;
    ctx.lineWidth = 3;
    roundRect(x, y, w, h, 8);
    ctx.stroke();
    if (label) {
        ctx.fillStyle = COLORS.softText;
        ctx.font = "700 " + Math.max(11, window.innerWidth * 0.008) + "px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(label, x + w / 2, y + 18);
    }
}

function drawRooms() {
    drawRoom(8, 6, 30, 24, "RECEPTION", "#e0d6c6");
    drawRoom(74, 6, 98, 28, "MICHAEL'S OFFICE", "#d9cebe");
    drawRoom(8, 34, 24, 66, "ACCOUNTING", "#ddd0c0");
    drawRoom(26, 26, 72, 58, "", "#d7ccbb");  // Main sales floor
    drawRoom(74, 30, 92, 80, "BACK OFFICE", "#d4cab8");
    drawRoom(8, 66, 42, 88, "CONFERENCE ROOM", "#ddd4c6");
    drawRoom(86, 80, 98, 94, "SERVER", "#cec2ae");

    
    var sx = wx(32), sy = wy(7);
    ctx.fillStyle = "#285a99";
    roundRect(sx, sy, 160, 32, 4);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "800 14px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("DUNDER MIFFLIN", sx + 10, sy + 15);
    ctx.font = "600 9px Inter, sans-serif";
    ctx.fillText("Paper Company Inc.", sx + 10, sy + 26);
}



function drawDesk(cx, cy, isServer) {
    var x = wx(cx), y = wy(cy);
    var dW = Math.max(32, window.innerWidth * 0.032);
    var dH = dW * 0.6;

    if (isServer) {
        // Server rack
        ctx.fillStyle = "#5e6672";
        roundRect(x - 12, y - 20, 24, 40, 3);
        ctx.fill();
        ctx.fillStyle = "#97a3b3";
        for (var k = 0; k < 5; k++) {
            ctx.fillRect(x - 8, y - 14 + k * 7, 16, 2);
        }
        return;
    }

    // Desk body
    ctx.fillStyle = COLORS.deskDark;
    roundRect(x - dW / 2, y - dH / 2, dW, dH, 4);
    ctx.fill();
    ctx.fillStyle = COLORS.desk;
    roundRect(x - dW / 2 + 2, y - dH / 2 + 2, dW - 4, dH - 4, 3);
    ctx.fill();

    // Monitor
    var mW = dW * 0.4;
    var mH = dH * 0.45;
    ctx.fillStyle = "#2b2f35";
    roundRect(x - mW / 2, y - mH / 2, mW, mH, 2);
    ctx.fill();

    // Chair
    ctx.fillStyle = "#8e8f94";
    ctx.beginPath();
    ctx.arc(x - dW / 2 - 6, y + 2, 5, 0, Math.PI * 2);
    ctx.fill();
}

function drawHostGlow(hostId) {
    if (!state || !state.hosts) return;
    var h = state.hosts[hostId];
    var p = HOSTS[hostId];
    if (!h || !p) return;

    var x = wx(p.x), y = wy(p.y);
    var inf = h.infected;
    var cont = h.containment;

    var halo = COLORS.clean;
    if (inf && cont === 2) halo = COLORS.quarantined;
    else if (inf && cont === 1) halo = COLORS.blocked;
    else if (inf && cont === 0) halo = COLORS.infected;
    else if (!inf && cont > 0) halo = COLORS.falseContain;

    var pulse = 0.6 + 0.4 * Math.sin(animTick * 0.07 + hostId);
    var radius = hostId === 17 ? 20 : 16;

    // Glow ring
    ctx.save();
    ctx.shadowColor = halo;
    ctx.shadowBlur = inf ? 18 * pulse : 8;
    ctx.strokeStyle = halo;
    ctx.lineWidth = inf ? 3 : 2;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // Infected pulse background
    if (inf && cont === 0) {
        ctx.fillStyle = "rgba(244,67,54,0.12)";
        ctx.beginPath();
        ctx.arc(x, y, radius + 4 * pulse, 0, Math.PI * 2);
        ctx.fill();
    }

    // Blocked dashed ring
    if (cont === 1) {
        ctx.save();
        ctx.strokeStyle = COLORS.blocked;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }

    // Quarantine cage bars
    if (cont === 2) {
        ctx.save();
        ctx.strokeStyle = COLORS.quarantined;
        ctx.lineWidth = 1.5;
        for (var bx = x - 14; bx <= x + 14; bx += 6) {
            ctx.beginPath();
            ctx.moveTo(bx, y - 16);
            ctx.lineTo(bx, y + 16);
            ctx.stroke();
        }
        ctx.restore();
    }

    // Name label
    var name = (h.display_name || DISPLAY_NAMES[hostId]);
    var labelColor = COLORS.text;
    if (inf && cont === 0) labelColor = COLORS.infected;
    else if (inf && cont === 1) labelColor = COLORS.blocked;
    else if (inf && cont === 2) labelColor = COLORS.quarantined;

    ctx.fillStyle = labelColor;
    ctx.font = "700 " + Math.max(10, window.innerWidth * 0.007) + "px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(name, x, y + radius + 14);
}



function drawAgents() {
    if (!state || !state.agents) return;

    for (var i = 0; i < state.agents.length; i++) {
        var ag = state.agents[i];
        var base = HOSTS[ag.host];
        if (!base) continue;

        var x = base.x, y = base.y;

        // Smooth transit interpolation
        if (ag.in_transit && ag.target != null && HOSTS[ag.target]) {
            var target = HOSTS[ag.target];
            var t = ag.transit_remaining > 1 ? 0.35 : 0.72;
            t += 0.05 * Math.sin(animTick * 0.08 + i);
            x = x + (target.x - x) * t;
            y = y + (target.y - y) * t;
        }

        // Offset so agents don't sit exactly on desks
        var offsets = [[-3, -5], [3, -5], [0, 5]];
        var sx = wx(x) + offsets[i][0] * 4;
        var sy = wy(y) + offsets[i][1] * 4;

        var r = Math.max(11, window.innerWidth * 0.01);

        // Agent glow circle
        ctx.save();
        ctx.shadowColor = AGENT_COLORS[i];
        ctx.shadowBlur = 12;
        ctx.fillStyle = AGENT_COLORS[i];
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Agent initial letter
        ctx.fillStyle = "#1a1510";
        ctx.font = "800 " + (r * 1.1) + "px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(AGENT_NAMES[i][0], sx, sy + 1);
        ctx.textBaseline = "alphabetic";

        // Name tag
        ctx.fillStyle = "rgba(18,14,10,0.75)";
        roundRect(sx - 28, sy - r - 18, 56, 14, 6);
        ctx.fill();
        ctx.fillStyle = "#f0e8da";
        ctx.font = "700 10px Inter, sans-serif";
        ctx.fillText(AGENT_NAMES[i], sx, sy - r - 8);

        // Transit arrow
        if (ag.in_transit && ag.target_name) {
            ctx.fillStyle = AGENT_COLORS[i];
            ctx.font = "600 " + Math.max(9, window.innerWidth * 0.006) + "px Inter, sans-serif";
            ctx.fillText("-> " + ag.target_name, sx, sy - r - 24);
        }
    }
}


function drawTraffic() {
    if (!state || !state.traffic) return;

    
    var normals = [];
    var malicious = [];
    for (var i = 0; i < state.traffic.length; i++) {
        var t = state.traffic[i];
        if (t.type === "normal") { if (normals.length < 8) normals.push(t); }
        else { if (malicious.length < 14) malicious.push(t); }
    }

    var all = normals.concat(malicious);
    for (var j = 0; j < all.length; j++) {
        var tr = all[j];
        var src = HOSTS[tr.src];
        var dst = tr.dst === -1 ? C2_POS : HOSTS[tr.dst];
        if (!src || !dst) continue;

        var x1 = wx(src.x), y1 = wy(src.y);
        var x2 = wx(dst.x), y2 = wy(dst.y);

        var color = COLORS.normal;
        var width = 1.2;
        var alpha = 0.25;
        var dashed = false;

        if (tr.type === "c2_beacon") { color = COLORS.c2; width = 2; alpha = 0.7; dashed = true; }
        else if (tr.type === "scan") { color = COLORS.scan; width = 1.6; alpha = 0.55; dashed = true; }
        else if (tr.type === "lateral") { color = COLORS.lateral; width = 2; alpha = 0.7; }
        else if (tr.type === "server_attack") { color = COLORS.attack; width = 3; alpha = 0.85; }

        var mx = (x1 + x2) / 2;
        var my = (y1 + y2) / 2 - (tr.type === "normal" ? 12 : 24);

        ctx.save();
        ctx.strokeStyle = withAlpha(color, alpha);
        ctx.lineWidth = width;
        if (dashed) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.quadraticCurveTo(mx, my, x2, y2);
        ctx.stroke();
        ctx.restore();

        // Animated dot moving along the line
        var phase = (animTick * 0.02 + j * 0.12) % 1;
        var pt = quadPoint(x1, y1, mx, my, x2, y2, phase);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, tr.type === "server_attack" ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fill();
    }

    // C2 node indicator
    if (malicious.some(function (t) { return t.type === "c2_beacon"; })) {
        var cx = wx(C2_POS.x), cy = wy(C2_POS.y);
        ctx.save();
        ctx.shadowColor = COLORS.c2;
        ctx.shadowBlur = 14;
        ctx.fillStyle = COLORS.c2;
        ctx.beginPath();
        ctx.arc(cx, cy, 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        ctx.fillStyle = "#fff";
        ctx.font = "700 10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("C2", cx, cy + 4);
    }
}

function quadPoint(x1, y1, cx, cy, x2, y2, t) {
    var mt = 1 - t;
    return {
        x: mt * mt * x1 + 2 * mt * t * cx + t * t * x2,
        y: mt * mt * y1 + 2 * mt * t * cy + t * t * y2
    };
}

function withAlpha(hex, alpha) {
    var bigint = parseInt(hex.slice(1), 16);
    var r = (bigint >> 16) & 255;
    var g = (bigint >> 8) & 255;
    var b = bigint & 255;
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
}


function drawOutcome() {
    if (!state || !state.outcome) return;

    ctx.fillStyle = "rgba(15,12,8,0.65)";
    ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

    var fs = Math.max(28, window.innerWidth * 0.035);
    ctx.font = "800 " + fs + "px Inter, sans-serif";
    ctx.textAlign = "center";

    if (state.outcome === "win") {
        ctx.fillStyle = COLORS.clean;
        ctx.fillText("NETWORK SECURED", window.innerWidth / 2, window.innerHeight / 2 - 10);
        ctx.font = "400 " + (fs * 0.4) + "px Inter, sans-serif";
        ctx.fillStyle = "#a0c0a0";
        ctx.fillText("All infections quarantined — defenders win", window.innerWidth / 2, window.innerHeight / 2 + 30);
    } else if (state.outcome === "loss") {
        ctx.fillStyle = COLORS.infected;
        ctx.fillText("SERVER BREACHED", window.innerWidth / 2, window.innerHeight / 2 - 10);
        ctx.font = "400 " + (fs * 0.4) + "px Inter, sans-serif";
        ctx.fillStyle = "#c0a0a0";
        ctx.fillText("File server compromised — attackers win", window.innerWidth / 2, window.innerHeight / 2 + 30);
    } else {
        ctx.fillStyle = COLORS.blocked;
        ctx.fillText("TIME EXPIRED", window.innerWidth / 2, window.innerHeight / 2 - 10);
        ctx.font = "400 " + (fs * 0.4) + "px Inter, sans-serif";
        ctx.fillStyle = "#c0b090";
        ctx.fillText("Episode truncated at step limit", window.innerWidth / 2, window.innerHeight / 2 + 30);
    }
}


function render() {
    animTick++;
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

    drawBackground();
    drawFloor();
    drawRooms();

    // Draw all desks
    for (var id in HOSTS) {
        var hid = parseInt(id);
        drawDesk(HOSTS[hid].x, HOSTS[hid].y, hid === 17);
    }

    drawTraffic();

    // host state glows
    for (var id2 in HOSTS) {
        drawHostGlow(parseInt(id2));
    }

    drawEffects();
    drawAgents();
    drawOutcome();

    requestAnimationFrame(render);
}

render();