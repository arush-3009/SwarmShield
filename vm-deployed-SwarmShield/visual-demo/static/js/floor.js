const canvas = document.getElementById('floor-canvas');
const ctx = canvas.getContext('2d');
let state = null;
let animFrame = 0;

const C = {
    carpet:'#c4b49a', carpetDark:'#b5a48a', wall:'#e8e0d0', wallLine:'#a09080',
    desk:'#8B7355', deskTop:'#a08a6a', text:'#4a3a2a', textLight:'#7a6a5a',
    clean:'#4CAF50', infected:'#e53935', blocked:'#FF9800', quarantined:'#78909C',
    falsPos:'#AB47BC', agent0:'#FFD700', agent1:'#4488FF', agent2:'#FF6B6B',
    serverRoom:'#d4c4a8', confRoom:'#d0c8b8', michaelOff:'#c8bca8', title:'#5a4a3a',
};

const ROOMS = {
    mainFloor:{x:0.15,y:0.30,w:0.55,h:0.55,label:null},
    michael:{x:0.02,y:0.02,w:0.22,h:0.25,label:"MICHAEL'S OFFICE"},
    conference:{x:0.55,y:0.02,w:0.28,h:0.25,label:"CONFERENCE ROOM"},
    accounting:{x:0.02,y:0.35,w:0.12,h:0.30,label:"ACCOUNTING"},
    server:{x:0.86,y:0.60,w:0.12,h:0.18,label:"SERVER ROOM"},
    reception:{x:0.25,y:0.02,w:0.15,h:0.15,label:"RECEPTION"},
};

const HOST_POS = {
    0:[0.35,0.48], 1:[0.35,0.58], 2:[0.50,0.70], 3:[0.25,0.70], 4:[0.50,0.48],
    5:[0.06,0.42], 6:[0.06,0.52], 7:[0.06,0.62],
    8:[0.65,0.50], 9:[0.65,0.60], 10:[0.65,0.70], 11:[0.65,0.80],
    12:[0.10,0.14], 13:[0.30,0.10],
    14:[0.62,0.10], 15:[0.70,0.14], 16:[0.78,0.10],
    17:[0.92,0.69],
};

const HOST_NAMES = ["Jim","Dwight","Stanley","Phyllis","Andy","Angela","Oscar","Kevin",
    "Desk 1","Desk 2","Desk 3","Desk 4","Michael","Pam","Conf 1","Conf 2","Conf 3","SERVER"];
const AGENT_COLORS = [C.agent0, C.agent1, C.agent2];
const AGENT_NAMES = ["Dwight","Jim","Michael"];

function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
window.addEventListener('resize', resize); resize();

function connectSSE() {
    const src = new EventSource('/stream');
    src.onmessage = function(ev) { const d = JSON.parse(ev.data); if (!d.heartbeat) state = d; };
    src.onerror = function() { src.close(); setTimeout(connectSSE, 2000); };
}
connectSSE();

document.addEventListener('keydown', function(e) {
    const k = e.key;
    if (k === ' ') { e.preventDefault(); post({action:'toggle_pause'}); }
    if (k === 'r' || k === 'R') post({action:'restart'});
    if (k === '+' || k === '=') post({action:'speed_up'});
    if (k === '-' || k === '_') post({action:'speed_down'});
    if (k === 'n' || k === 'N') post({action:'step'});
    if (k === 'd' || k === 'D') post({action:'deterministic', value: !(state&&state.deterministic)});
});
function post(body) { fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }

function px(f){return f*canvas.width;} function py(f){return f*canvas.height;}
function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();}

function drawOffice() {
    ctx.fillStyle=C.carpet; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle=C.carpetDark;
    for(let i=0;i<canvas.width;i+=20)for(let j=0;j<canvas.height;j+=20)if((i+j)%40===0)ctx.fillRect(i,j,10,10);
    for(const[key,room]of Object.entries(ROOMS)){
        const rx=px(room.x),ry=py(room.y),rw=px(room.w),rh=py(room.h);
        let rc=C.carpet; if(key==='michael')rc=C.michaelOff; if(key==='conference')rc=C.confRoom; if(key==='server')rc=C.serverRoom;
        ctx.fillStyle=rc; rr(rx,ry,rw,rh,4); ctx.fill();
        ctx.strokeStyle=C.wallLine; ctx.lineWidth=2.5; rr(rx,ry,rw,rh,4); ctx.stroke();
        if(room.label){ctx.fillStyle=C.textLight;ctx.font=`600 ${Math.max(10,canvas.width*0.009)}px sans-serif`;ctx.textAlign='center';ctx.fillText(room.label,rx+rw/2,ry+14);}
    }
    ctx.strokeStyle=C.wallLine;ctx.lineWidth=3;rr(4,4,canvas.width-8,canvas.height-8,8);ctx.stroke();
}

function drawHost(id) {
    const pos=HOST_POS[id]; if(!pos)return;
    const x=px(pos[0]),y=py(pos[1]),dW=Math.max(36,canvas.width*0.04),dH=dW*0.6;
    let inf=false,cont=0;
    if(state&&state.hosts){const h=state.hosts[id];if(h){inf=h.infected;cont=h.containment;}}
    let glow=C.clean,mc='#30a040';
    if(inf&&cont===2){glow=C.quarantined;mc='#607080';}
    else if(inf&&cont===1){glow=C.blocked;mc='#cc7700';}
    else if(inf){glow=C.infected;mc='#cc2020';}
    else if(cont>0){glow=C.falsPos;mc='#8844aa';}
    if(inf&&cont===0){const p=0.4+0.6*Math.abs(Math.sin(animFrame*0.06));ctx.shadowColor=C.infected;ctx.shadowBlur=15*p;}
    ctx.fillStyle=C.desk;rr(x-dW/2,y-dH/2,dW,dH,3);ctx.fill();
    ctx.fillStyle=C.deskTop;rr(x-dW/2+2,y-dH/2+2,dW-4,dH-4,2);ctx.fill();
    ctx.shadowColor='transparent';ctx.shadowBlur=0;
    const mW=dW*0.5,mH=dH*0.55;
    ctx.fillStyle=mc;rr(x-mW/2,y-mH/2,mW,mH,2);ctx.fill();
    ctx.fillStyle=glow;ctx.globalAlpha=0.3;rr(x-mW/2+1,y-mH/2+1,mW-2,mH-2,1);ctx.fill();ctx.globalAlpha=1;
    if(cont===2){ctx.strokeStyle='#556';ctx.lineWidth=1.5;for(let bx=x-dW/2+4;bx<x+dW/2;bx+=6){ctx.beginPath();ctx.moveTo(bx,y-dH/2);ctx.lineTo(bx,y+dH/2);ctx.stroke();}}
    if(id===17){ctx.fillStyle='#556';const sw=dW*0.3;ctx.fillRect(x-sw/2,y+dH/2+2,sw,4);ctx.fillRect(x-sw/2,y+dH/2+8,sw,4);}
    ctx.fillStyle=C.text;ctx.font=`600 ${Math.max(9,canvas.width*0.008)}px sans-serif`;ctx.textAlign='center';ctx.fillText(HOST_NAMES[id],x,y+dH/2+14);
}

function drawAgents() {
    if(!state||!state.agents)return;
    for(let i=0;i<state.agents.length;i++){
        const ag=state.agents[i],hp=HOST_POS[ag.host]; if(!hp)continue;
        let x=px(hp[0]),y=py(hp[1]);
        const off=[[-20,-25],[20,-25],[0,28]]; x+=off[i][0]; y+=off[i][1];
        const r=Math.max(10,canvas.width*0.012);
        ctx.shadowColor=AGENT_COLORS[i];ctx.shadowBlur=8;
        ctx.fillStyle=AGENT_COLORS[i];ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();
        ctx.shadowColor='transparent';ctx.shadowBlur=0;
        ctx.fillStyle='#000';ctx.font=`700 ${r*1.1}px sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';
        ctx.fillText(AGENT_NAMES[i][0],x,y+1);ctx.textBaseline='alphabetic';
        if(ag.in_transit&&ag.target_name){ctx.fillStyle=AGENT_COLORS[i];ctx.font=`600 ${Math.max(8,canvas.width*0.007)}px sans-serif`;ctx.fillText('→'+ag.target_name,x,y-r-4);}
    }
}

function drawTraffic() {
    if(!state||!state.traffic)return;
    for(const t of state.traffic){
        const sp=HOST_POS[t.src]; if(!sp)continue;
        let dp; if(t.dst===-1)dp=[sp[0],-0.05]; else dp=HOST_POS[t.dst]; if(!dp)continue;
        const sx=px(sp[0]),sy=py(sp[1]),dx=px(dp[0]),dy=py(dp[1]);
        ctx.strokeStyle=t.type==='c2_beacon'?'#ff444488':t.type==='server_attack'?'#ff000099':t.type==='scan'?'#ff880066':'#ff660066';
        ctx.lineWidth=t.type==='server_attack'?2.5:1.5;ctx.setLineDash([4,4]);
        ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(dx,dy);ctx.stroke();ctx.setLineDash([]);
    }
}

function drawHUD() {
    if(!state)return;
    const W=canvas.width,fs=Math.max(12,W*0.011);
    ctx.fillStyle='rgba(44,36,24,0.85)';rr(W*0.15,py(0.90),W*0.70,py(0.08),8);ctx.fill();
    ctx.font=`700 ${fs}px sans-serif`;ctx.textAlign='center';
    const y=py(0.945),gap=W*0.12,cx=W/2;
    ctx.fillStyle='#c0b090';ctx.fillText(`Step ${state.timestep}/${state.max_timesteps}`,cx-gap*2,y);
    ctx.fillStyle=C.infected;ctx.fillText(`Infected: ${state.counts.infected_total}`,cx-gap,y);
    ctx.fillStyle=C.quarantined;ctx.fillText(`Quarantined: ${state.counts.quarantined_total}`,cx,y);
    const dp=(state.server_damage/state.server_damage_max*100).toFixed(0);
    ctx.fillStyle=dp>75?C.infected:dp>40?C.blocked:C.clean;ctx.fillText(`Server: ${dp}% dmg`,cx+gap,y);
    ctx.fillStyle='#c0b090';ctx.fillText(`Ep ${state.episode}  W:${state.wins} L:${state.losses}`,cx+gap*2,y);
    if(state.outcome){
        ctx.fillStyle='rgba(44,36,24,0.7)';ctx.fillRect(0,0,W,canvas.height);
        ctx.font=`700 ${W*0.04}px sans-serif`;ctx.textAlign='center';
        if(state.outcome==='win'){ctx.fillStyle=C.clean;ctx.fillText('NETWORK SECURED',W/2,canvas.height/2-10);ctx.font=`400 ${W*0.015}px sans-serif`;ctx.fillStyle='#a0c0a0';ctx.fillText('All infections quarantined',W/2,canvas.height/2+30);}
        else if(state.outcome==='loss'){ctx.fillStyle=C.infected;ctx.fillText('SERVER BREACHED',W/2,canvas.height/2-10);ctx.font=`400 ${W*0.015}px sans-serif`;ctx.fillStyle='#c0a0a0';ctx.fillText('File server compromised',W/2,canvas.height/2+30);}
        else{ctx.fillStyle=C.blocked;ctx.fillText('TIME EXPIRED',W/2,canvas.height/2-10);}
    }
    if(!state.outcome){ctx.fillStyle=C.title;ctx.font=`700 ${Math.max(14,W*0.013)}px sans-serif`;ctx.textAlign='left';ctx.fillText('SWARMSHIELD — Dunder Mifflin Network Defense',16,22);}
}

function render() {
    animFrame++;ctx.clearRect(0,0,canvas.width,canvas.height);
    drawOffice();for(let i=0;i<18;i++)drawHost(i);drawTraffic();drawAgents();drawHUD();
    requestAnimationFrame(render);
}
render();
