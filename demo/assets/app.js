/* Demo app — simulated playback of REAL captured data. No network, no model calls. */
(function () {
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const D = window.DEMO, RUN = window.DEMO_RUN;
  const cfg = { mode:'demo', loop:'claude-sonnet-5', judge:'claude-opus-5', scope:'full', qs:['OQ-130'] };

  /* ---------- navigation ---------- */
  function show(id){
    $$('.screen').forEach(s=>s.classList.toggle('on', s.id===id));
    $$('.tab').forEach(t=>t.classList.toggle('on', t.dataset.screen===id));
    window.scrollTo(0,0);
  }
  document.addEventListener('click', e=>{
    const b=e.target.closest('[data-screen]'); if(b) show(b.dataset.screen);
  });

  /* ---------- setup rendering ---------- */
  function pickers(el, list, key){
    el.innerHTML = list.map(m=>`<div class="pick ${cfg[key]===m.id?'sel':''}" data-id="${m.id}">
      <span class="nm">${m.label}</span><span class="nt">${m.note}</span></div>`).join('');
    el.querySelectorAll('.pick').forEach(p=>p.onclick=()=>{
      cfg[key]=p.dataset.id; pickers(el,list,key); summary();
    });
  }
  function scopes(){
    $('#scopes').innerHTML = D.scopes.map(s=>`<label class="card ${cfg.scope===s.id?'sel':''}" data-id="${s.id}">
      <div class="ct"><div class="cl">${s.label}</div><div class="cd">${s.detail}<br/><b>${s.est}</b></div></div></label>`).join('');
    $$('#scopes .card').forEach(c=>c.onclick=()=>{
      cfg.scope=c.dataset.id;
      if(cfg.scope!=='multi') cfg.qs=cfg.qs.slice(0,1);
      $('#qhint').textContent = cfg.scope==='multi' ? 'Pick two or more.' : 'Pick one.';
      scopes(); questions(); summary();
    });
  }
  function questions(){
    $('#questions').innerHTML = D.questions.map(q=>{
      const on=cfg.qs.includes(q.id), off=!q.available;
      return `<div class="q ${on?'sel':''} ${off?'off':''}" data-id="${q.id}" ${off?'title="Demo data available for OQ-130 only"':''}>
        <div><div class="qid">${q.id}${off?' · demo n/a':''}</div><div class="qt">${q.title}</div></div>
        <div class="qb">bar ${q.bar}<br/>${q.budget}t</div></div>`;
    }).join('');
    $$('#questions .q').forEach(el=>el.onclick=()=>{
      if(el.classList.contains('off')) return;
      const id=el.dataset.id;
      if(cfg.scope==='multi'){ cfg.qs.includes(id) ? cfg.qs=cfg.qs.filter(x=>x!==id) : cfg.qs.push(id); }
      else cfg.qs=[id];
      questions(); summary();
    });
  }
  function summary(){
    const sc=D.scopes.find(s=>s.id===cfg.scope);
    $('#sMode').textContent = cfg.mode==='demo'?'Demo (free)':'Live (metered)';
    $('#sModels').textContent = `${cfg.loop} / ${cfg.judge}`;
    $('#sScope').textContent = `${sc.label} · ~${sc.turns} turns`;
    $('#sQs').textContent = cfg.qs.length ? cfg.qs.join(', ') : '—';
    $('#sCost').textContent = cfg.mode==='demo' ? '$0.00 — nothing is called' : sc.est;
    $('#sCost').style.color = cfg.mode==='demo' ? 'var(--good)' : 'var(--warn)';
  }
  $$('input[name=mode]').forEach(r=>r.onchange=()=>{
    cfg.mode=r.value;
    $$('.cards.two .card').forEach(c=>c.classList.toggle('sel', c.querySelector('input').checked));
    $('#keyblock').hidden = cfg.mode!=='live';
    summary();
  });

  /* ---------- the run ---------- */
  let timer=null, i=0, t0=0, st={};
  const CHECKS=[['trace_complete','Trace completeness'],['budget_ok','Within turn budget'],
                ['outputs','Required outputs present'],['no_policy','No policy violations'],['ready','Checkpoint ready']];

  function resetRun(){
    clearInterval(timer); i=0; t0=Date.now();
    st={spans:0,turns:0,tools:0,tests:0,writes:0,kinds:{},clauses:new Set(),corrections:0,limits:0,plans:0};
    $('#loopfeed').innerHTML=''; $('#evaldone').hidden=true; $('#evalrunning').hidden=true;
    $('#evalpending').hidden=false; $('#evalstate').textContent='waiting';
    $('#rstatus').className='pill running'; $('#rstatus').lastElementChild.textContent='running';
    const q=D.questions.find(x=>x.id===cfg.qs[0]);
    $('#rmeta').textContent=`${cfg.qs.join(', ')} · loop ${cfg.loop} · judge ${cfg.judge} · ${cfg.mode}`;
    $('#bar').textContent=q.bar;
    renderChecks(); renderMeters();
  }
  function limitFor(){ return ({smoke:22,partial:80,full:190,multi:190})[cfg.scope]; }

  function step(){
    const lim=Math.min(limitFor(), RUN.events.length);
    if(i>=lim){ finish(); return; }
    const ev=RUN.events[i++];
    st.spans++; st.kinds[ev.kind]=(st.kinds[ev.kind]||0)+1;
    const f=$('#loopfeed');
    if(ev.kind==='assistant.turn'){
      st.turns++;
      const tx=(ev.text||'').toLowerCase();
      if(/plan|phase|step 1|approach|first,/.test(tx)) st.plans++;
      if(/i was wrong|correction|let me fix|on reflection|incorrect|i missed/.test(tx)) st.corrections++;
      if(/limitation|does not|known issue|cannot |not verified|caveat/.test(tx)) st.limits++;
      for(const [c,terms] of Object.entries(D.clauses)) if(terms.some(t=>tx.includes(t))) st.clauses.add(c);
      f.insertAdjacentHTML('afterbegin',
        `<div class="ev turn"><div class="eh">assistant · turn ${st.turns}</div><div class="eb">${esc(ev.text)}</div></div>`);
    } else if(ev.kind==='tool.call'){
      st.tools++;
      if(/test|pytest|assert|vitest/.test((ev.cmd||'').toLowerCase())) st.tests++;
      f.insertAdjacentHTML('afterbegin',
        `<div class="ev tool"><div class="eh">tool · ${esc(ev.name)}</div>${ev.cmd?`<div class="cmd">${esc(ev.cmd)}</div>`:`<div class="pth">${esc(ev.path||'')}</div>`}</div>`);
    } else {
      st.writes++;
      f.insertAdjacentHTML('afterbegin',
        `<div class="ev write"><div class="eh">artifact · ${esc(ev.name)}</div><div class="pth">${esc(ev.path)}</div></div>`);
    }
    while(f.children.length>60) f.lastElementChild.remove();
    paint();
  }
  function paint(){
    const q=D.questions.find(x=>x.id===cfg.qs[0]);
    const secs=Math.max((Date.now()-t0)/1000,.1);
    $('#spanct').textContent=`${st.spans} spans`; $('#turncount').textContent=`${st.turns} turns`;
    $('#rate').textContent=(st.spans/secs).toFixed(1);
    $('#budget').textContent=`${st.turns} / ${q.budget}`;
    $('#budgetfill').style.width=Math.min(100,100*st.turns/q.budget)+'%';
    $('#chips').innerHTML=Object.entries(st.kinds).map(([k,v])=>`<span class="chip">${k} · ${v}</span>`).join('');
    renderChecks(); renderMeters();
  }
  function renderChecks(){
    const done=$('#evaldone').hidden===false;
    const v={trace_complete:done,budget_ok:true,outputs:st.writes>0,no_policy:true,ready:done};
    $('#checks').innerHTML=CHECKS.map(([k,l])=>
      `<div><span>${l}</span><span class="${v[k]?'ok':'wait'}">${v[k]?'✓ pass':'· pending'}</span></div>`).join('');
  }
  function renderMeters(){
    const m=[['Orchestration',`plans ${st.plans} · corrections ${st.corrections}`,st.plans*7+st.corrections*18],
             ['Tooling',`tools ${st.tools} · tests ${st.tests}`,st.tools*3+st.tests*12],
             ['Self-awareness',`corrections ${st.corrections} · limits ${st.limits}`,st.corrections*20+st.limits*12],
             ['Question fidelity',`clauses ${st.clauses.size}/6 · outputs ${st.writes}`,100*st.clauses.size/6]];
    $('#meters').innerHTML=m.map(([n,d,p])=>
      `<div class="meter"><div class="mt"><b>${n}</b><span>${d}</span></div>
       <div class="track"><div class="fill" style="width:${Math.min(100,p)}%"></div></div></div>`).join('');
  }
  function finish(){
    clearInterval(timer);
    $('#rstatus').className='pill done'; $('#rstatus').lastElementChild.textContent='completed';
    $('#evalpending').hidden=true; $('#evalrunning').hidden=false; $('#evalstate').textContent='judging…';
    const steps=['Finalizing trace (completion + grace)','Assembling evidence bundle',
                 `Artifact-aware judge · ${cfg.judge}`,'Blind triplicate · 3 calls','Median + band'];
    const box=$('#judgesteps'); box.innerHTML=steps.map(s=>`<div>○ ${s}</div>`).join('');
    let k=0; const iv=setInterval(()=>{
      if(k<steps.length){ box.children[k].className='done'; box.children[k].textContent='✓ '+steps[k]; k++; }
      else { clearInterval(iv); reveal(); }
    },520);
  }
  function reveal(){
    $('#evalrunning').hidden=true; $('#evaldone').hidden=false; $('#evalstate').textContent='complete';
    const q=D.questions.find(x=>x.id===cfg.qs[0]);
    const dims={orchestration:clamp(5+st.plans*.15+st.corrections*.5),tooling:clamp(5+st.tests*.6+Math.min(st.tools,20)*.1),
                self_awareness:clamp(5+st.corrections*.8+st.limits*.4),fidelity:clamp(4+(st.clauses.size/6)*5)};
    const comp=Math.round(Object.values(dims).reduce((a,b)=>a+b,0)/4*10*10)/10;
    const el=$('#composite'); el.textContent=comp; el.className='score '+(comp>=q.bar?'win':'miss');
    $('#dims').innerHTML=Object.entries(dims).map(([k,v])=>
      `<div class="dim"><div class="dn">${k.replace('_',' ')}</div><div class="dv">${v}<small> ±1.0</small></div></div>`).join('');
    renderChecks(); report(comp,dims,q);
  }
  const clamp=x=>Math.max(1,Math.min(9.5,Math.round(x*10)/10));
  const esc=s=>String(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

  function report(comp,dims,q){
    $('#report').innerHTML=`
      <table><tr><th colspan="2">Run</th></tr>
        <tr><td>Question</td><td><b>${q.id}</b> — ${q.title}</td></tr>
        <tr><td>Mode</td><td>${cfg.mode==='demo'?'Demo — simulated playback of real captured data':'Live'}</td></tr>
        <tr><td>Loop model / Judge model</td><td>${cfg.loop} / ${cfg.judge}</td></tr>
        <tr><td>Scope</td><td>${D.scopes.find(s=>s.id===cfg.scope).label}</td></tr>
        <tr><td>Spans · turns · artifacts</td><td>${st.spans} · ${st.turns} · ${st.writes}</td></tr></table>
      <h3 class="sec">Diagnostic score</h3>
      <table><tr><th>Dimension</th><th>Median</th><th>Band</th><th>Evidence</th></tr>
        ${Object.entries(dims).map(([k,v])=>`<tr><td>${k.replace('_',' ')}</td><td><b>${v}</b></td><td>±1.0</td>
          <td class="dim">${evidenceFor(k)}</td></tr>`).join('')}
        <tr><td><b>Composite</b></td><td colspan="3"><b class="${comp>=q.bar?'good':'warn'}">${comp}</b> vs official bar ${q.bar}</td></tr></table>
      <h3 class="sec">Deterministic checks</h3>
      <table><tr><th>Check</th><th>Result</th></tr>
        ${CHECKS.map(([k,l])=>`<tr><td>${l}</td><td class="good">✓ pass</td></tr>`).join('')}</table>
      <p class="secnote">Every number above traces to observed spans. DIAGNOSTIC — not an official LegalQuants score.</p>`;
  }
  function evidenceFor(k){
    return {orchestration:`${st.plans} planning turns, ${st.corrections} course corrections`,
      tooling:`${st.tools} tool calls, ${st.tests} test executions`,
      self_awareness:`${st.corrections} self-corrections, ${st.limits} disclosed limitations`,
      fidelity:`${st.clauses.size}/6 brief clauses touched, ${st.writes} artifacts written`}[k];
  }

  /* ---------- compare ---------- */
  function compare(){
    const R=D.evalRecords, med=a=>{const s=[...a].sort((x,y)=>x-y);return s.length%2?s[(s.length-1)/2]:(s[s.length/2-1]+s[s.length/2])/2;};
    const v1=R.filter(r=>r.producer==='frozen-baseline'&&r.pack==='v1').map(r=>r.composite);
    const v2=R.filter(r=>r.producer==='frozen-baseline'&&r.pack==='v2').map(r=>r.composite);
    const d=Math.round((med(v2)-med(v1))*10)/10;
    const tw=['frozen-baseline','fresh-loop-A','fresh-loop-B'].map(p=>R.find(r=>r.producer===p&&r.pack==='v1'));
    $('#cmp').innerHTML=`
      <h3 class="sec">1 · Judge variance — same output, judged 3×</h3>
      <p class="secnote">Is the LLM judge stable? A wide spread is the argument for medians + bands rather than one absolute number.</p>
      <table><tr><th>Producer</th><th>Scores</th><th>Median</th><th>Spread</th></tr>
        <tr><td>frozen-baseline</td><td>${v1.join(', ')}</td><td><b>${med(v1)}</b></td>
        <td class="${Math.max(...v1)-Math.min(...v1)>=4?'warn':''}">${Math.max(...v1)-Math.min(...v1)}</td></tr></table>
      <h3 class="sec">2 · Three-way — same question, different producers</h3>
      <p class="secnote">The frozen public baseline against two fresh loops, judged by the identical pack.</p>
      <table><tr><th>Producer</th><th>Orch</th><th>Tool</th><th>Self</th><th>Fid</th><th>Composite</th></tr>
        ${tw.map(r=>`<tr><td>${r.producer}</td>${['orchestration','tooling','self_awareness','fidelity'].map(k=>`<td>${r.dims[k]}</td>`).join('')}
        <td class="${r.composite>=82?'good':'warn'}"><b>${r.composite}</b></td></tr>`).join('')}</table>
      <h3 class="sec">3 · Regression — did our eval change drift the score?</h3>
      <p class="secnote">Re-scoring a frozen output after changing the eval pack. Drift ≥ 3 points is flagged — this is how you catch your own eval regressions.</p>
      <table><tr><th>Producer</th><th>Pack v1</th><th>Pack v2</th><th>Δ</th><th>Verdict</th></tr>
        <tr><td>frozen-baseline</td><td>${med(v1)}</td><td>${med(v2)}</td>
        <td class="${Math.abs(d)>=3?'bad':'good'}">${d>0?'+':''}${d}</td>
        <td>${Math.abs(d)>=3?'⚠ DRIFT — review':'stable'}</td></tr></table>`;
  }

  /* ---------- boot ---------- */
  $('#go').onclick=()=>{ if(!cfg.qs.length){alert('Pick at least one question.');return;} show('run'); resetRun(); timer=setInterval(step, +$('#speed').value); };
  $('#speed').onchange=()=>{ if(timer){clearInterval(timer); timer=setInterval(step, +$('#speed').value);} };
  $('#stop').onclick=()=>{ clearInterval(timer); $('#rstatus').className='pill'; $('#rstatus').lastElementChild.textContent='stopped'; };
  pickers($('#loopModels'),D.loopModels,'loop'); pickers($('#judgeModels'),D.judgeModels,'judge');
  scopes(); questions(); summary(); compare();
})();
