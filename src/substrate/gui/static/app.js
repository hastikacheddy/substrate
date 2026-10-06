'use strict';
/* substrate GUI. Two views: Run (any experiment: edit inputs, watch the scales, read provenance) and Transfer (the multi-molecule,
   multi-method calibration study). All numbers come from the server; the page only draws them. */

const $ = (sel, root = document) => root.querySelector(sel);
const state = {
  tab: 'run', status: null, experiments: [],
  run: { path: null, info: null, edits: {}, samples: 0, seed: 0, result: null, prev: null, selected: 0, busy: false, auto: true, error: null, token: 0 },
  transfer: { refs: [], selected: new Set(), result: null, cached: null, view: 'matrix', metric: 'rmse', pair: null, rIndex: 0, busy: false, log: [], rates: null, learning: null },
};

/* ---- server ----------------------------------------------------------------------------------------------------------------- */
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let data = null;
  try { data = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}
/** Start a background job and follow it to the end; onLog receives each new message. */
async function jobRun(path, body, onLog) {
  let job = await api(path, body), seen = 0;
  for (;;) {
    job.messages.forEach(m => onLog && onLog(m.text));
    seen = job.n_messages;
    if (job.status === 'done') return job.result;
    if (job.status === 'error') throw new Error(job.error.message);
    await new Promise(r => setTimeout(r, job.elapsed < 3 ? 150 : 500));
    job = await api(`/api/job?id=${job.id}&since=${seen}`);
  }
}

/* ---- formatting -------------------------------------------------------------------------------------------------------------- */
const humanScale = n => n.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
function fmtQ(q) {
  if (q.array) return `array ${q.array.shape.join('×')}` + (q.array.values ? ` [${q.array.values.map(v => fmt(v)).join(', ')}]` : q.array.min != null ? `, ${fmt(q.array.min)} … ${fmt(q.array.max)}` : '');
  let t = fmt(q.value, 4);
  if (q.sigma != null && q.value != null) {
    if (q.band && q.sigma > 0.5 * Math.abs(q.value)) t += `  (68% ${fmt(q.band[0], 2)} to ${fmt(q.band[1], 2)})`;     // the spread rivals the value: a band, not ±
    else t += ` ± ${fmt(q.sigma, 2)}`;
  }
  return t;
}
const unitText = u => (u === '1' || u === 'count' ? '' : u);
function change(prev, now) {
  if (!prev || prev.value == null || now.value == null || prev.array || now.array || prev.value === now.value) return null;
  const a = prev.value, b = now.value;
  if (a > 0 && b > 0) { const r = b / a; if (Math.abs(Math.log10(r)) < 1e-9) return null; return r >= 10 || r <= 0.1 ? `×${fmt(r, 2)}` : `${r >= 1 ? '+' : '−'}${Math.abs((r - 1) * 100).toFixed(1)}%`; }
  return `Δ ${fmt(b - a, 3)}`;
}
const pct = (a, b) => `${Math.round((a / b) * 100)}%`;

/* ---- shell -------------------------------------------------------------------------------------------------------------------- */
function theme(mode) {
  if (mode) { document.documentElement.setAttribute('data-theme', mode); try { localStorage.setItem('substrate-theme', mode); } catch (e) { /* private window */ } }
  window.dispatchEvent(new Event('themechange'));
}
function boot() {
  try { const t = localStorage.getItem('substrate-theme'); if (t) document.documentElement.setAttribute('data-theme', t); } catch (e) { /* no storage */ }
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => theme());
  $('#theme').addEventListener('click', () => theme(isDark() ? 'light' : 'dark'));
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => setTab(t.dataset.tab)));
  api('/api/status').then(st => {
    state.status = st;
    $('#status').replaceChildren(h('span', { class: `dot ${st.pyscf ? 'good' : 'bad'}` }), `PySCF ${st.pyscf ? 'ready' : 'not available'}`, h('span', { class: 'faint', text: ` · v${st.version}` }));
    $('#status').title = st.pyscf ? 'Real quantum-chemistry experiments can run (energies already computed are read from the on-disk cache).'
      : 'Real quantum-chemistry experiments need PySCF (on Windows: scripts/setup_qc_env.sh in WSL). Model experiments run normally.';
  }).catch(e => { $('#status').textContent = 'server unreachable: ' + e.message; });
  api('/api/experiments').then(d => { state.experiments = d.experiments; renderSide(); const first = d.experiments.find(e => !e.invalid && !e.needs_qc && !e.group); if (first && !state.run.path) pick(first.path); });
  setTab('run');
}
function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.tab').forEach(t => t.setAttribute('aria-selected', t.dataset.tab === tab));
  $('#run-view').classList.toggle('hidden', tab !== 'run');
  $('#transfer-view').classList.toggle('hidden', tab !== 'transfer');
  if (tab === 'transfer' && !state.transfer.refs.length) initTransfer();
}

/* ================================================================================================================================
   RUN
   ================================================================================================================================ */
function renderSide() {
  const side = $('#run-side');
  const groups = {};
  state.experiments.forEach(e => (groups[e.group || 'pipelines'] ||= []).push(e));
  const titles = { pipelines: 'Pipelines', references: 'Reference surfaces (real quantum chemistry)' };
  side.replaceChildren(...Object.entries(groups).flatMap(([g, items]) => [
    h('div', { class: 'group-title', text: titles[g] || g }),
    ...items.map(e => h('button', { class: 'exp', 'aria-current': e.path === state.run.path, onclick: () => pick(e.path), title: e.invalid || e.path },
      h('span', { class: 't', text: e.invalid ? e.id : e.name }),
      h('span', { class: 'm' }, e.invalid ? h('span', { class: 'tag err', text: 'unreadable' }) : [h('span', { class: 'faint', text: e.id }), e.needs_qc ? h('span', { class: 'tag qc', text: 'PySCF' }) : null,
        e.propagation.length && e.propagation[e.propagation.length - 1] !== e.scale ? h('span', { text: `${humanScale(e.scale)} → ${humanScale(e.propagation[e.propagation.length - 1])}` }) : h('span', { text: humanScale(e.scale) })])))]));
}
async function pick(path) {
  const r = state.run;
  r.path = path; r.edits = {}; r.result = null; r.prev = null; r.error = null; r.info = null; r.selected = 0;
  renderSide(); renderRun();
  try {
    r.info = await api(`/api/experiment?path=${encodeURIComponent(path)}`);
    r.samples = r.info.samples; r.seed = r.info.seed;
    r.auto = !r.info.needs_qc;
    renderRun();
    if (r.auto) execute();
  } catch (e) { r.error = e.message; renderRun(); }
}
let editTimer = 0;
function scheduleRun() { clearTimeout(editTimer); if (state.run.auto) editTimer = setTimeout(execute, 450); }
async function execute() {
  const r = state.run, token = ++r.token;
  r.busy = true; r.error = null; renderRun();
  const started = performance.now();
  try {
    const result = await jobRun('/api/run', { path: r.path, overrides: r.edits, samples: r.samples, seed: r.seed });
    if (token !== r.token) return;                                            // a newer run superseded this one
    if (r.result && r.result.ok && r.result.path === r.path) r.prev = r.result;
    r.result = result; r.busy = false;
    r.selected = result.ok ? result.stages.length - 1 : result.stages.length;
    if (r.selected < 0) r.selected = 0;
    r.auto = r.auto && (performance.now() - started) < 6000;                  // a slow experiment stops re-running on every keystroke
  } catch (e) { if (token === r.token) { r.busy = false; r.error = e.message; } }
  if (token === r.token) renderRun();
}

function renderRun() {
  const r = state.run, main = $('#run-main');
  if (!r.path) { main.replaceChildren(h('div', { class: 'empty', text: 'Choose an experiment on the left.' })); return; }
  if (r.error && !r.info) { main.replaceChildren(h('div', { class: 'card callout' }, h('div', { class: 'callout-title', text: 'Could not load the experiment' }), r.error)); return; }
  if (!r.info) { main.replaceChildren(h('div', { class: 'empty' }, h('span', { class: 'spin' }))); return; }
  const info = r.info;
  const route = info.propagation.map(humanScale).join(' → ');
  const edited = Object.keys(r.edits).length;
  const desc = h('p', { class: 'desc clamp', text: info.header || '' });
  const more = info.header && info.header.length > 260 ? h('button', { class: 'link', text: 'more', onclick() { desc.classList.toggle('clamp'); more.textContent = desc.classList.contains('clamp') ? 'more' : 'less'; } }) : null;
  const controls = h('div', { class: 'controls' },
    h('button', { class: 'btn', disabled: r.busy, onclick: execute, text: r.busy ? 'Running…' : 'Run' }),
    h('label', { class: 'field' }, 'Ensemble draws', h('input', { type: 'number', min: 0, max: 2000, step: 10, value: r.samples, style: 'width:80px', onchange(e) { r.samples = Math.max(0, Math.min(2000, parseInt(e.target.value) || 0)); scheduleRun(); } })),
    h('label', { class: 'field' }, 'Seed', h('input', { type: 'number', min: 0, step: 1, value: r.seed, style: 'width:70px', onchange(e) { r.seed = parseInt(e.target.value) || 0; scheduleRun(); } })),
    h('label', { class: 'field' }, h('input', { type: 'checkbox', checked: r.auto, onchange(e) { r.auto = e.target.checked; } }), 'run when an input changes'),
    edited ? h('button', { class: 'btn ghost', text: `Reset ${edited} edit${edited > 1 ? 's' : ''}`, onclick() { r.edits = {}; renderRun(); scheduleRun(); } }) : null,
    r.busy ? h('span', { class: 'spin', role: 'status', 'aria-label': 'running' }) : null,
    r.result && !r.busy ? h('span', { class: 'faint', text: `${r.result.ok ? 'ran' : 'stopped'} in ${fmt(r.result.seconds, 3)} s${r.result.samples ? ` with ${r.result.samples} draws` : ''}` }) : null,
    r.info.needs_qc ? h('span', { class: 'faint', text: 'real chemistry: energies are read from the on-disk cache, so a first run of a new geometry can take minutes' }) : null);
  const frame = h('div', { class: r.busy && r.result ? 'refetch' : '' }, r.error ? h('div', { class: 'card callout' }, h('div', { class: 'callout-title', text: 'Run failed' }), r.error) : null, r.result ? renderResult() : (r.busy ? null : h('div', { class: 'empty', text: 'Press Run.' })));
  main.replaceChildren(h('div', { class: 'title' }, h('h1', { text: info.name }), h('div', { class: 'sub', text: `${info.id} · ${humanScale(info.scale)} · ${info.kind} · ${route}` }), desc, more), controls, inputsPanel(), frame);
}

function inputsPanel() {
  const r = state.run, params = Object.entries(r.info.parameters);
  const open = state.run.inputsOpen ?? true;
  const rows = params.map(([name, q]) => {
    const edit = r.edits[name], value = edit?.value ?? q.value, sigma = edit && 'sigma' in edit ? edit.sigma : q.sigma;
    const row = h('div', { class: 'param' + (edit ? ' edited' : '') });
    const vi = h('input', { type: 'number', step: 'any', value, 'aria-label': name, onchange(e) { setEdit(name, q, { value: e.target.valueAsNumber }); } });
    const si = h('input', { type: 'number', step: 'any', class: 'sigma', value: sigma ?? '', placeholder: 'σ', 'aria-label': `${name} uncertainty`, title: 'one-sigma uncertainty (turns on ensembles)', onchange(e) { setEdit(name, q, { sigma: e.target.value === '' ? null : e.target.valueAsNumber }); } });
    row.append(h('label', { text: name, title: name }), h('span', { style: 'display:flex;gap:6px;align-items:center' }, vi, si), h('span', { class: 'unit', text: unitText(q.unit) }));
    return row;
  });
  return h('details', { class: 'panel', open, ontoggle(e) { state.run.inputsOpen = e.target.open; } },
    h('summary', {}, 'Inputs', h('span', { class: 'faint', text: `${params.length} parameters · edit a value or add a σ, and the whole chain re-runs` }), Object.keys(r.edits).length ? h('span', { class: 'edited-note', text: 'edited' }) : null),
    h('div', { class: 'params' }, rows));
}
function setEdit(name, q, patch) {
  const r = state.run, cur = { ...(r.edits[name] || {}) };
  if ('value' in patch) { if (!isFinite(patch.value)) return renderRun(); cur.value = patch.value; }
  if ('sigma' in patch) { if (patch.sigma != null && !(patch.sigma >= 0)) return renderRun(); cur.sigma = patch.sigma; }
  const sameValue = cur.value == null || cur.value === q.value, sameSigma = !('sigma' in cur) || cur.sigma === (q.sigma ?? null);
  if (sameValue && sameSigma) delete r.edits[name]; else r.edits[name] = cur;
  renderRun(); scheduleRun();
}

function renderResult() {
  const r = state.run, res = r.result;
  const solves = res.plan.filter(p => p.kind === 'solve');
  const failedAt = res.ok ? -1 : res.stages.length;
  const warnCount = i => { const st = res.stages[i]; return st ? (st.solve.warnings.length + (st.translate ? st.translate.warnings.length : 0)) : 0; };
  const cards = solves.map((p, i) => {
    const st = res.stages[i], state_ = st ? 'done' : i === failedAt ? 'failed' : 'pending';
    const ms = st ? (st.solve.seconds + (st.translate ? st.translate.seconds : 0)) * 1000 : null;
    return [i ? h('span', { class: 'arrow', text: '→' }) : null,
      h('button', { class: `stage ${state_}`, 'aria-selected': i === r.selected, onclick() { r.selected = i; renderRun(); } },
        h('span', { class: 'sc' }, h('span', { class: `glyph ${state_ === 'done' ? 'ok' : state_ === 'failed' ? 'bad' : ''}`, text: state_ === 'done' ? '✓' : state_ === 'failed' ? '✕' : '·' }), humanScale(p.scale)),
        h('span', { class: 'co', text: p.component }),
        h('span', { class: 'st' }, state_ === 'done' ? `${ms < 100 ? fmt(ms, 2) : Math.round(ms)} ms` : state_ === 'failed' ? 'refused' : 'not reached',
          warnCount(i) ? h('span', { class: 'badge warn', text: `${warnCount(i)} warning${warnCount(i) > 1 ? 's' : ''}` }) : null))];
  });
  const summary = [];
  if (!res.ok) summary.push(h('div', { class: 'card callout', style: 'margin-bottom:14px' }, h('div', { class: 'callout-title', text: res.failed ? `Stopped at the ${humanScale(res.failed.scale)} stage: the physics does not support this hop` : 'The experiment could not be planned' }),
    h('div', { text: res.error.message }), h('p', { class: 'note', text: 'This is a refusal by design: the pipeline stops with a reason instead of inventing a number. Nothing downstream was computed. Change an input above to move out of this regime.' })));
  if (res.ensemble) summary.push(h('div', { class: 'note', style: 'margin-bottom:14px' }, `Ensemble: ${res.ensemble.n_ok} of ${res.ensemble.n_requested} draws valid over ${res.ensemble.varied.length ? res.ensemble.varied.join(', ') : 'no varied parameters'} (seed ${res.ensemble.seed}). Where the spread rivals a value, a 68% band is shown instead of ±.`));
  return h('div', {}, summary, h('div', { class: 'strip', role: 'tablist', 'aria-label': 'scales' }, cards), stageDetail());
}

function stageDetail() {
  const r = state.run, res = r.result, i = r.selected, st = res.stages[i];
  if (!st) {
    const plan = res.plan.filter(p => p.kind === 'solve')[i];
    return h('div', { class: 'card' }, h('h3', { text: plan ? humanScale(plan.scale) : 'Stage' }), h('p', { class: 'muted', text: res.ok ? 'Nothing to show.' : (i === res.stages.length ? 'This is the stage that refused (see above); nothing was computed here.' : 'Not reached: an earlier stage refused.') }));
  }
  const sys = st.system, prev = r.prev && r.prev.stages[i] && r.prev.stages[i].system.kind === sys.kind ? r.prev.stages[i].system : null;
  const obsRows = Object.entries(sys.observables);
  const filter = h('input', { type: 'text', placeholder: 'filter', style: 'width:110px', 'aria-label': 'filter observables', oninput(e) { const q = e.target.value.toLowerCase(); body.querySelectorAll('tr[data-n]').forEach(tr => tr.classList.toggle('hidden', !tr.dataset.n.includes(q))); } });
  const body = h('table', { class: 't' }, h('tr', {}, h('th', { text: 'observable' }), h('th', { class: 'r', text: 'value' }), h('th', { text: 'unit' }), prev ? h('th', { class: 'r', text: 'vs previous run' }) : null),
    obsRows.map(([name, q]) => { const c = prev && prev.observables[name] ? change(prev.observables[name], q) : null;
      return h('tr', { 'data-n': name.toLowerCase() }, h('td', { class: 'mono', text: name }), h('td', { class: 'r', text: fmtQ(q) }), h('td', { class: 'muted', text: unitText(q.unit) }), prev ? h('td', { class: 'r delta' }, c ? c : h('small', { text: '–' })) : null); }));
  const approx = (rec, label) => rec ? h('div', { style: 'margin-bottom:10px' }, h('div', {}, h('b', { text: label }), ' ', h('span', { class: 'mono', text: rec.component })),
    rec.approximations.length ? h('ul', { class: 'plain' }, rec.approximations.map(a => h('li', { text: a }))) : h('p', { class: 'faint', text: 'no approximations declared' }),
    ...rec.warnings.map(w => h('div', { class: 'note warn', text: '⚠ ' + w }))) : null;
  const chain = rec => rec ? h('div', { class: 'mono faint', text: `${rec.input} → ${rec.output}` }) : null;
  return h('div', {}, h('div', { class: 'grid2' }, sys.plots.map(p => plotCard(p))),
    h('div', { class: 'grid2', style: 'margin-top:14px' },
      h('div', { class: 'card' }, h('h3', {}, h('span', { text: `${humanScale(sys.scale)} results` }), h('span', { class: 'grow' }), obsRows.length > 8 ? filter : null), h('div', { class: 'scroll' }, obsRows.length ? body : h('p', { class: 'muted', text: 'This stage reports no observables.' }))),
      h('div', { class: 'card' }, h('h3', { text: 'What this stage assumes' }), approx(st.translate, 'Translated in by'), approx(st.solve, 'Solved by'),
        sys.uncertainty_notes.length ? sys.uncertainty_notes.map(w => h('div', { class: 'note warn', text: '⚠ ' + w })) : null,
        h('h3', { style: 'margin-top:12px', text: 'Lineage' }), st.translate ? chain(st.translate) : null, chain(st.solve), h('p', { class: 'faint', style: 'margin:6px 0 0', text: 'Each product carries the content hash of its input and output, so a result can be traced back to exactly what produced it.' }))));
}

/* ================================================================================================================================
   TRANSFER
   ================================================================================================================================ */
async function initTransfer() {
  const t = state.transfer;
  try {
    t.refs = (await api('/api/references')).references.filter(r => !r.invalid);
    t.selected = new Set(t.refs.map(r => r.name));
    renderTransfer(); checkCached();
  } catch (e) { $('#transfer-main').replaceChildren(h('div', { class: 'card callout' }, e.message)); }
}
async function checkCached() {
  const t = state.transfer, names = [...t.selected];
  if (names.length < 2) { t.result = null; t.cached = false; return renderTransfer(); }
  try {
    const d = await api(`/api/transfer?refs=${names.join(',')}`);
    if (names.join() !== [...t.selected].join()) return;
    t.cached = d.cached; if (d.cached) { t.result = d.result; t.pair = null; } else t.result = null;
  } catch (e) { t.cached = false; }
  renderTransfer();
}
async function computeTransfer(recompute) {
  const t = state.transfer; t.busy = true; t.log = []; renderTransfer();
  try {
    t.result = await jobRun('/api/transfer/run', { refs: [...t.selected], recompute }, m => { t.log.push(m); const el = $('#tlog'); if (el) { el.textContent = t.log.join('\n'); el.scrollTop = el.scrollHeight; } });
    t.cached = true; t.pair = null; t.rates = null; t.learning = null;
  } catch (e) { t.log.push('error: ' + e.message); }
  t.busy = false; renderTransfer();
}
function renderTransfer() {
  const t = state.transfer, main = $('#transfer-main');
  const mols = {};
  t.refs.forEach(r => (mols[r.molecule] ||= []).push(r));
  const chips = Object.entries(mols).map(([m, rs]) => h('div', { style: 'margin-bottom:6px' }, h('span', { class: 'faint', style: 'display:inline-block;min-width:170px', text: m.replace(/_/g, ' ') }),
    h('span', { class: 'chips', style: 'display:inline-flex' }, rs.map(r => h('label', { class: 'chip' + (t.selected.has(r.name) ? ' on' : ''), title: `${r.method}, ${r.range[0]}–${r.range[1]} Å, ${r.grid[0]}×${r.grid[1]} grid` },
      h('input', { type: 'checkbox', checked: t.selected.has(r.name), onchange(e) { e.target.checked ? t.selected.add(r.name) : t.selected.delete(r.name); t.result = null; renderTransfer(); checkCached(); } }), r.method)))));
  const head = h('div', { class: 'title' }, h('h1', { text: 'Do fitted parameters transfer?' }),
    h('p', { class: 'desc', text: 'The valence-bond model is calibrated against each real surface on its own. Each calibration is then applied, with its parameters unchanged, to every other surface: if the parameters meant something beyond one molecule and one level of theory, the off-diagonal cells would be as good as the diagonal.' }));
  const controls = h('div', { class: 'controls' },
    h('button', { class: 'btn', disabled: t.busy || t.selected.size < 2, onclick: () => computeTransfer(false), text: t.busy ? 'Working…' : t.cached ? 'Show' : 'Compute' }),
    t.cached ? h('button', { class: 'btn ghost', disabled: t.busy, onclick: () => computeTransfer(true), text: 'Recompute' }) : null,
    t.busy ? h('span', { class: 'spin' }) : null,
    h('span', { class: 'faint', text: t.selected.size < 2 ? 'choose at least two references' : t.cached ? `saved result${t.result && t.result.created ? ' from ' + t.result.created : ''}${t.result && t.result.seconds ? ` (took ${t.result.seconds} s)` : ''}` : 'not computed yet: reads the scans from the energy cache and fits each (about a minute per few references)' }));
  const log = t.busy || (t.log.length && !t.result) ? h('pre', { class: 'log', id: 'tlog', text: t.log.join('\n') }) : null;
  const body = t.result ? transferBody() : (t.busy ? null : h('div', { class: 'empty', text: t.selected.size < 2 ? 'Choose references above.' : 'Press Compute.' }));
  main.replaceChildren(...[head, h('div', { style: 'margin-top:14px' }, chips), controls, log, body].filter(Boolean));
}

function transferBody() {
  const t = state.transfer, res = t.result;
  const views = [['matrix', 'Cross-prediction'], ['fits', 'Own fits'], ['params', 'Parameters'], ['rates', 'Downstream rates'], ['learning', 'Few-shot learning']];
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'view' }, views.map(([k, label]) => h('button', { 'aria-pressed': t.view === k, text: label, onclick() { t.view = k; renderTransfer(); } })));
  const view = { matrix: matrixView, fits: fitsView, params: paramsView, rates: ratesView, learning: learningView }[t.view](res);
  return h('div', { style: 'margin-top:14px' }, h('div', { style: 'margin-bottom:14px' }, seg), view);
}

/* -- cross-prediction ------------------------------------------------------------------------------------------------------ */
const METRICS = {
  rmse: { label: "RMSE over each target's fit window (eV)", max: 0.6, text: v => v.toFixed(3) },
  barrier: { label: 'mean |barrier error| (eV)', max: 0.6, text: v => v.toFixed(3) },
  wells: { label: 'distances with the wrong number of wells', max: 4, text: v => String(v) },
};
function matrixView(res) {
  const t = state.transfer, m = METRICS[t.metric], names = res.names, grid = res.matrix[t.metric];
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'metric' }, [['rmse', 'Energy error'], ['barrier', 'Barrier error'], ['wells', 'Wells']].map(([k, label]) => h('button', { 'aria-pressed': t.metric === k, text: label, onclick() { t.metric = k; renderTransfer(); } })));
  const table = h('table', { class: 'matrix' });
  table.append(h('tr', {}, h('th'), names.map(n => h('th', { class: 'col', text: n }))));
  names.forEach((s_, i) => {
    table.append(h('tr', {}, h('th', { class: 'row', text: s_ }), names.map((tn, j) => {
      const v = grid[i][j], rgb = v == null ? null : seq(v / m.max), sel = t.pair && t.pair[0] === s_ && t.pair[1] === tn;
      const td = h('td', { class: 'cell' + (i === j ? ' diag' : '') + (sel ? ' selected' : ''), tabindex: 0, style: rgb ? `background:${css(rgb)};color:${inkOn(rgb)}` : '', text: v == null ? '–' : m.text(v),
        title: `${s_} → ${tn}`, onclick() { t.pair = [s_, tn]; t.rIndex = Math.min(t.rIndex, res.refs[j].r.length - 1); renderTransfer(); }, onkeydown(e) { if (e.key === 'Enter') td.click(); } });
      return td;
    })));
  });
  if (t.metric !== 'wells') {
    const key = t.metric === 'rmse' ? 'constant_rmse' : 'mean_barrier', label = t.metric === 'rmse' ? 'constant guess' : 'no-barrier guess';
    if (t.metric === 'rmse') table.append(h('tr', { class: 'base' }, h('th', { class: 'row', text: 'fit window (eV)' }), res.refs.map(r => h('td', { text: fmt(r.window), title: 'each target is scored over the energy window its own fit used' }))));
    table.append(h('tr', { class: 'base' }, h('th', { class: 'row', text: label }), res.refs.map(r => h('td', { text: fmt(r.baselines[key], 3), title: t.metric === 'rmse' ? 'rmse of predicting the single best constant energy' : 'mean barrier: the error of predicting no barrier' }))));
  }
  const key = h('div', { class: 'scale-key' }, '0', h('canvas', { width: 180, height: 1 }), `≥ ${m.max}`, h('span', { text: m.label }));
  requestAnimationFrame(() => { const c = key.querySelector('canvas').getContext('2d'), img = c.createImageData(180, 1); for (let i = 0; i < 180; i++) img.data.set([...seq(i / 179), 255], i * 4); c.putImageData(img, 0, 0); });
  const read = h('div', { class: 'note' }, 'Row = whose parameters, column = whose real energies. The diagonal (outlined) is each reference fitted to itself. Anything much worse than the diagonal is a failure to transfer; “constant guess” is what predicting one number for every point would score, so a cell at or above it carries no information. Click a cell to see why.');
  const left = h('div', { class: 'card', style: 'overflow:auto' }, h('div', { class: 'plotbar', style: 'margin-bottom:10px' }, seg), table, key, read);
  return h('div', {}, left, t.pair ? pairDetail(res) : h('p', { class: 'muted', text: 'Select a cell for the surface it was asked to predict.' }));
}
function pairDetail(res) {
  const t = state.transfer, [sn, tn] = t.pair, si = res.names.indexOf(sn), ti = res.names.indexOf(tn), target = res.refs[ti], pair = res.pairs[`${sn}|${tn}`], own = res.pairs[`${tn}|${tn}`];
  const j = Math.min(t.rIndex, target.r.length - 1), col = a => a.map(row => row[j]);
  const slice = { type: 'line', title: `${tn} at R = ${fmt(target.r[j], 3)} Å`, xlabel: 'proton position (Å)', ylabel: 'energy above the minimum (eV)', x: target.x, ymax: Math.max(2.0, target.window + 0.5),
    series: [{ name: `${tn} (real energies)`, y: col(target.e), dots: true, color: 'var(--ink)' }, { name: `${sn}'s parameters`, y: col(pair.model), slot: 0 }, ...(sn !== tn ? [{ name: `${tn}'s own fit`, y: col(own.model), dashed: true, slot: 1 }] : [])] };
  const barriers = { type: 'line', title: 'Barrier against heavy-atom distance', xlabel: 'R (Å)', ylabel: 'barrier (eV)', x: target.r,
    series: [{ name: 'real', y: target.barrier, dots: true, color: 'var(--ink)' }, { name: `${sn}'s parameters`, y: pair.barrier, slot: 0 }, ...(sn !== tn ? [{ name: `${tn}'s own fit`, y: own.barrier, dashed: true, slot: 1 }] : [])] };
  const slider = h('input', { type: 'range', min: 0, max: target.r.length - 1, value: j, 'aria-label': 'heavy-atom distance', oninput(e) { t.rIndex = +e.target.value; const host = $('#pair-slice'); host.replaceChildren(plotCard(buildSlice())); $('#r-read').textContent = `R = ${fmt(target.r[t.rIndex], 3)} Å`; } });
  const buildSlice = () => { const jj = Math.min(t.rIndex, target.r.length - 1), c2 = a => a.map(row => row[jj]); return { ...slice, title: `${tn} at R = ${fmt(target.r[jj], 3)} Å`, series: [{ ...slice.series[0], y: c2(target.e) }, { ...slice.series[1], y: c2(pair.model) }, ...(sn !== tn ? [{ ...slice.series[2], y: c2(own.model) }] : [])] }; };
  const stats = h('table', { class: 't' }, h('tr', {}, h('th', { text: '' }), h('th', { class: 'r', text: `${sn}'s parameters` }), sn !== tn ? h('th', { class: 'r', text: `${tn}'s own fit` }) : null),
    [[`rmse ≤ ${fmt(target.window)} eV`, p => fmt(p.rmse[String(target.window)], 3)], ['worst error', p => fmt(p.max_error[String(target.window)], 3)], ['mean |barrier error|', p => fmt(p.barrier_error, 3)], ['wrong well count', p => String(p.wells_wrong)], ['energy-zero shift', p => fmt(p.shift, 3)]].map(([l, f]) => h('tr', {}, h('td', { text: l }), h('td', { class: 'r', text: f(pair) }), sn !== tn ? h('td', { class: 'r', text: f(own) }) : null)));
  return h('div', { style: 'margin-top:14px' },
    h('h2', { style: 'margin-bottom:8px;font-size:16px', text: sn === tn ? `${sn} fitted to itself` : `${sn}'s parameters asked to predict ${tn}` }),
    h('div', { class: 'grid2' },
      h('div', {}, h('div', { class: 'field', style: 'margin-bottom:8px' }, 'distance', slider, h('span', { id: 'r-read', class: 'num', text: `R = ${fmt(target.r[j], 3)} Å` })), h('div', { id: 'pair-slice' }, plotCard(slice))),
      h('div', {}, plotCard(barriers), h('div', { class: 'card', style: 'margin-top:14px' }, h('h3', { text: 'Scores (energy zero shifted to the best constant)' }), stats))));
}

/* -- own fits and parameters ------------------------------------------------------------------------------------------------- */
function fitsView(res) {
  const rows = res.refs.map(r => { const c = r.calibration, own = res.pairs[`${r.name}|${r.name}`];
    return h('tr', {}, h('td', { text: r.name }), h('td', { class: 'r', text: fmt(r.window) }), h('td', { class: 'r', text: fmt(own.rmse['0.5'], 3) }), h('td', { class: 'r', text: fmt(own.rmse['1'], 3) }), h('td', { class: 'r', text: fmt(own.rmse[String(r.window)], 3) }), h('td', { class: 'r', text: fmt(own.max_error[String(r.window)], 3) }),
      h('td', { class: 'r', text: fmt(own.barrier_error, 3) }), h('td', { class: 'r', text: String(own.wells_wrong) }),
      h('td', { text: Object.keys(c.pinned).length ? Object.keys(c.pinned).join(', ') : '–' }), h('td', { class: 'r', text: fmt(c.reduced_chi2, 2) }), h('td', { class: 'r', text: `${c.starts_agreeing}/${c.starts}` })); });
  return h('div', { class: 'card', style: 'overflow:auto' }, h('table', { class: 't' }, h('tr', {}, ['reference', 'fit window', 'rmse ≤0.5', '≤1.0', '≤ window', 'worst', 'barrier err', 'wrong wells', 'on a bound', 'χ²', 'starts'].map((x, i) => h('th', { class: i && i < 8 || i > 8 ? 'r' : '', text: x }))), rows),
    h('div', { class: 'note' }, 'Errors in eV over the reference points in each window above the surface minimum; the fit window is declared per reference (an asymmetric surface needs a wider one so the fit sees its barrier). Barriers are measured from the donor-side well. “On a bound” lists parameters the data could not pin: widening that bound changes no cross-prediction RMSE by more than 0.008 eV, but those parameters then mean nothing individually. “Starts” is how many of the multi-start fits reached the best answer.'));
}
function paramsView(res) {
  const names = [...new Set(res.refs.flatMap(r => Object.keys(r.calibration.parameters)))];
  const head = h('tr', {}, h('th', { text: 'parameter' }), res.refs.map(r => h('th', { class: 'r', text: r.name })));
  const rows = names.map(n => h('tr', {}, h('td', { class: 'mono', text: n }), res.refs.map(r => { const c = r.calibration, bound = n in c.pinned;
    if (!(n in c.parameters)) return h('td', { class: 'r faint', text: '0 (fixed)', title: 'a symmetric reference has no energy offset between its two sides' });
    return h('td', { class: 'r', title: bound ? `on its ${c.pinned[n]} bound: not determined by the data` : '' }, fmt(c.parameters[n], 3), c.sigma[n] != null ? h('small', { class: 'faint', text: ` ±${fmt(c.sigma[n], 2)}` }) : h('small', { class: 'err', text: ' bound' })); })));
  return h('div', { class: 'card', style: 'overflow:auto' }, h('table', { class: 't' }, head, rows),
    h('div', { class: 'note' }, 'The X–H length (morse_r_eq) and Morse width (morse_alpha) look alike across all nine, but the coupling and the O···O well, which decide the barrier, differ by factors of 2 to 10. ± is the fit uncertainty, not the model error. These are effective parameters of a fixed functional form.'));
}

/* -- downstream rates -------------------------------------------------------------------------------------------------------- */
function ratesView(res) {
  const t = state.transfer, names = res.names;
  const level = h('input', { type: 'number', min: 0.05, max: 1.2, step: 0.05, value: t.rates ? t.rates.barrier : 0.4, style: 'width:80px', id: 'barrier' });
  const run = h('button', { class: 'btn', disabled: t.busy, text: 'Run the real chain', onclick: async () => {
    t.busy = true; t.log = []; renderTransfer();
    try { t.rates = await jobRun('/api/transfer/rates', { refs: names, barrier: parseFloat($('#barrier').value) }, m => { t.log.push(m); const el = $('#tlog'); if (el) { el.textContent = t.log.join('\n'); el.scrollTop = el.scrollHeight; } }); } catch (e) { t.log.push('error: ' + e.message); }
    t.busy = false; renderTransfer(); } });
  const controls = h('div', { class: 'controls' }, h('label', { class: 'field' }, 'at the distance where the target\'s barrier is', level, 'eV'), run,
    h('span', { class: 'faint', text: 'the real chain (quantum route, 300 K) for each target against the same chain run on each calibration; new geometries cost real quantum chemistry' }));
  if (!t.rates) return h('div', {}, controls, h('div', { class: 'empty', text: 'Not run yet.' }));
  const rt = t.rates, table = h('table', { class: 'matrix' });
  table.append(h('tr', {}, h('th'), rt.names.map(n => h('th', { class: 'col', text: n }))));
  rt.names.forEach((sn, i) => table.append(h('tr', {}, h('th', { class: 'row', text: sn }), rt.names.map((tn, j) => {
    const c = rt.cells[`${sn}|${tn}`], v = c && c.ratio;
    if (!v) return h('td', { class: 'cell', text: '–', title: (c && c.note) || (rt.real[tn] && rt.real[tn].note) || 'no rate', style: 'cursor:default' });
    const rgb = diverge(Math.log10(v) / 3);
    return h('td', { class: 'cell' + (i === j ? ' diag' : ''), style: `background:${css(rgb)};color:${inkOn(rgb)}`, text: v >= 100 || v < 0.01 ? v.toExponential(0).replace('e+', 'e') : v.toFixed(2), title: `${sn} → ${tn}: model rate ${fmt(c.k_model)} /s, real ${fmt(rt.real[tn].k)} /s; isotope effect ${fmt(c.kie_model, 3)} vs ${fmt(rt.real[tn].kie, 3)}` });
  }))));
  table.append(h('tr', { class: 'base' }, h('th', { class: 'row', text: 'real k_H (1/s)' }), rt.names.map(n => h('td', { text: rt.real[n] && rt.real[n].k ? fmt(rt.real[n].k, 2) : '–' }))));
  table.append(h('tr', { class: 'base' }, h('th', { class: 'row', text: 'at R (Å)' }), rt.names.map(n => h('td', { text: rt.real[n] && rt.real[n].distance ? fmt(rt.real[n].distance, 3) : '–' }))));
  return h('div', {}, controls, h('div', { class: 'card', style: 'overflow:auto' }, h('h3', { text: `Model rate ÷ real rate, target barrier ${rt.barrier} eV` }), table,
    h('div', { class: 'note' }, 'Row = whose parameters, column = whose real chain. 1 means the calibrated model reproduces the real rate; red = the model is too fast, blue = too slow, and the colour saturates at a factor of 1000. “–” means a chain refused (hover for why: usually the barrier is below the zero-point energy, so there is no reactant state and no rate to compare).')));
}

/* -- few-shot learning --------------------------------------------------------------------------------------------------------- */
function learningView(res) {
  const t = state.transfer, names = res.names;
  const target = h('select', { id: 'l-target', 'aria-label': 'target' }, names.map(n => h('option', { value: n, text: n, selected: t.learning ? t.learning.target === n : n === names[names.length - 1] })));
  const source = h('select', { id: 'l-source', 'aria-label': 'source' }, h('option', { value: '', text: '(none: from scratch only)' }), names.map(n => h('option', { value: n, text: n, selected: t.learning ? t.learning.source === n : n === names[0] })));
  const widths = h('input', { type: 'text', id: 'l-widths', value: '0.1, 0.3, 1', style: 'width:110px', 'aria-label': 'prior widths' });
  const run = h('button', { class: 'btn', disabled: t.busy, text: 'Run', onclick: async () => {
    t.busy = true; t.log = []; renderTransfer();
    try { t.learning = await jobRun('/api/transfer/learning', { refs: names, target: $('#l-target').value, source: $('#l-source').value || null, sigmas: $('#l-widths').value.split(',').map(x => parseFloat(x)).filter(x => x > 0), counts: [0, 1, 2, 3, 6] }, m => t.log.push(m)); } catch (e) { t.log.push('error: ' + e.message); }
    t.busy = false; renderTransfer(); } });
  const controls = h('div', { class: 'controls' }, h('label', { class: 'field' }, 'target', target), h('label', { class: 'field' }, 'prior from', source), h('label', { class: 'field' }, 'prior widths (relative)', widths), run);
  const guide = h('div', { class: 'note' }, 'Fit the target from only k of its heavy-atom distances (the even-numbered ones of the grid) and score the five odd-numbered ones, which are never trained on. k = 0 is the source’s parameters unchanged. A prior of relative width s holds each parameter near the source’s value with a Gaussian of s × its size. From scratch, one distance cannot be fitted at all (the line starts at k = 2). The widths are not tuned on the test distances: all are shown.');
  if (!t.learning) return h('div', {}, controls, h('div', { class: 'empty', text: 'Not run yet.' }), guide);
  const L = t.learning, ks = L.counts;
  const at = (arr, k) => { const p = arr.find(q => q.k === k); return p && p.rmse != null && !p.note ? p.rmse : null; };
  const series = [{ name: 'from scratch', y: ks.map(k => at(L.scratch, k)), slot: 1 }, ...Object.entries(L.priors).map(([s_, arr], i) => ({ name: `prior from ${L.source}, s = ${s_}`, y: ks.map(k => at(arr, k)), slot: i === 0 ? 0 : i + 1 }))];
  const plot = { type: 'line', title: `${L.target}: held-out error against distances used`, xlabel: 'target heavy-atom distances used in the fit (k)', ylabel: 'RMSE on held-out distances (eV)', x: ks, xticks: ks, series, ymax: 0.5, hlines: L.floor != null ? [{ y: L.floor, label: 'best this model form does' }] : [] };
  return h('div', {}, controls, plotCard(plot), guide);
}

boot();
