'use strict';
/* Chart primitives for the substrate GUI: plain SVG and canvas, no dependencies. Every label that comes from data goes in through
   textContent. Each chart has a hover layer (crosshair or per-cell tooltip), a legend when there are two or more series, and a table view. */

const SVGNS = 'http://www.w3.org/2000/svg';

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'text') el.textContent = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  for (const kid of kids.flat(Infinity)) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}

/* ---- numbers ------------------------------------------------------------------------------------------------------------------- */
function fmt(v, digits = 4) {
  if (v == null || (typeof v === 'number' && !isFinite(v))) return '–';
  if (typeof v !== 'number') return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return v.toExponential(digits - 1).replace(/\.?0+e/, 'e').replace('e+', 'e');
  return String(parseFloat(v.toPrecision(digits)));
}
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo, raw = span / n, pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * pow).find(c => c >= raw) || 10 * pow;
  const out = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return out;
}
const tickFmt = ticks => {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1, big = Math.max(...ticks.map(Math.abs));
  return t => (big >= 1e5 || (big < 1e-2 && big > 0) ? t.toExponential(1).replace('e+', 'e') : String(parseFloat(t.toFixed(Math.max(0, 1 - Math.floor(Math.log10(step)))))));
};

/* ---- colour ---------------------------------------------------------------------------------------------------------------------- */
const BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
const hex = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
function isDark() {
  const t = document.documentElement.getAttribute('data-theme');
  return t === 'dark' || (t !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);
}
/** Sequential: one hue, light to dark with magnitude (reversed on a dark surface so small values recede). */
function seq(t) {
  t = Math.min(1, Math.max(0, t));
  if (isDark()) t = 1 - t;
  const x = t * (BLUE.length - 1), i = Math.min(BLUE.length - 2, Math.floor(x));
  return mix(hex(BLUE[i]), hex(BLUE[i + 1]), x - i);
}
/** Diverging: blue (t < 0) and red (t > 0) around a neutral grey. */
function diverge(t) {
  t = Math.min(1, Math.max(-1, t));
  const mid = hex(isDark() ? '#383835' : '#f0efec'), pole = hex(t < 0 ? (isDark() ? '#3987e5' : '#2a78d6') : (isDark() ? '#e66767' : '#e34948'));
  return mix(mid, pole, Math.abs(t));
}
const css = rgb => `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
const inkOn = rgb => ((0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255 > 0.55 ? '#0b0b0b' : '#ffffff');
const SERIES = i => `var(--s${(i % 8) + 1})`;
const colorOf = (sr, i) => sr.color || SERIES(sr.slot ?? i);

/* ---- shared plumbing ------------------------------------------------------------------------------------------------------------ */
/** Call draw() now, on resize, and when the theme changes; stop when the element leaves the page. */
function autoDraw(el, draw) {
  let frame = 0;
  const run = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(() => { if (el.isConnected) draw(); }); };
  new ResizeObserver(run).observe(el);
  const onTheme = () => { if (!el.isConnected) window.removeEventListener('themechange', onTheme); else run(); };
  window.addEventListener('themechange', onTheme);
  run();
}
function tooltip(host) {
  const tip = h('div', { class: 'tip hidden' });
  host.append(tip);
  return {
    show(html, px, py) {
      tip.replaceChildren(...html);
      tip.classList.remove('hidden');
      const w = host.clientWidth, tw = tip.offsetWidth;
      tip.style.left = Math.max(0, Math.min(w - tw, px + 14)) + 'px';
      tip.style.top = Math.max(0, py - tip.offsetHeight - 10) + 'px';
    },
    hide() { tip.classList.add('hidden'); },
  };
}
const tipHead = text => h('div', { class: 'h', text });
const tipRow = (color, value, name, dash) => h('div', { class: 'row' }, color ? h('i', { style: `border-color:${color};${dash ? 'border-top-style:dashed;' : ''}` }) : null, h('b', { text: value }), h('span', { class: 'muted', text: name || '' }));

function legend(series) {
  if (series.length < 2) return null;
  return h('div', { class: 'legend' }, series.map((sr, i) => h('span', {}, h('i', { class: sr.dots ? 'dots' : sr.dashed ? 'dash' : '', style: sr.dots ? `background:${colorOf(sr, i)}` : `border-color:${colorOf(sr, i)}` }), sr.name)));
}

/* ---- line plot ----------------------------------------------------------------------------------------------------------------- */
function linePlot(spec) {
  const root = h('div', { class: 'chart' });
  root.append(legend(spec.series) || '');
  const host = h('div', { class: 'chart' });
  root.append(host);
  const tip = tooltip(host);
  const xs = spec.x, H = spec.height || 250, M = { l: 54, r: 14, t: 8, b: 36 };
  let cursor = null;
  const draw = () => {
    const W = host.clientWidth || 520, pw = W - M.l - M.r, ph = H - M.t - M.b;
    const allY = spec.series.flatMap(sr => sr.y.filter(v => v != null)).concat((spec.hlines || []).map(l => l.y));
    let y0 = Math.min(...allY), y1 = Math.max(...allY);
    if (spec.ymax != null) y1 = Math.min(y1, spec.ymax);
    const pad = (y1 - y0) * 0.05 || 1; y0 -= pad; y1 += pad;
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const X = v => M.l + (xmax > xmin ? (v - xmin) / (xmax - xmin) : 0.5) * pw, Y = v => M.t + (1 - (v - y0) / (y1 - y0)) * ph;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, height: H, tabindex: 0, role: 'img', 'aria-label': spec.title });
    const clip = 'c' + Math.random().toString(36).slice(2);
    svg.append(s('defs', {}, s('clipPath', { id: clip }, s('rect', { x: M.l, y: M.t, width: pw, height: ph }))));
    const yt = niceTicks(y0, y1, 5), xt = spec.xticks || niceTicks(xmin, xmax, Math.max(2, Math.floor(pw / 90)));
    const yf = tickFmt(yt), xf = tickFmt(xt);
    yt.forEach(t => svg.append(s('line', { class: 'gridline', x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t) }), s('text', { x: M.l - 6, y: Y(t) + 4, 'text-anchor': 'end' }, yf(t))));
    xt.forEach(t => svg.append(s('text', { x: X(t), y: H - M.b + 15, 'text-anchor': 'middle' }, xf(t))));
    svg.append(s('line', { class: 'ax', x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }), s('text', { x: M.l + pw / 2, y: H - 4, 'text-anchor': 'middle' }, spec.xlabel || ''),
      s('text', { x: 12, y: M.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${M.t + ph / 2})` }, spec.ylabel || ''));
    const body = s('g', { 'clip-path': `url(#${clip})` });
    (spec.hlines || []).forEach(l => body.append(s('line', { x1: M.l, x2: W - M.r, y1: Y(l.y), y2: Y(l.y), stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 3' }),
      s('text', { x: W - M.r - 4, y: Y(l.y) - 3, 'text-anchor': 'end' }, l.label || '')));
    spec.series.forEach((sr, i) => {
      const sx = sr.x || xs, color = colorOf(sr, i);
      if (sr.dots) {
        sr.y.forEach((v, k) => v != null && body.append(s('circle', { cx: X(sx[k]), cy: Y(v), r: 4, fill: color, stroke: 'var(--surface)', 'stroke-width': 2 })));
        return;
      }
      let d = '', pen = false;
      sr.y.forEach((v, k) => { if (v == null) { pen = false; return; } d += `${pen ? 'L' : 'M'}${X(sx[k]).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; });
      body.append(s('path', { d, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': sr.dashed ? '6 4' : null }));
    });
    svg.append(body);
    const cross = s('line', { y1: M.t, y2: H - M.b, stroke: 'var(--ink-3)', 'stroke-width': 1, class: 'hidden' });
    svg.append(cross);
    const nearest = px => { const xv = xmin + ((px - M.l) / pw) * (xmax - xmin); let best = 0; xs.forEach((v, k) => { if (Math.abs(v - xv) < Math.abs(xs[best] - xv)) best = k; }); return best; };
    const show = k => {
      cursor = k; const cx = X(xs[k]);
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.classList.remove('hidden');
      const rows = [tipHead(`${spec.xlabel || 'x'} = ${fmt(xs[k])}`)];
      spec.series.forEach((sr, i) => {
        const sx = sr.x || xs; let j = 0; sx.forEach((v, m) => { if (Math.abs(v - xs[k]) < Math.abs(sx[j] - xs[k])) j = m; });
        if (sr.y[j] != null && (!sr.x || Math.abs(sx[j] - xs[k]) < 1e-9 + (xmax - xmin) / (2 * xs.length))) rows.push(tipRow(colorOf(sr, i), fmt(sr.y[j]), sr.name, sr.dashed));
      });
      tip.show(rows, cx, M.t + ph / 2);
    };
    const hide = () => { cross.classList.add('hidden'); tip.hide(); cursor = null; };
    svg.addEventListener('pointermove', e => { const r = svg.getBoundingClientRect(); const px = (e.clientX - r.left) * (W / r.width); if (px >= M.l && px <= W - M.r) show(nearest(px)); else hide(); });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('keydown', e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); show(Math.min(xs.length - 1, Math.max(0, (cursor ?? 0) + (e.key === 'ArrowRight' ? 1 : -1)))); } else if (e.key === 'Escape') hide(); });
    svg.addEventListener('blur', hide);
    host.querySelector('svg')?.remove();
    host.prepend(svg);
  };
  autoDraw(host, draw);
  return root;
}

/* ---- heat map ------------------------------------------------------------------------------------------------------------------- */
function heatmapPlot(spec) {
  const root = h('div', { class: 'chart' });
  const host = h('div', { class: 'chart' });
  root.append(host);
  const tip = tooltip(host);
  const nx = spec.x.length, ny = spec.y.length, H = spec.height || 280, M = { l: 54, r: 14, t: 8, b: 36 };
  const flat = spec.z.flat().filter(v => v != null);
  const lo = spec.min ?? Math.min(...flat), hi = spec.clip ?? spec.max ?? Math.max(...flat);
  const bar = h('div', { class: 'colorbar' }, h('span', { text: fmt(lo) }), h('canvas', { width: 256, height: 1 }), h('span', { text: (spec.clip != null ? '≥ ' : '') + fmt(hi) }), h('span', { class: 'cap', text: spec.zlabel || '' }));
  root.append(bar);
  const canvas = h('canvas', { style: 'position:absolute;left:0;top:0' });
  host.append(canvas);
  const draw = () => {
    const W = host.clientWidth || 520, pw = W - M.l - M.r, ph = H - M.t - M.b, dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(pw * dpr); canvas.height = Math.round(ph * dpr);
    canvas.style.cssText = `position:absolute;left:${M.l}px;top:${M.t}px;width:${pw}px;height:${ph}px;border-radius:2px`;
    const small = document.createElement('canvas'); small.width = nx; small.height = ny;
    const img = small.getContext('2d').createImageData(nx, ny);
    for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
      const v = spec.z[ix][iy], p = ((ny - 1 - iy) * nx + ix) * 4, c = v == null ? (isDark() ? [44, 44, 42] : [225, 224, 217]) : (spec.diverging ? diverge(((v - (spec.mid ?? 0)) / (hi - (spec.mid ?? 0))) || 0) : seq((v - lo) / (hi - lo || 1)));
      img.data.set([c[0], c[1], c[2], 255], p);
    }
    small.getContext('2d').putImageData(img, 0, 0);
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = spec.smooth !== false; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
    const cb = bar.querySelector('canvas').getContext('2d'), grad = cb.createImageData(256, 1);
    for (let i = 0; i < 256; i++) { const c = spec.diverging ? diverge((i / 255) * 2 - 1) : seq(i / 255); grad.data.set([c[0], c[1], c[2], 255], i * 4); }
    cb.putImageData(grad, 0, 0);
    const xmin = spec.x[0], xmax = spec.x[nx - 1], ymin = spec.y[0], ymax = spec.y[ny - 1];
    const dx = (xmax - xmin) / (nx - 1 || 1), dy = (ymax - ymin) / (ny - 1 || 1);
    const X = v => M.l + ((v - (xmin - dx / 2)) / (nx * dx || 1)) * pw, Y = v => M.t + (1 - (v - (ymin - dy / 2)) / (ny * dy || 1)) * ph;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': spec.title, style: 'position:relative' });
    const xt = niceTicks(xmin, xmax, Math.max(2, Math.floor(pw / 90))), yt = niceTicks(ymin, ymax, 5), xf = tickFmt(xt), yf = tickFmt(yt);
    xt.forEach(t => svg.append(s('text', { x: X(t), y: H - M.b + 15, 'text-anchor': 'middle' }, xf(t))));
    yt.forEach(t => svg.append(s('text', { x: M.l - 6, y: Y(t) + 4, 'text-anchor': 'end' }, yf(t))));
    svg.append(s('text', { x: M.l + pw / 2, y: H - 4, 'text-anchor': 'middle' }, spec.xlabel || ''), s('text', { x: 12, y: M.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${M.t + ph / 2})` }, spec.ylabel || ''));
    const mark = s('rect', { fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2, class: 'hidden' });
    svg.append(mark);
    svg.addEventListener('pointermove', e => {
      const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * (W / r.width), py = (e.clientY - r.top) * (H / r.height);
      if (px < M.l || px > M.l + pw || py < M.t || py > M.t + ph) { mark.classList.add('hidden'); tip.hide(); return; }
      const ix = Math.min(nx - 1, Math.max(0, Math.floor(((px - M.l) / pw) * nx))), iy = Math.min(ny - 1, Math.max(0, Math.floor((1 - (py - M.t) / ph) * ny)));
      mark.setAttribute('x', M.l + (ix * pw) / nx); mark.setAttribute('y', M.t + ((ny - 1 - iy) * ph) / ny); mark.setAttribute('width', pw / nx); mark.setAttribute('height', ph / ny); mark.classList.remove('hidden');
      tip.show([tipHead(`${fmt(spec.x[ix])}, ${fmt(spec.y[iy])}`), tipRow(null, fmt(spec.z[ix][iy]), spec.zlabel)], px, py);
    });
    svg.addEventListener('pointerleave', () => { mark.classList.add('hidden'); tip.hide(); });
    host.querySelector('svg')?.remove();
    host.append(svg);
    host.style.height = H + 'px';
  };
  autoDraw(host, draw);
  return root;
}

/* ---- bars ----------------------------------------------------------------------------------------------------------------------- */
function barPlot(spec) {
  const host = h('div', { class: 'chart' });
  const tip = tooltip(host);
  const n = spec.values.length, H = spec.height || 210, M = { l: 54, r: 14, t: 18, b: 40 };
  const draw = () => {
    const W = host.clientWidth || 420, pw = W - M.l - M.r, ph = H - M.t - M.b;
    const lo = Math.min(0, ...spec.values.filter(v => v != null)), hi = Math.max(0, ...spec.values.filter(v => v != null)), pad = (hi - lo) * 0.08 || 1;
    const y0 = lo < 0 ? lo - pad : 0, y1 = hi + pad, Y = v => M.t + (1 - (v - y0) / (y1 - y0)) * ph;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': spec.title });
    const yt = niceTicks(y0, y1, 4), yf = tickFmt(yt);
    yt.forEach(t => svg.append(s('line', { class: 'gridline', x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t) }), s('text', { x: M.l - 6, y: Y(t) + 4, 'text-anchor': 'end' }, yf(t))));
    svg.append(s('line', { class: 'ax', x1: M.l, x2: W - M.r, y1: Y(0), y2: Y(0) }), s('text', { x: 12, y: M.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${M.t + ph / 2})` }, spec.ylabel || ''));
    const band = pw / n, bw = Math.min(24, band * 0.6);
    spec.values.forEach((v, i) => {
      if (v == null) return;
      const cx = M.l + band * (i + 0.5), top = Math.min(Y(v), Y(0)), hgt = Math.max(1, Math.abs(Y(v) - Y(0)));
      const r = Math.min(4, hgt / 2), up = v >= 0;
      const d = up ? `M${cx - bw / 2},${top + hgt} V${top + r} Q${cx - bw / 2},${top} ${cx - bw / 2 + r},${top} H${cx + bw / 2 - r} Q${cx + bw / 2},${top} ${cx + bw / 2},${top + r} V${top + hgt} Z`
                   : `M${cx - bw / 2},${top} V${top + hgt - r} Q${cx - bw / 2},${top + hgt} ${cx - bw / 2 + r},${top + hgt} H${cx + bw / 2 - r} Q${cx + bw / 2},${top + hgt} ${cx + bw / 2},${top + hgt - r} V${top} Z`;
      const bar = s('path', { d, fill: 'var(--s1)' });
      svg.append(bar, s('text', { x: cx, y: up ? top - 4 : top + hgt + 12, 'text-anchor': 'middle', style: 'fill:var(--ink-2)' }, fmt(v, 3)),
        s('text', { x: cx, y: H - M.b + 16, 'text-anchor': 'middle' }, spec.labels[i].length > 12 ? spec.labels[i].slice(0, 11) + '…' : spec.labels[i]));
      const hit = s('rect', { x: M.l + band * i, y: M.t, width: band, height: ph, fill: 'transparent', tabindex: 0, 'aria-label': `${spec.labels[i]}: ${fmt(v)}` });
      const on = () => { bar.setAttribute('fill', 'var(--s1)'); bar.setAttribute('opacity', 0.8); tip.show([tipHead(spec.labels[i]), tipRow(null, fmt(v), spec.ylabel)], cx, Y(v)); };
      const off = () => { bar.removeAttribute('opacity'); tip.hide(); };
      hit.addEventListener('pointermove', on); hit.addEventListener('pointerleave', off); hit.addEventListener('focus', on); hit.addEventListener('blur', off);
      svg.append(hit);
    });
    host.querySelector('svg')?.remove();
    host.prepend(svg);
  };
  autoDraw(host, draw);
  return host;
}

/* ---- a plot in a card, with a table view ------------------------------------------------------------------------------------- */
function tableOf(spec) {
  const cap = 300;
  if (spec.type === 'bars') return h('table', { class: 't' }, h('tr', {}, h('th', { text: 'name' }), h('th', { class: 'r', text: spec.ylabel || 'value' })), spec.labels.map((l, i) => h('tr', {}, h('td', { text: l }), h('td', { class: 'r', text: fmt(spec.values[i], 6) }))));
  if (spec.type === 'line') {
    const idx = spec.x.map((_, i) => i), step = Math.max(1, Math.ceil(idx.length / cap));
    const shared = spec.series.filter(sr => !sr.x);
    return h('table', { class: 't' }, h('tr', {}, h('th', { class: 'r', text: spec.xlabel || 'x' }), shared.map(sr => h('th', { class: 'r', text: sr.name }))),
      idx.filter(i => i % step === 0).map(i => h('tr', {}, h('td', { class: 'r', text: fmt(spec.x[i], 6) }), shared.map(sr => h('td', { class: 'r', text: fmt(sr.y[i], 6) })))));
  }
  const ys = spec.y;
  return h('table', { class: 't' }, h('tr', {}, h('th', { class: 'r', text: `${spec.xlabel || 'x'} ↓ / ${spec.ylabel || 'y'} →` }), ys.map(y => h('th', { class: 'r', text: fmt(y, 4) }))),
    spec.x.map((x, ix) => h('tr', {}, h('td', { class: 'r', text: fmt(x, 4) }), ys.map((_, iy) => h('td', { class: 'r', text: fmt(spec.z[ix][iy], 4) })))));
}
function plotCard(spec, extra) {
  const view = h('div', {});
  const draw = spec.type === 'heatmap' ? heatmapPlot : spec.type === 'bars' ? barPlot : linePlot;
  const chart = draw(spec);
  let table = false;
  const toggle = h('button', { class: 'link', 'aria-pressed': 'false', text: 'table', onclick() {
    table = !table; toggle.textContent = table ? 'chart' : 'table'; toggle.setAttribute('aria-pressed', table);
    view.replaceChildren(table ? h('div', { class: 'scroll' }, tableOf(spec)) : chart);
  } });
  view.append(chart);
  return h('div', { class: 'card' }, h('h3', {}, h('span', { text: spec.title }), h('span', { class: 'grow' }), extra || null, toggle), view);
}
