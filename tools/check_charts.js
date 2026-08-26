// Chart verification for docs/overview.html.
//  - every chart populates
//  - no SVG text node renders below MIN_PX at either width
//  - no text node's ink escapes its SVG box (the clipping check)
//  - no NaN/undefined in any SVG attribute
//  - no console or page errors
const { chromium } = require('playwright');
const path = require('path');

const FILE = process.argv[2] || path.resolve(__dirname, '../docs/overview.html');
const MIN_PX = 9.0;                 // below this is not readable on a phone
const WIDTHS = [['desktop', 1440], ['phone', 393]];
// Use Playwright's own managed Chromium by default (what CI installs).
// CHROME_PATH overrides it for environments with a preinstalled browser.
const EXEC = process.env.CHROME_PATH || null;

(async () => {
  const browser = await chromium.launch(EXEC ? { executablePath: EXEC } : {});
  let failures = 0;

  for (const [label, width] of WIDTHS) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errs = [];
    page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
    page.on('pageerror', e => errs.push('pageerror: ' + e.message));
    await page.goto('file://' + FILE);
    await page.waitForTimeout(900);

    const res = await page.evaluate((MIN_PX) => {
      const out = { charts: [], tiny: [], clipped: [], nan: [], hidden: [], overlap: [] };
      document.querySelectorAll('svg.chart').forEach(svg => {
        const id = svg.id || '(no id)';
        const vb = svg.viewBox.baseVal;
        const box = svg.getBoundingClientRect();
        // A responsive pair (.pipe-wide/.pipe-narrow) keeps both variants in
        // the DOM and hides one with display:none. The hidden one measures 0,
        // so skip it rather than reporting every label as 0px.
        if (box.width === 0) { out.hidden.push(id); return; }
        const scale = box.width / vb.width;
        const texts = [...svg.querySelectorAll('text')];
        out.charts.push({ id, nodes: svg.childElementCount, texts: texts.length, scale: +scale.toFixed(3) });

        svg.querySelectorAll('*').forEach(node => {
          for (const a of node.attributes) {
            if (/NaN|undefined/i.test(a.value)) out.nan.push(`${id} <${node.tagName} ${a.name}="${a.value}">`);
          }
        });

        // Labels that overlap each other read as one mangled string
        // ("80%Blackout"). They clip nothing, so the box check above misses
        // them entirely; this compares every visible label pair in the chart.
        const rects = texts.map(t => ({ r: t.getBoundingClientRect(), s: (t.textContent || '').trim() }))
          .filter(o => o.r.width > 0 && o.s);
        for (let i = 0; i < rects.length; i++) {
          for (let j = i + 1; j < rects.length; j++) {
            const a = rects[i].r, b = rects[j].r;
            const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left);
            const oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
            if (ox <= 0 || oy <= 0) continue;
            // Text rects include ascent/descent, so labels on adjacent lines
            // always share a pixel or two vertically. A real collision is two
            // labels on roughly the SAME baseline whose glyphs touch: require
            // near-full vertical overlap, then any horizontal overlap at all.
            const hMin = Math.min(a.height, b.height);
            if (oy >= hMin * 0.5 && ox >= 1.5) {
              out.overlap.push({ id, a: rects[i].s.slice(0, 18), b: rects[j].s.slice(0, 18), ox: +ox.toFixed(1) });
            }
          }
        }

        // Text-into-marker collisions. The text-vs-text pass above missed the
        // legend bug entirely: "noisy but usable" did not overlap the next
        // LABEL, it overlapped the next swatch, which is a <rect>. Only small
        // marks are considered — big rects are backgrounds, band fills, row
        // stripes and matrix cells, which text is meant to sit on top of.
        const marks = [...svg.querySelectorAll('rect, circle')]
          .map(n => n.getBoundingClientRect())
          .filter(r => r.width > 0 && r.width * r.height < 900);
        rects.forEach(o => {
          marks.forEach(mr => {
            const ox = Math.min(o.r.right, mr.right) - Math.max(o.r.left, mr.left);
            const oy = Math.min(o.r.bottom, mr.bottom) - Math.max(o.r.top, mr.top);
            if (ox <= 1 || oy <= Math.min(o.r.height, mr.height) * 0.5) return;
            // Text centred on a mark is deliberate — a number inside a step
            // badge, a value inside a matrix cell. A collision is text whose
            // centre sits outside the mark it is running into.
            const cx = (o.r.left + o.r.right) / 2, cy = (o.r.top + o.r.bottom) / 2;
            const centredOn = cx >= mr.left && cx <= mr.right && cy >= mr.top && cy <= mr.bottom;
            if (!centredOn) out.overlap.push({ id, a: o.s.slice(0, 18), b: '<mark>', ox: +ox.toFixed(1) });
          });
        });

        texts.forEach(t => {
          const px = parseFloat(getComputedStyle(t).fontSize) * scale;
          if (px < MIN_PX) out.tiny.push({ id, px: +px.toFixed(2), text: (t.textContent || '').slice(0, 28) });
          const r = t.getBoundingClientRect();
          if (r.width === 0) return;
          const pad = 0.5;
          if (r.left < box.left - pad || r.right > box.right + pad || r.top < box.top - pad || r.bottom > box.bottom + pad) {
            out.clipped.push({ id, text: (t.textContent || '').slice(0, 28),
              over: +Math.max(box.left - r.left, r.right - box.right, box.top - r.top, r.bottom - box.bottom).toFixed(1) });
          }
        });
      });
      return out;
    }, MIN_PX);

    console.log(`\n── ${label} (${width}px) ─────────────────────────────`);
    const empty = res.charts.filter(c => c.nodes === 0);
    console.log(`charts: ${res.charts.length}  hidden: ${res.hidden.length}  empty: ${empty.length}  scale: ${[...new Set(res.charts.map(c => c.scale))].join(', ')}`);
    if (empty.length) { console.log('  EMPTY:', empty.map(c => c.id).join(', ')); failures++; }
    if (res.nan.length) { console.log(`  NaN/undefined attrs: ${res.nan.length}`); res.nan.slice(0, 8).forEach(x => console.log('   ', x)); failures++; }
    if (errs.length) { console.log(`  JS errors: ${errs.length}`); errs.slice(0, 5).forEach(x => console.log('   ', x)); failures++; }

    if (res.tiny.length) {
      failures++;
      const byChart = {};
      res.tiny.forEach(t => { (byChart[t.id] = byChart[t.id] || []).push(t); });
      console.log(`  TEXT BELOW ${MIN_PX}px: ${res.tiny.length} nodes across ${Object.keys(byChart).length} charts`);
      for (const [id, list] of Object.entries(byChart)) {
        const min = Math.min(...list.map(t => t.px));
        console.log(`    ${id.padEnd(14)} ${String(list.length).padStart(3)} nodes, smallest ${min}px  e.g. "${list[0].text}"`);
      }
    } else console.log(`  text size: OK (all >= ${MIN_PX}px)`);

    if (res.clipped.length) {
      failures++;
      console.log(`  CLIPPED: ${res.clipped.length}`);
      res.clipped.slice(0, 12).forEach(c => console.log(`    ${c.id.padEnd(14)} "${c.text}" by ${c.over}px`));
    } else console.log('  clipping: OK');

    if (res.overlap.length) {
      failures++;
      console.log(`  OVERLAPPING LABELS: ${res.overlap.length}`);
      res.overlap.slice(0, 12).forEach(o => console.log(`    ${o.id.padEnd(14)} "${o.a}" x "${o.b}" overlapping by ${o.ox}px`));
    } else console.log('  label overlap: OK');

    await page.close();
  }

  await browser.close();
  console.log(failures ? `\nFAIL (${failures} check groups)` : '\nPASS');
  process.exit(failures ? 1 : 0);
})();
