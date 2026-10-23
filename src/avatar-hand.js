// avatar-hand.js — lift the hand out of the picture so it can wave.
//
//  1. MediaPipe HandLandmarker gives 21 points on the hand (0 = wrist).
//  2. MASK: the area around those points, minus pixels that look like the
//     backdrop (sampled just outside the hand), minus anything past the wrist.
//  3. HAND LAYER: the original pixels, with the mask as transparency.
//  4. BACKGROUND LAYER: the picture with the hand painted out ("inpainting"):
//     we fill the hole from its edges inward, ring by ring, each pixel taking
//     the average of its already-known neighbours. Then when the hand moves,
//     there's no second "ghost" hand left behind.
// Works on plain {data, width, height} pixel arrays, so it's testable without a browser.

const smooth = (e0, e1, x) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

function convexHull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), q) <= 0) lower.pop(); lower.push(q); }
  for (const q of p.reverse()) { while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), q) <= 0) upper.pop(); upper.push(q); }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function inPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

const grow = (poly, cx, cy, by) => poly.map(([x, y]) => {
  const d = Math.hypot(x - cx, y - cy) || 1;
  return [x + ((x - cx) / d) * by, y + ((y - cy) / d) * by];
});

/**
 * @param {{data: Uint8ClampedArray, width: number, height: number}} img
 * @param {{x:number,y:number}[]} hand 21 normalised landmarks
 * @returns {{ handRGBA: Uint8ClampedArray, bgRGBA: Uint8ClampedArray, pivot: [number, number], size: number, colorKeyed: boolean }}
 */
export function liftHand(img, hand) {
  const { data, width: W, height: Hh } = img;
  const pts = hand.map((p) => [p.x * W, p.y * Hh]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const cx = xs.reduce((a, b) => a + b) / 21, cy = ys.reduce((a, b) => a + b) / 21;
  const hull = convexHull(pts);
  const inner = grow(hull, cx, cy, 0.14 * size);   // fingers are wider than their centre-line points
  const outer = grow(hull, cx, cy, 0.24 * size);   // ring for sampling the backdrop colour

  // Hand direction: wrist → middle-finger knuckle. Pixels behind the wrist are arm, not hand.
  const [wx, wy] = pts[0], [mx, my] = pts[9];
  const al = Math.hypot(mx - wx, my - wy) || 1;
  const ax = (mx - wx) / al, ay = (my - wy) / al;

  // Working area: the outer ring plus room for the flood fill below.
  const pad = 0.15 * size;
  const x0 = Math.max(0, Math.floor(Math.min(...outer.map((p) => p[0])) - pad)), x1 = Math.min(W - 1, Math.ceil(Math.max(...outer.map((p) => p[0])) + pad));
  const y0 = Math.max(0, Math.floor(Math.min(...outer.map((p) => p[1])) - pad)), y1 = Math.min(Hh - 1, Math.ceil(Math.max(...outer.map((p) => p[1])) + pad));

  // Backdrop colour = median of the ring between `inner` and `outer`, on the finger side of the wrist.
  const ring = [[], [], []];
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if (inPoly(x, y, inner) || !inPoly(x, y, outer)) continue;
    if ((x - wx) * ax + (y - wy) * ay < 0.2 * size) continue;
    const k = (y * W + x) * 4;
    for (let c = 0; c < 3; c++) ring[c].push(data[k + c]);
  }
  const med = ring.map((v) => { v.sort((a, b) => a - b); return v[v.length >> 1] ?? 0; });
  const spread = ring[0].length ? ring.reduce((s, v) => s + (v[Math.floor(v.length * 0.9)] - v[Math.floor(v.length * 0.1)]), 0) / 3 : 255;

  // Alpha mask.
  const alpha = new Float32Array(W * Hh);
  let sum = 0, n = 0;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if (!inPoly(x, y, inner)) continue;
    const along = (x - wx) * ax + (y - wy) * ay;
    const wrist = smooth(-0.12 * size, 0.02 * size, along);
    const k = (y * W + x) * 4;
    const d = Math.hypot(data[k] - med[0], data[k + 1] - med[1], data[k + 2] - med[2]);
    alpha[y * W + x] = wrist * smooth(22, 55, d);
    sum += alpha[y * W + x]; n++;
  }
  // If the backdrop is busy or similar to skin, colour keying fails: use the shape alone.
  const colorKeyed = spread < 60 && n > 0 && sum / n > 0.3;

  // The palm is wider than its tracking points (especially near the wrist), so
  // also take nearby pixels that match the hand's own colour.
  if (colorKeyed) {
    const hc = [[], [], []];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (alpha[y * W + x] < 0.95) continue;
      const k = (y * W + x) * 4;
      for (let c = 0; c < 3; c++) hc[c].push(data[k + c]);
    }
    const handCol = hc.map((v) => { v.sort((a, b) => a - b); return v[v.length >> 1] ?? 0; });
    // Flood fill: spread from the hand into touching pixels that look like hand, not backdrop.
    const queue = [];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (alpha[y * W + x] > 0.5) queue.push(y * W + x);
    while (queue.length) {
      const i = queue.pop(), x = i % W, y = (i / W) | 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const xx = x + dx, yy = y + dy, j = yy * W + xx;
        if (xx < x0 || xx > x1 || yy < y0 || yy > y1 || alpha[j] > 0.5) continue;
        const k = j * 4;
        const dHand = Math.hypot(data[k] - handCol[0], data[k + 1] - handCol[1], data[k + 2] - handCol[2]);
        const dBack = Math.hypot(data[k] - med[0], data[k + 1] - med[1], data[k + 2] - med[2]);
        const wrist = smooth(-0.12 * size, 0.02 * size, (xx - wx) * ax + (yy - wy) * ay);
        const a = dHand < dBack ? wrist * (1 - smooth(30, 70, dHand)) : 0;
        if (a > alpha[j]) { alpha[j] = a; if (a > 0.5) queue.push(j); }
      }
    }
  }
  if (!colorKeyed) {
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (!inPoly(x, y, inner)) continue;
      alpha[y * W + x] = smooth(-0.12 * size, 0.02 * size, (x - wx) * ax + (y - wy) * ay);
    }
  }
  boxBlur(alpha, W, Hh, x0, y0, x1, y1, 1);
  boxBlur(alpha, W, Hh, x0, y0, x1, y1, 1);

  const handRGBA = new Uint8ClampedArray(data.length);
  for (let i = 0; i < W * Hh; i++) {
    if (alpha[i] <= 0) continue;
    handRGBA.set([data[i * 4], data[i * 4 + 1], data[i * 4 + 2], Math.round(alpha[i] * 255)], i * 4);
  }

  // Background with the hand painted out. The hole is a bit bigger than the hand
  // so its soft edge leaves no halo.
  const hole = new Uint8Array(W * Hh);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    let a = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < W && yy < Hh) a = Math.max(a, alpha[yy * W + xx]);
    }
    if (a > 0.02) hole[y * W + x] = 1;
  }
  const bgRGBA = inpaint(data, hole, W, Hh, x0, y0, x1, y1);
  return { handRGBA, bgRGBA, hole, pivot: [wx / W, wy / Hh], size: size / Math.max(W, Hh), colorKeyed };
}

function boxBlur(a, W, H, x0, y0, x1, y1, r) {
  const src = a.slice();
  for (let y = Math.max(0, y0 - r); y <= Math.min(H - 1, y1 + r); y++) {
    for (let x = Math.max(0, x0 - r); x <= Math.min(W - 1, x1 + r); x++) {
      let s = 0, n = 0;
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < W && yy < H) { s += src[yy * W + xx]; n++; }
      }
      a[y * W + x] = s / n;
    }
  }
}

/** Onion-peel inpainting: fill the hole ring by ring from the outside in. */
export function inpaint(data, hole, W, H, x0 = 0, y0 = 0, x1 = W - 1, y1 = H - 1) {
  const out = new Uint8ClampedArray(data);
  const missing = hole.slice();
  let left = missing.reduce((s, v) => s + v, 0);
  while (left > 0) {
    const fill = [];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (!missing[y * W + x]) continue;
      let r = 0, g = 0, b = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if ((dx || dy) && xx >= 0 && yy >= 0 && xx < W && yy < H && !missing[yy * W + xx]) {
          const k = (yy * W + xx) * 4;
          r += out[k]; g += out[k + 1]; b += out[k + 2]; n++;
        }
      }
      if (n) fill.push([y * W + x, r / n, g / n, b / n]);
    }
    if (!fill.length) break; // hole touches nothing known (shouldn't happen)
    for (const [i, r, g, b] of fill) { out.set([r, g, b, 255], i * 4); missing[i] = 0; }
    left -= fill.length;
  }
  return out;
}
