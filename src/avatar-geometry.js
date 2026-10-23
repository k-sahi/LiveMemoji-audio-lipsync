// avatar-geometry.js — turn 478 MediaPipe landmarks on a picture into an
// animatable 3D face mesh. Pure math, no Three.js, so it can be tested on its own.
//
//  1. Landmarks → 3D points. x, y come from the picture; z (depth) is MediaPipe's
//     estimate of how far each point is from the camera. That depth is what makes
//     a flat picture into a relief you can turn.
//  2. Delaunay triangulation connects the points into triangles. We drop the
//     triangles inside the inner lips, leaving a HOLE where the mouth opens.
//  3. If the picture's lips are parted (e.g. a smiling Memoji), we pull them
//     together so the avatar's resting mouth is closed.
//  4. Every vertex gets weights: how much it moves with the jaw, the upper lip,
//     rounding ("oo"), and each smile corner. deform() mixes those by the
//     predicted blendshapes, every frame.

// Landmark index loops from MediaPipe's face mesh topology.
export const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
  377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
export const LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
export const LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
const LOWER_LIP = new Set([324, 318, 402, 317, 14, 87, 178, 88, 95, 375, 321, 405, 314, 17, 84, 181, 91, 146]);
const MOUTH_CORNERS = new Set([61, 291, 78, 308]);
// Eyes: upper lid, lower lid, and the two corners. "Right" = the person's right (picture's left).
const EYES = [
  { upper: [246, 161, 160, 159, 158, 157, 173], lower: [7, 163, 144, 145, 153, 154, 155], corners: [33, 133] },
  { upper: [466, 388, 387, 386, 385, 384, 398], lower: [249, 390, 373, 374, 380, 381, 382], corners: [263, 362] },
];
const BROWS = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55, 300, 293, 334, 296, 336, 276, 283, 282, 295, 285];
const TOP = 10, CHIN = 152, UPPER_MID = 13, LOWER_MID = 14, CORNER_R = 61, CORNER_L = 291;
const NUM_MESH_POINTS = 468; // 468–477 are iris points; not needed for the mesh

const smooth = (e0, e1, x) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

// ---- Delaunay triangulation (Bowyer–Watson) -------------------------------------
// Adds points one by one; every triangle whose circumcircle contains the new
// point is removed, and the hole is re-filled with triangles to the new point.
export function delaunay(pts) {
  const n = pts.length;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of pts) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
  const d = Math.max(maxX - minX, maxY - minY) * 20;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const P = [...pts, [cx - d, cy - d], [cx + d, cy - d], [cx, cy + d]]; // super-triangle
  const circ = (a, b, c) => {
    const [ax, ay] = P[a], [bx, by] = P[b], [qx, qy] = P[c];
    const D = 2 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by));
    const ux = ((ax * ax + ay * ay) * (by - qy) + (bx * bx + by * by) * (qy - ay) + (qx * qx + qy * qy) * (ay - by)) / D;
    const uy = ((ax * ax + ay * ay) * (qx - bx) + (bx * bx + by * by) * (ax - qx) + (qx * qx + qy * qy) * (bx - ax)) / D;
    return { a, b, c, ux, uy, r2: (ax - ux) ** 2 + (ay - uy) ** 2 };
  };
  let tris = [circ(n, n + 1, n + 2)];
  for (let i = 0; i < n; i++) {
    const [px, py] = P[i];
    const bad = [], keep = [];
    for (const t of tris) ((px - t.ux) ** 2 + (py - t.uy) ** 2 < t.r2 ? bad : keep).push(t);
    const edges = new Map();
    for (const t of bad) {
      for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
        const key = u < v ? `${u},${v}` : `${v},${u}`;
        edges.set(key, edges.has(key) ? null : [u, v]); // shared edges cancel out
      }
    }
    for (const e of edges.values()) if (e) keep.push(circ(e[0], e[1], i));
    tris = keep;
  }
  return tris.filter((t) => t.a < n && t.b < n && t.c < n).map((t) => [t.a, t.b, t.c]);
}

/** y on a polyline [[x,y],…] sorted by x (clamped at the ends). */
export function lerpCurve(pts, x) {
  if (x <= pts[0][0]) return pts[0][1];
  for (let j = 1; j < pts.length; j++) {
    if (x <= pts[j][0]) {
      const [ax, ay] = pts[j - 1], [bx, by] = pts[j];
      return ay + ((x - ax) / (bx - ax || 1)) * (by - ay);
    }
  }
  return pts[pts.length - 1][1];
}

export function pointInPolygon([x, y], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * @param {{x:number,y:number,z:number}[]} landmarks from FaceLandmarker (normalised to the image)
 * @param {number} aspect image width / height
 * @param {{depth?: number}} opts depth exaggeration (1 = as measured)
 */
export function buildFace(landmarks, aspect, { depth = 1 } = {}) {
  const N = Math.min(NUM_MESH_POINTS, landmarks.length);
  // Scene units: picture is `aspect` wide and 1 tall, centred on the origin, y up, +z toward the viewer.
  const pos = new Float32Array(N * 3);
  const uv = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    const { x, y, z } = landmarks[i];
    pos[i * 3] = (x - 0.5) * aspect;
    pos[i * 3 + 1] = 0.5 - y;
    pos[i * 3 + 2] = -z * aspect * depth;
    uv[i * 2] = x;
    uv[i * 2 + 1] = 1 - y;
  }
  const P = (i) => [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];

  // ---- triangles: Delaunay on the 2D layout, minus the mouth hole and anything outside the face oval
  const flat = Array.from({ length: N }, (_, i) => [pos[i * 3], pos[i * 3 + 1]]);
  const ovalPoly = FACE_OVAL.map((i) => flat[i]);
  const innerPoly = LIPS_INNER.map((i) => flat[i]);
  const tris = delaunay(flat).filter(([a, b, c]) => {
    const centroid = [(flat[a][0] + flat[b][0] + flat[c][0]) / 3, (flat[a][1] + flat[b][1] + flat[c][1]) / 3];
    return pointInPolygon(centroid, ovalPoly) && !pointInPolygon(centroid, innerPoly);
  });
  // Consistent winding (counter-clockwise seen from the front) so no triangle is culled.
  const indices = new Uint32Array(tris.length * 3);
  tris.forEach(([a, b, c], k) => {
    const cross = (flat[b][0] - flat[a][0]) * (flat[c][1] - flat[a][1]) - (flat[b][1] - flat[a][1]) * (flat[c][0] - flat[a][0]);
    indices.set(cross > 0 ? [a, b, c] : [a, c, b], k * 3);
  });

  // ---- face measurements
  const mean = (idx, k) => idx.reduce((s, i) => s + pos[i * 3 + k], 0) / idx.length;
  const C = { x: mean(LIPS_INNER, 0), y: mean(LIPS_INNER, 1), z: mean(LIPS_INNER, 2) };
  const H = Math.hypot(P(TOP)[0] - P(CHIN)[0], P(TOP)[1] - P(CHIN)[1]);
  const Wm = Math.hypot(P(CORNER_L)[0] - P(CORNER_R)[0], P(CORNER_L)[1] - P(CORNER_R)[1]) / 2;
  let zBack = Infinity;
  for (let i = 0; i < N; i++) zBack = Math.min(zBack, pos[i * 3 + 2]);
  zBack -= 0.004;
  // Depth eases to zero toward the face outline: the outline sits flat on the
  // background picture, and the relief rises gently inside it. A sudden step
  // at the edge would make the cheeks overhang the background when the head turns.
  const ovalPts = FACE_OVAL.map((i) => [pos[i * 3], pos[i * 3 + 1]]);
  const edgeDist = (x, y) => {
    let best = Infinity;
    for (let k = 0; k < ovalPts.length; k++) {
      const [ax, ay] = ovalPts[k], [bx, by] = ovalPts[(k + 1) % ovalPts.length];
      const t = Math.max(0, Math.min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2 || 1)));
      best = Math.min(best, Math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay)));
    }
    return best;
  };
  for (let i = 0; i < N; i++) {
    const ease = smooth(0, 0.12 * H, edgeDist(pos[i * 3], pos[i * 3 + 1]));
    pos[i * 3 + 2] = zBack + (pos[i * 3 + 2] - zBack) * ease;
  }
  for (const i of FACE_OVAL) pos[i * 3 + 2] = zBack;

  // ---- per-vertex movement weights
  const w = {
    jaw: new Float32Array(N), upper: new Float32Array(N), lower: new Float32Array(N),
    round: new Float32Array(N), cornerL: new Float32Array(N), cornerR: new Float32Array(N),
  };
  const cL = P(CORNER_L), cR = P(CORNER_R);
  const sigma2 = 2 * (0.6 * Wm) ** 2;
  for (let i = 0; i < N; i++) {
    const [X, Y] = P(i);
    const dx = X - C.x, dy = Y - C.y;
    const r = Math.hypot(dx / Wm, dy / (0.8 * Wm));
    const lipRegion = 1 - smooth(1.0, 2.0, r);
    const centre = 1 - smooth(0.55, 1.05, Math.abs(dx) / Wm);    // 1 mid-lip, 0 at the corners
    const below = 1 - smooth(-0.005 * H, -0.06 * H, -dy);         // ~1 well below the mouth
    const narrow = 1 - smooth(1.3 * Wm, 2.8 * Wm, Math.abs(dx));  // fades toward the ears
    w.jaw[i] = LOWER_LIP.has(i) ? 1 : MOUTH_CORNERS.has(i) ? 0.35 : dy < 0 ? smooth(0, 1, below) * narrow : 0;
    w.upper[i] = dy >= 0 ? lipRegion * centre : 0;
    w.lower[i] = dy < 0 ? lipRegion * centre : 0;
    w.round[i] = lipRegion;
    w.cornerL[i] = Math.exp(-((X - cL[0]) ** 2 + (Y - cL[1]) ** 2) / sigma2);
    w.cornerR[i] = Math.exp(-((X - cR[0]) ** 2 + (Y - cR[1]) ** 2) / sigma2);
  }
  for (const i of FACE_OVAL) { w.cornerL[i] = w.cornerR[i] = w.round[i] = 0; }

  // ---- close parted lips in the resting pose
  // The inner-lip loop pairs up: each upper-lip point has a lower-lip partner
  // straight below it (191↔95, 80↔88, … 13↔14 … 415↔324). For each pair we know
  // the local midline and half-gap, so a smile that's wider on one side closes
  // all the way to both corners, not just in the middle.
  const gap = P(UPPER_MID)[1] - P(LOWER_MID)[1];
  let closedBy = 0;
  if (gap > 0.012 * H) {
    closedBy = gap / 2;
    const n = LIPS_INNER.length; // 20: index 0 and 10 are the corners
    const knots = [{ x: P(LIPS_INNER[0])[0], mid: P(LIPS_INNER[0])[1], half: 0 }];
    for (let k = 1; k < n / 2; k++) {
      const up = P(LIPS_INNER[k]), lo = P(LIPS_INNER[n - k]);
      knots.push({ x: (up[0] + lo[0]) / 2, mid: (up[1] + lo[1]) / 2, half: Math.max(0, (up[1] - lo[1]) / 2 - 0.001 * H) });
    }
    knots.push({ x: P(LIPS_INNER[n / 2])[0], mid: P(LIPS_INNER[n / 2])[1], half: 0 });
    knots.sort((a, b) => a.x - b.x);
    const at = (x) => { // piecewise-linear interpolation along the mouth; zero beyond the corners
      if (x <= knots[0].x || x >= knots[knots.length - 1].x) return null;
      let j = 1;
      while (knots[j].x < x) j++;
      const a = knots[j - 1], b = knots[j], t = (x - a.x) / (b.x - a.x || 1);
      return { mid: a.mid + t * (b.mid - a.mid), half: a.half + t * (b.half - a.half) };
    };
    for (let i = 0; i < N; i++) {
      const k = at(pos[i * 3]);
      if (!k || k.half <= 0) continue;
      const dy = pos[i * 3 + 1] - k.mid;
      // Full strength on the lips themselves, fading out over the skin beyond.
      const reach = 1 - smooth(k.half, k.half + 0.12 * H, Math.abs(dy));
      pos[i * 3 + 1] += (dy >= 0 ? -1 : 1) * k.half * reach;
    }
  }

  // ---- eyes (for blinking) and brows (for raising)
  // Each lid is a curve y(x); a blink squeezes the eye opening down onto the
  // lower lid, and stretches the skin above it so nothing tears.
  const curve = (idx, corners) => [...corners, ...idx]
    .map((i) => [pos[i * 3], pos[i * 3 + 1]]).sort((a, b) => a[0] - b[0]);
  const eyes = EYES.map((e) => {
    const up = curve(e.upper, e.corners), lo = curve(e.lower, e.corners);
    return { up, lo, x0: up[0][0], x1: up[up.length - 1][0] };
  });
  const browPts = BROWS.map((i) => [pos[i * 3], pos[i * 3 + 1]]);
  w.brow = new Float32Array(N);
  const browSigma2 = 2 * (0.07 * H) ** 2;
  for (let i = 0; i < N; i++) {
    const [X, Y] = [pos[i * 3], pos[i * 3 + 1]];
    let d2 = Infinity;
    for (const [bx, by] of browPts) d2 = Math.min(d2, (X - bx) ** 2 + (Y - by) ** 2);
    // Only above the eyes: the lids themselves stay put (the blink handles them).
    const aboveEyes = eyes.some((e) => X > e.x0 && X < e.x1 && Y > lerpCurve(e.up, X) + 0.01 * H) ||
      eyes.every((e) => X <= e.x0 || X >= e.x1);
    w.brow[i] = aboveEyes && Y > C.y + 0.25 * H ? Math.exp(-d2 / browSigma2) : 0;
  }
  for (const i of FACE_OVAL) w.brow[i] = 0;

  return {
    rest: pos, uv, indices, weights: w, count: N, eyes,
    info: { C, H, Wm, zBack, aspect, cornerL: cL, cornerR: cR, closedBy, triangles: tris.length },
  };
}

/**
 * Move the vertices for one frame.
 * @param face result of buildFace
 * @param {(name:string)=>number} g blendshape value by name (0..1)
 * @param {Float32Array} out positions to write (same length as face.rest)
 * @returns {{open:number}} how far the jaw dropped (scene units), for the teeth/tongue
 */
export function deform(face, g, out, { blink = 0, brow = 0 } = {}) {
  const { rest, weights: w, count, info } = face;
  const { C, H, Wm } = info;
  const clamp = (v) => Math.min(1, Math.max(0, v));
  const open = clamp(g('jawOpen'));
  const lowerDown = clamp((g('mouthLowerDownLeft') + g('mouthLowerDownRight')) / 2);
  const upperUp = clamp((g('mouthUpperUpLeft') + g('mouthUpperUpRight')) / 2);
  const round = clamp(g('mouthFunnel') + 0.7 * g('mouthPucker'));
  const smileL = clamp(g('mouthSmileLeft')), smileR = clamp(g('mouthSmileRight'));
  const wideL = clamp(smileL + 0.6 * g('mouthStretchLeft')), wideR = clamp(smileR + 0.6 * g('mouthStretchRight'));

  const jawDrop = open * 0.13 * H;
  for (let i = 0; i < count; i++) {
    let x = rest[i * 3], y = rest[i * 3 + 1], z = rest[i * 3 + 2];
    y -= jawDrop * w.jaw[i] + lowerDown * 0.03 * H * w.lower[i];
    z -= open * 0.03 * H * w.jaw[i];
    y += upperUp * 0.025 * H * w.upper[i];
    x -= (x - C.x) * 0.35 * round * w.round[i];
    z += 0.035 * H * round * w.round[i];
    // Corners: stretch outward; smiles also lift. (Left = the person's left = picture's right.)
    x += (wideL * w.cornerL[i] - wideR * w.cornerR[i]) * 0.22 * Wm * (info.cornerL[0] > C.x ? 1 : -1);
    y += (smileL * w.cornerL[i] + smileR * w.cornerR[i]) * 0.07 * Wm;
    out[i * 3] = x; out[i * 3 + 1] = y; out[i * 3 + 2] = z;
  }

  // Brows lift (and the forehead skin with them).
  if (brow > 0) for (let i = 0; i < count; i++) out[i * 3 + 1] += brow * 0.025 * H * w.brow[i];

  // Blink: inside each eye, squeeze [lower lid … upper lid] toward the lower lid,
  // and stretch the band of skin above the upper lid to fill the space.
  if (blink > 0) {
    const b = Math.min(1, blink);
    for (const e of face.eyes) {
      for (let i = 0; i < count; i++) {
        const x = out[i * 3];
        if (x <= e.x0 || x >= e.x1) continue;
        const y = out[i * 3 + 1];
        const yl = lerpCurve(e.lo, x), yu = lerpCurve(e.up, x);
        const h = yu - yl;
        if (h <= 0) continue;
        const band = 1.4 * h, yu2 = yl + h * (1 - b);
        if (y >= yl && y <= yu) out[i * 3 + 1] = yl + (y - yl) * (1 - b);
        else if (y > yu && y < yu + band) out[i * 3 + 1] = yu2 + (y - yu) * ((yu + band - yu2) / band);
      }
    }
  }
  return { jawDrop, round };
}
