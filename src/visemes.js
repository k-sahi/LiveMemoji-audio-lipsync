// visemes.js — capture how a REAL mouth moves, and transfer that motion to any face.
//
// A "viseme" is the visible mouth shape of a sound: "ah" (open), "ee" (spread),
// "oo" (rounded), "mm" (lips pressed), "ff" (lower lip on teeth), plus a smile.
//
// CAPTURE (capture.html): for each shape we average MediaPipe's 478 landmarks
// over a second and subtract the neutral face. Head movement is removed first:
// every frame is aligned to the neutral face using points that don't move when
// you talk (eye corners, nose bridge, forehead, temples). The result, per shape,
// is how each of the 478 points moves, measured in "eye-distance units" along
// the face's own left-right and up-down axes. That makes it independent of how
// big the face is, where it is, and how it's tilted.
//
// RETARGET (avatar): measure the same axes and eye distance on any picture's
// landmarks, and convert each motion into that picture's coordinates.

export const SHAPES = ['ah', 'ee', 'oo', 'mm', 'ff', 'smile'];

// Points that stay put while you talk.
export const STABLE = [33, 133, 362, 263, 6, 168, 197, 195, 5, 4, 10, 151, 9, 234, 454, 127, 356, 70, 300];
const EYE_R = [33, 133], EYE_L = [362, 263];

/** Eye-line frame of a face: origin between the eyes, x = toward the person's left eye, unit = eye distance. */
export function faceFrame(P) { // P(i) -> [x, y, z]
  const mid = (a, b) => [(P(a)[0] + P(b)[0]) / 2, (P(a)[1] + P(b)[1]) / 2, (P(a)[2] + P(b)[2]) / 2];
  const r = mid(...EYE_R), l = mid(...EYE_L);
  const dx = l[0] - r[0], dy = l[1] - r[1];
  const iod = Math.hypot(dx, dy);
  return { origin: [(r[0] + l[0]) / 2, (r[1] + l[1]) / 2], ux: [dx / iod, dy / iod], iod };
}

/**
 * 2D similarity alignment (rotation + uniform scale + translation) that best maps
 * `src` points onto `dst` points (least squares). Returns a function for any point.
 */
export function similarity(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let a = 0, b = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - sx, y = src[i][1] - sy, u = dst[i][0] - dx, v = dst[i][1] - dy;
    a += x * u + y * v; b += x * v - y * u; ss += x * x + y * y;
  }
  a /= ss; b /= ss; // [a -b; b a] = scale * rotation
  const s = Math.hypot(a, b);
  let sz = 0, dz = 0;
  for (let i = 0; i < n; i++) { sz += src[i][2] ?? 0; dz += dst[i][2] ?? 0; }
  sz /= n; dz /= n;
  return ([x, y, z = 0]) => [a * (x - sx) - b * (y - sy) + dx, b * (x - sx) + a * (y - sy) + dy, s * (z - sz) + dz];
}

/** Landmarks (normalised 0..1) → pixel-space points, so x and y use the same units. */
export const toPixels = (lm, W, H) => lm.map((p) => [p.x * W, p.y * H, p.z * W]);

/** Align one frame's points to the neutral face using only the stable points. */
export function alignToNeutral(pts, neutral) {
  const f = similarity(STABLE.map((i) => pts[i]), STABLE.map((i) => neutral[i]));
  return pts.map(f);
}

/** Average a list of point arrays. */
export function meanPoints(frames) {
  const n = frames[0].length, out = Array.from({ length: n }, () => [0, 0, 0]);
  for (const f of frames) for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) out[i][k] += f[i][k] / frames.length;
  return out;
}

/** shape (aligned, averaged) − neutral, expressed in the neutral face's eye-line frame. Image coords: y down. */
export function toCanonicalDelta(shape, neutral) {
  const fr = faceFrame((i) => neutral[i]);
  const [ux, uy] = fr.ux;
  const out = new Float32Array(neutral.length * 3);
  for (let i = 0; i < neutral.length; i++) {
    const dx = shape[i][0] - neutral[i][0], dy = shape[i][1] - neutral[i][1], dz = shape[i][2] - neutral[i][2];
    out[i * 3] = (dx * ux + dy * uy) / fr.iod;      // along the eye line
    out[i * 3 + 1] = (-dx * uy + dy * ux) / fr.iod; // perpendicular, pointing DOWN the face
    out[i * 3 + 2] = dz / fr.iod;                   // MediaPipe z: negative = toward the camera
  }
  return out;
}

/**
 * Convert canonical deltas to a picture's scene coordinates (y up, +z toward viewer).
 * @param rest Float32Array of the picture's rest positions (scene units, y up)
 * @param count number of points to convert
 */
export function retarget(canon, rest, count) {
  const fr = faceFrame((i) => [rest[i * 3], rest[i * 3 + 1], rest[i * 3 + 2]]);
  const [ux, uy] = fr.ux;          // eye line in scene coords (y up)
  const down = [uy, -ux];          // "down the face" in y-up coords
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const a = canon[i * 3], b = canon[i * 3 + 1], c = canon[i * 3 + 2];
    out[i * 3] = (a * ux + b * down[0]) * fr.iod;
    out[i * 3 + 1] = (a * uy + b * down[1]) * fr.iod;
    out[i * 3 + 2] = -c * fr.iod;
  }
  return out;
}

/** Check a capture looks sensible: each shape should mainly move the mouth, and noticeably. */
export function describeCapture(deltas) {
  const size = (d, idx) => Math.max(...idx.map((i) => Math.hypot(d[i * 3], d[i * 3 + 1])));
  const MOUTH = [13, 14, 61, 291, 0, 17], STILL = [33, 263, 6, 10];
  return Object.fromEntries(Object.entries(deltas).map(([k, d]) => [k, { mouth: size(d, MOUTH), still: size(d, STILL) }]));
}
