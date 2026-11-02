// toon-analyze.js — read a photo's face and turn it into cartoon "traits".
//
// Input: the picture's pixels + MediaPipe's 478 landmarks.
// Output: colours (skin, hair, eyes, lips, brows, shirt), proportions (face width,
// jaw, eye size and spacing, nose, mouth), hair (volume, length, curliness) and
// accessories (glasses, mustache). The cartoon head (toon-head.js) is built from these.
//
// Everything is measured in "eye-distance" units along the face's own axes, so it
// doesn't matter how big the face is in the photo or how it's tilted.
// Each guess can be overridden in the page (hair style, glasses), because pixel
// heuristics are never perfect.

import { faceFrame } from './visemes.js';

const LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];

const lum = ([r, g, b]) => 0.299 * r + 0.587 * g + 0.114 * b;
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const median = (arr) => {
  if (!arr.length) return null;
  const s = [0, 1, 2].map((c) => arr.map((p) => p[c]).sort((x, y) => x - y)[arr.length >> 1]);
  return s;
};

export function analyzeFace(img, landmarks) {
  const { data, width: W, height: H } = img;
  const px = (x, y) => {
    const xi = Math.max(0, Math.min(W - 1, Math.round(x))), yi = Math.max(0, Math.min(H - 1, Math.round(y)));
    const k = (yi * W + xi) * 4;
    return [data[k], data[k + 1], data[k + 2]];
  };
  const L = landmarks.map((p) => [p.x * W, p.y * H, p.z * W]);
  const fr = faceFrame((i) => L[i]);
  const iod = fr.iod;
  const [ux, uy] = fr.ux;           // along the eye line (image coords)
  const dn = [-uy, ux];             // down the face (image coords, y down)
  // image point → face coords (eye-distance units; x toward the person's left eye, y down the face)
  const toFace = ([x, y]) => {
    const dx = x - fr.origin[0], dy = y - fr.origin[1];
    return [(dx * ux + dy * uy) / iod, (dx * dn[0] + dy * dn[1]) / iod];
  };
  // face coords → image point
  const toImg = (fx, fy) => [fr.origin[0] + (fx * ux + fy * dn[0]) * iod, fr.origin[1] + (fx * uy + fy * dn[1]) * iod];
  const F = L.map(toFace);
  const patch = (fx, fy, r = 0.06) => {
    const out = [];
    for (let a = -r; a <= r; a += r / 3) for (let b = -r; b <= r; b += r / 3) out.push(px(...toImg(fx + a, fy + b)));
    return out;
  };
  const region = (x0, x1, y0, y1, step = 0.03) => {
    const out = [];
    for (let fx = x0; fx <= x1; fx += step) for (let fy = y0; fy <= y1; fy += step) out.push(px(...toImg(fx, fy)));
    return out;
  };

  // ---- colours
  const skinSamples = [205, 425, 151, 6, 50, 280].flatMap((i) => patch(F[i][0], F[i][1], 0.07));
  const skin = median(skinSamples);
  const skinLum = lum(skin);
  // Background: the picture's outer border.
  const border = [];
  for (let t = 0; t <= 1; t += 0.05) border.push(px(t * (W - 1), 2), px(t * (W - 1), H - 3), px(2, t * (H - 1)), px(W - 3, t * (H - 1)));
  const background = median(border);
  const lipSamples = [[0, 13], [17, 14], [37, 82], [267, 312], [84, 87], [314, 317]].flatMap(([o, i]) =>
    patch((F[o][0] + F[i][0]) / 2, (F[o][1] + F[i][1]) / 2, 0.03));
  const lips = median(lipSamples);
  const irisR = L.length > 472 ? Math.hypot(L[469][0] - L[471][0], L[469][1] - L[471][1]) / 2 / iod : 0.18;
  const iris = L.length > 477
    ? median([468, 473].flatMap((i) => patch(F[i][0], F[i][1], irisR * 0.45)))
    : [70, 45, 30];
  const browSamples = [105, 66, 107, 334, 296, 336].flatMap((i) => patch(F[i][0], F[i][1], 0.04));
  const darkest = (arr, frac = 0.35) => median([...arr].sort((a, b) => lum(a) - lum(b)).slice(0, Math.max(1, Math.floor(arr.length * frac))));
  const brows = darkest(browSamples);

  // ---- hair: pixels above the forehead and beside the temples that are neither skin nor background
  const isHair = (c) => dist(c, skin) > 55 && dist(c, background) > 45;
  const top = F[10][1];                         // forehead top (y, face coords; negative = above the eyes)
  const faceH = F[152][1] - F[10][1];           // forehead-to-chin, in eye-distances
  const halfW = (F[454][0] - F[234][0]) / 2;    // half face width
  const above = region(-halfW * 0.7, halfW * 0.7, top - faceH * 0.3, top - faceH * 0.04);
  const hairAbove = above.filter(isHair);
  const hairFracTop = hairAbove.length / above.length;
  const hairCol = hairFracTop > 0.15 ? darkest(hairAbove, 0.6) : brows;
  // How high the hair goes: scan up from the forehead until we hit background.
  const scan = (fx, fy, dx, dy, max) => {
    let run = 0, last = 0;
    for (let s = 0; s < max; s += 0.03) {
      const c = px(...toImg(fx + dx * s, fy + dy * s));
      if (dist(c, background) < 40) { if (++run >= 3) break; } else { run = 0; last = s; }
    }
    return last;
  };
  const volumeTop = hairFracTop > 0.15 ? scan(0, top, 0, -1, faceH * 1.2) / faceH : 0;
  const volumeSide = hairFracTop > 0.15
    ? (scan(F[234][0], F[234][1], -1, 0, faceH) + scan(F[454][0], F[454][1], 1, 0, faceH)) / 2 / faceH
    : 0;
  // Long hair: hair beside the jaw, below the chin line.
  const chinY = F[152][1];
  const besideJaw = [...region(F[234][0] - halfW * 0.45, F[234][0] - halfW * 0.08, chinY - faceH * 0.1, chinY + faceH * 0.25),
    ...region(F[454][0] + halfW * 0.08, F[454][0] + halfW * 0.45, chinY - faceH * 0.1, chinY + faceH * 0.25)];
  // (must look like THIS person's hair colour, so a dark shirt or chair doesn't count)
  const longFrac = hairFracTop > 0.15 ? besideJaw.filter((c) => isHair(c) && dist(c, hairCol) < 60).length / Math.max(1, besideJaw.length) : 0;
  // Curliness: how much the brightness flickers inside the hair (curls = lots of small highlights).
  let flicker = 0, nFlick = 0;
  for (const row of [0.06, 0.1, 0.14, 0.18, 0.22]) {
    for (let fx = -halfW * 0.6; fx < halfW * 0.6; fx += 0.02) {
      const fy = top - faceH * row;
      const a = px(...toImg(fx, fy)), b = px(...toImg(fx + 0.02, fy));
      if (isHair(a) && isHair(b)) { flicker += Math.abs(lum(a) - lum(b)); nFlick++; }
    }
  }
  const curliness = nFlick > 5 ? flicker / nFlick / Math.max(20, lum(hairCol) + 20) : 0;
  let hairStyle = 'short';
  if (hairFracTop < 0.15) hairStyle = 'none';
  else if (longFrac > 0.35) hairStyle = 'long';
  else if (volumeTop > 0.28 || volumeSide > 0.22 || curliness > 0.35) hairStyle = 'curly';

  // ---- glasses: dark frame pixels just under each eye and over the nose bridge
  const darkFrac = (arr, f = 0.55) => arr.filter((c) => lum(c) < skinLum * f && dist(c, skin) > 50).length / Math.max(1, arr.length);
  const underEye = (lower, inner, outer) => region(Math.min(F[inner][0], F[outer][0]), Math.max(F[inner][0], F[outer][0]),
    F[lower][1] + 0.08, F[lower][1] + 0.38, 0.025);
  const below = (darkFrac(underEye(145, 133, 33)) + darkFrac(underEye(374, 362, 263))) / 2;
  const bridge = darkFrac(region(F[168][0] - 0.1, F[168][0] + 0.1, F[168][1] - 0.1, F[168][1] + 0.1, 0.02));
  const glasses = (below > 0.07 && bridge > 0.05) || below > 0.16;

  // ---- facial hair. Easy to fake: the shadow under the nose, a hand near the chin, a dim room.
  // So it must be clearly darker than the cheeks, look like this person's hair or brows,
  // and appear on BOTH sides of the face (a shadow or a hand is usually one-sided).
  const cheekLum = median([...patch(F[205][0], F[205][1], 0.07), ...patch(F[425][0], F[425][1], 0.07)].map((c) => [lum(c), 0, 0]))[0];
  const hairy = (c) => lum(c) < cheekLum * 0.55 && (dist(c, hairCol) < 70 || dist(c, brows) < 60) && dist(c, skin) > 60;
  const frac = (arr, fn) => arr.filter(fn).length / Math.max(1, arr.length);
  const halves = (x0, x1, y0, y1) => [frac(region(x0, 0, y0, y1, 0.025), hairy), frac(region(0, x1, y0, y1, 0.025), hairy)];
  const [mL, mR] = halves(F[61][0] * 0.6, F[291][0] * 0.6, F[2][1] + 0.05, F[0][1] - 0.04);
  const mustache = Math.min(mL, mR) > 0.45;
  const [bL, bR] = halves(F[61][0] * 0.9, F[291][0] * 0.9, F[17][1] + 0.08, F[152][1] - 0.05);
  const beard = Math.min(bL, bR) > 0.45;
  // Stubble: the lower face is evenly a bit darker than the cheeks (both sides).
  const lowL = median(region(F[61][0] * 0.9, F[61][0] * 0.3, F[17][1] + 0.08, F[152][1] - 0.08).map((c) => [lum(c), 0, 0]))[0];
  const lowR = median(region(F[291][0] * 0.3, F[291][0] * 0.9, F[17][1] + 0.08, F[152][1] - 0.08).map((c) => [lum(c), 0, 0]))[0];
  const stubble = !beard && Math.max(lowL, lowR) < cheekLum * 0.72 && Math.abs(lowL - lowR) < cheekLum * 0.12;

  // ---- clothes: under the chin
  const shirtArea = region(-halfW * 0.6, halfW * 0.6, chinY + faceH * 0.2, chinY + faceH * 0.4, 0.05)
    .filter((c) => dist(c, skin) > 40);
  const shirt = shirtArea.length > 8 ? median(shirtArea) : [90, 110, 160];

  // ---- proportions (eye-distance units)
  const eyeWidth = (Math.hypot(F[33][0] - F[133][0], F[33][1] - F[133][1]) + Math.hypot(F[362][0] - F[263][0], F[362][1] - F[263][1])) / 2;
  const eyeOpen = (Math.abs(F[159][1] - F[145][1]) + Math.abs(F[386][1] - F[374][1])) / 2;
  const jawW = Math.hypot(F[172][0] - F[397][0], F[172][1] - F[397][1]);
  const cheekW = Math.hypot(F[234][0] - F[454][0], F[234][1] - F[454][1]);

  // Mouth outline relative to the mouth centre (for the cartoon's lip shape), in face coords.
  const mc = LIPS_INNER.reduce((a, i) => [a[0] + F[i][0] / 20, a[1] + F[i][1] / 20], [0, 0]);
  const rel = (i) => [F[i][0] - mc[0], F[i][1] - mc[1]];

  return {
    colors: { skin, hair: hairCol, iris, lips, brows, shirt, background },
    face: {
      height: faceH, cheekWidth: cheekW, jawRatio: jawW / cheekW,
      eyeWidth, eyeOpen, irisRadius: irisR,
      browY: (F[105][1] + F[334][1]) / 2, noseY: F[4][1], noseWidth: Math.hypot(F[98][0] - F[327][0], F[98][1] - F[327][1]),
      mouthY: mc[1], mouthWidth: Math.hypot(F[61][0] - F[291][0], F[61][1] - F[291][1]),
      chinY, foreheadY: top,
    },
    mouth: { inner: LIPS_INNER.map(rel), outer: LIPS_OUTER.map(rel), innerIdx: LIPS_INNER, outerIdx: LIPS_OUTER },
    hair: { style: hairStyle, volumeTop, volumeSide, curliness, longFrac, coverage: hairFracTop },
    accessories: { glasses, mustache, beard, stubble, glassesScore: { below, bridge }, facialScore: { mL, mR, bL, bR, lowL, lowR, cheekLum } },
  };
}
