// listening.js: the avatar notices HOW you talk, not just the sounds, and reacts like a listener.
//
//   hesitation pause   you stop for a moment mid-thought  → head tilts, brows lift, eyes drift up:
//                                                           "I'm with you, take your time"
//   "umm" / "uhh"      a steady, neutral, held sound      → the same patient, thinking look
//   you carry on       speech resumes after a hesitation  → a small encouraging nod
//   end of your turn   a longer silence after you spoke   → a double nod and a smile: "got it"
//
// All of this comes from the same 30-per-second audio features the mouth uses (loudness from
// voice-envelope.js, vowel shape from formants.js), so it adds no latency and needs no model.
// It's a hand-built, rule-based version of the "reads hesitation and responds in real time"
// behaviour that a learned conversational model would do end to end.

const TIMING = {
  pauseMs: 350,        // a gap this long mid-speech counts as a hesitation pause
  endOfTurnMs: 1300,   // a gap this long means you've finished your turn
  minTurnMs: 700,      // you must have spoken this long before an end-of-turn nod
  fillerMs: 320,       // a steady neutral sound held this long = "umm" / "uhh"
  forgetMs: 4500,      // after this long quiet, go back to idle
};

export function createListener() {
  const s = {
    phase: 'idle',        // idle | speaking | pause | ended
    kind: '',             // what's being reacted to right now (for the on-screen label)
    speechStart: 0, lastLoud: 0, lastT: 0,
    hes: 0, hesTarget: 0, tilt: 1,
    smile: 0, smileUntil: 0,
    nods: [],
    // "umm" detector: a running average of the vowel shape and how much it moves
    avg: null, wobble: 1, steadyMs: 0, filler: false,
  };

  const nod = (now, count = 1, amp = 0.08) => {
    for (let i = 0; i < count; i++) s.nods.push({ start: now + i * 360, amp: amp * (i ? 0.7 : 1) });
  };
  const counts = { umm: 0, pause: 0 };   // per turn, for the conversation mode
  const startHesitation = (kind) => {
    counts[kind] = (counts[kind] ?? 0) + 1;
    if (s.hesTarget === 0) s.tilt = Math.random() < 0.5 ? -1 : 1;   // tilt to a random side each time
    s.hesTarget = 1;
    s.kind = kind;
  };

  return {
    /**
     * Call once per audio frame.
     * @param now ms timestamp
     * @param loud is the person clearly making sound right now
     * @param shape this frame's vowel shape {open, round, spread} or null when unvoiced
     */
    hear({ now, loud, shape }) {
      const dt = s.lastT ? Math.min(100, now - s.lastT) : 33;
      s.lastT = now;

      // ---- "umm" / "uhh": voiced, held steady, and neutral (not a clear "ah", "ee" or "oo")
      if (loud && shape) {
        const v = [shape.open, shape.round, shape.spread];
        if (!s.avg) s.avg = v.slice();
        const dev = Math.abs(v[0] - s.avg[0]) + Math.abs(v[1] - s.avg[1]) + Math.abs(v[2] - s.avg[2]);
        for (let i = 0; i < 3; i++) s.avg[i] += (v[i] - s.avg[i]) * 0.3;
        s.wobble += (dev - s.wobble) * 0.35;
        const neutral = s.avg[0] < 0.62 && s.avg[1] < 0.45 && s.avg[2] < 0.25;
        s.steadyMs = s.wobble < 0.15 && neutral ? s.steadyMs + dt : Math.max(0, s.steadyMs - 2 * dt);
      } else {
        s.steadyMs = Math.max(0, s.steadyMs - 2 * dt);
        if (!loud) s.avg = null;
      }
      if (!s.filler && s.steadyMs > TIMING.fillerMs) { s.filler = true; startHesitation('umm'); }
      else if (s.filler && s.steadyMs < 120) {
        s.filler = false;
        if (loud) { s.hesTarget = 0; s.kind = 'listening'; nod(now, 1, 0.05); }
      }

      // ---- pauses and turns
      if (loud) {
        if (s.phase === 'idle' || s.phase === 'ended') { s.phase = 'speaking'; s.speechStart = now; s.kind = 'listening'; }
        else if (s.phase === 'pause') {
          // you carried on after a hesitation: encourage you with a little nod
          s.phase = 'speaking';
          if (!s.filler) { s.hesTarget = 0; s.kind = 'listening'; nod(now, 1, 0.05); }
        }
        s.lastLoud = now;
      } else {
        const gap = now - s.lastLoud, spoke = s.lastLoud - s.speechStart;
        if (s.phase === 'speaking' && gap > TIMING.pauseMs && spoke > 400) {
          s.phase = 'pause';
          startHesitation('pause');
        } else if ((s.phase === 'pause' || s.phase === 'speaking') && gap > TIMING.endOfTurnMs) {
          s.phase = 'ended';
          s.hesTarget = 0; s.filler = false; s.steadyMs = 0;
          if (spoke > TIMING.minTurnMs) { nod(now, 2); s.smileUntil = now + 1400; s.kind = 'got it'; }
          else s.kind = '';
        } else if (s.phase === 'ended' && gap > TIMING.forgetMs) {
          s.phase = 'idle'; s.kind = '';
        }
      }
    },

    /** How many "umm"s and hesitation pauses since the last call (then starts counting again). */
    takeCounts() {
      const out = { ...counts };
      counts.umm = 0; counts.pause = 0;
      return out;
    },

    /** The listener's body language right now, to add on top of the normal animation. */
    pose(now) {
      const dt = s.poseT ? Math.min(100, now - s.poseT) : 16;
      s.poseT = now;
      s.hes += (s.hesTarget - s.hes) * (1 - Math.exp(-dt / 280));
      s.smile += ((now < s.smileUntil ? 1 : 0) - s.smile) * (1 - Math.exp(-dt / 200));
      let nodPitch = 0;
      s.nods = s.nods.filter((n) => now < n.start + 420);
      for (const n of s.nods) {
        const u = (now - n.start) / 420;
        if (u > 0) nodPitch += n.amp * Math.sin(Math.PI * u);   // head goes down and back up
      }
      const h = s.hes;
      return {
        roll: 0.1 * h * s.tilt,
        yaw: 0.05 * h * s.tilt,
        pitch: -0.025 * h + nodPitch,
        brow: 0.6 * h,
        smile: 0.12 * h + 0.45 * s.smile,
        gaze: h > 0.25 ? { x: 0.55 * s.tilt, y: -0.55 } : null,   // eyes drift up and to the side, "thinking with you"
        label: labelFor(s.kind, h),
      };
    },
  };
}

function labelFor(kind, h) {
  if (kind === 'umm' && h > 0.2) return '🤔 heard "umm…", waiting patiently';
  if (kind === 'pause' && h > 0.2) return '🤔 you paused, listening';
  if (kind === 'got it') return '👍 got it';
  if (kind === 'listening') return '👂 listening';
  return '';
}
