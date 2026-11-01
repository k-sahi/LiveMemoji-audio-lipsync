// voice-envelope.js — how open should the jaw be, from loudness alone?
//
// Works on any microphone and at any volume, because it never uses absolute
// levels. It keeps the last 10 seconds of loudness (in dB) and measures each
// new frame against THIS mic's own range:
//   floor = 15th percentile (background noise and pauses; low enough that
//           continuous talking with few pauses still counts as talking)
//   peak  = 97th percentile (loud syllables)
// Openness = 0 until a frame is 6 dB above the floor, 1 at the peak.
// Opens fast (syllables are quick), closes a little slower.
//
// Tested on my recordings from two different mics, and at 10x lower volume:
// the mouth is open while speaking and closed in pauses in every case.

export function createVoiceEnvelope({ startDb = 6, minRangeDb = 12, low = 0.15, high = 0.97, windowFrames = 300 } = {}) {
  const history = [];
  let env = 0, db = -120, floor = -120;
  return {
    /** @param {number} rms one frame's RMS amplitude; returns openness 0..1 */
    update(rms) {
      db = 20 * Math.log10(rms + 1e-9);
      history.push(db);
      if (history.length > windowFrames) history.shift();
      const sorted = [...history].sort((a, b) => a - b);
      floor = sorted[Math.floor(low * (sorted.length - 1))];
      const peak = sorted[Math.floor(high * (sorted.length - 1))];
      const range = Math.max(minRangeDb, peak - floor - startDb);
      const target = Math.min(1, Math.max(0, (db - floor - startDb) / range));
      env += (target - env) * (target > env ? 0.6 : 0.25);
      return env;
    },
    get value() { return env; },
    /** dB above the background level, for a level meter */
    get aboveFloor() { return Math.max(0, db - floor); },
  };
}
