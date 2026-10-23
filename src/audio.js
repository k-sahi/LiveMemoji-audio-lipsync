// audio.js — microphone capture via the Web Audio API.

// Turn OFF the browser's voice processing. Echo cancellation, noise
// suppression and auto-gain change the signal in unpredictable ways.
// The features we train on must match the features we see at inference.
const MIC_CONSTRAINTS = {
  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
};

// Microphones that usually add 150–400 ms of delay (bad for lipsync).
const LAGGY_MIC = /airpods|bluetooth|hands-?free|headset|buds|beats/i;

/**
 * Steps 1–2: simple capture through an AnalyserNode (the browser does the FFT).
 *   mic stream → MediaStreamSource → AnalyserNode
 */
export async function startMic({ fftSize = 1024 } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);

  const analyser = ctx.createAnalyser();
  analyser.fftSize = fftSize;           // samples per analysis window (power of 2)
  analyser.smoothingTimeConstant = 0;   // no averaging between frames — we want raw data

  // Note: we do NOT connect to ctx.destination, or you'd hear yourself (feedback).
  source.connect(analyser);
  return { ctx, analyser, stream, sampleRate: ctx.sampleRate };
}

/**
 * Step 4+: frame-accurate capture on the audio thread.
 *   mic → MediaStreamSource → AudioWorklet 'frame-tap' → (silent) → speakers
 * Calls onFrame(samples, endFrame) exactly `fps` times per second of audio.
 */
export async function startMicFrames({ fps = 30, windowSize = 1024, onFrame }) {
  const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
  const ctx = new AudioContext();
  await ctx.resume();
  await ctx.audioWorklet.addModule(new URL('./audio-worklet.js', import.meta.url));

  const hop = Math.round(ctx.sampleRate / fps); // 1600 samples at 48 kHz
  const source = ctx.createMediaStreamSource(stream);
  const tap = new AudioWorkletNode(ctx, 'frame-tap', { processorOptions: { windowSize, hop } });
  // Browsers only run nodes that lead to the speakers, so route through a
  // muted gain node: the worklet runs, but you don't hear yourself.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(tap).connect(mute).connect(ctx.destination);
  tap.port.onmessage = (e) => onFrame(e.data.samples, e.data.endFrame);

  const micLabel = stream.getAudioTracks()[0]?.label || 'unknown microphone';

  /** Convert an audio sample index to performance.now() time (ms). */
  function audioFrameToPerfTime(endFrame) {
    const ts = ctx.getOutputTimestamp?.();
    if (!ts || !ts.performanceTime) return performance.now();
    return ts.performanceTime + (endFrame / ctx.sampleRate - ts.contextTime) * 1000;
  }

  return {
    ctx, stream, sampleRate: ctx.sampleRate, hop,
    fps: ctx.sampleRate / hop,
    micLabel,
    micLooksLaggy: LAGGY_MIC.test(micLabel),
    audioFrameToPerfTime,
  };
}
