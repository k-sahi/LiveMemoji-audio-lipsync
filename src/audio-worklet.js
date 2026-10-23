// audio-worklet.js — runs on the browser's dedicated AUDIO thread.
//
// The main thread is busy (face tracking, drawing), so timers there drift.
// The audio thread is never late: it receives the mic in 128-sample blocks,
// on time, every time. Here we keep the last `windowSize` samples in a ring
// buffer and, every `hop` samples (sampleRate / 30 → exactly 30 frames/sec),
// post a copy of that window to the main thread together with its exact
// sample position. The main thread can process it late; the timing stays exact.

class FrameTap extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { windowSize, hop } = options.processorOptions;
    this.win = windowSize;
    this.hop = hop;
    this.ring = new Float32Array(windowSize);
    this.pos = 0;          // next write index in the ring
    this.total = 0;        // samples seen so far
    this.sinceLast = 0;    // samples since the last posted frame
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.ring[this.pos] = ch[i];
      this.pos = (this.pos + 1) % this.win;
      this.total++;
      this.sinceLast++;
      if (this.sinceLast >= this.hop && this.total >= this.win) {
        this.sinceLast = 0;
        // Unroll the ring so out[0] is the oldest sample.
        const out = new Float32Array(this.win);
        for (let j = 0; j < this.win; j++) out[j] = this.ring[(this.pos + j) % this.win];
        // `currentFrame` = sample index at the start of this 128-sample block.
        this.port.postMessage({ samples: out, endFrame: currentFrame + i + 1 }, [out.buffer]);
      }
    }
    return true;
  }
}

registerProcessor('frame-tap', FrameTap);
