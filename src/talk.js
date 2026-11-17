// talk.js: two-way conversation. You talk, the Memoji understands, thinks, and talks back.
//
//   you speak ──► speech recognition (browser) ──► text
//        + how you said it (pauses, "umm"s, pace) from listening.js
//   ──► LLM (OpenAI or Claude, the visitor's own API key), streamed token by token
//   ──► split into sentences ──► each sentence spoken as soon as it's complete (browser voice)
//   ──► the mouth is driven from the words being spoken (letters → mouth shapes)
//
// Two things borrowed from how full-duplex systems behave:
//   • streaming: it starts speaking after the FIRST sentence, not after the whole reply,
//     and measures the delay from "you stopped" to "it started talking";
//   • barge-in: if you start talking while it speaks, it stops mid-sentence and listens.
//
// The API key is only ever sent to the provider you choose, directly from your browser.

const PROVIDERS = {
  openai: { label: 'OpenAI', model: 'gpt-4o-mini' },
  anthropic: { label: 'Claude (Anthropic)', model: 'claude-haiku-5-5' },
};
export const defaultModel = (provider) => PROVIDERS[provider]?.model ?? '';

const SYSTEM = `You are the friendly talking avatar in a real-time lip-sync demo, having a live, spoken, face-to-face conversation.
Rules for every reply:
- Speak naturally, like a person talking: 1-2 short sentences, under 40 words.
- No lists, markdown, emojis, or stage directions; it will be read aloud.
- Be warm and genuinely curious. Often end with one short, specific follow-up question about what they said.
- Each user message may end with a note in square brackets about HOW they spoke (pauses, "umm"s, pace).
  Use it gently and implicitly (for example, reassure someone who hesitated, or slow down for someone thinking aloud),
  but never mention the note or quote it.`;

// ---- speech → mouth shapes ---------------------------------------------------------------
// A rough letter → mouth mapping, good enough for a cartoon: vowels open the mouth in their
// own shape, m/b/p close the lips, f/v tuck the lip, other consonants barely open it.
function letterShape(text, i) {
  const ch = text[i]?.toLowerCase() ?? ' ';
  const next = text[i + 1]?.toLowerCase() ?? ' ';
  if (ch === 'o' && next === 'o') return { env: 0.8, open: 0.25, round: 1, spread: 0, fric: 0 };
  if (ch === 'e' && next === 'e') return { env: 0.8, open: 0.2, round: 0, spread: 1, fric: 0 };
  switch (ch) {
    case 'a': return { env: 0.95, open: 0.95, round: 0, spread: 0.1, fric: 0 };
    case 'e': return { env: 0.8, open: 0.45, round: 0, spread: 0.7, fric: 0 };
    case 'i': case 'y': return { env: 0.75, open: 0.3, round: 0, spread: 0.9, fric: 0 };
    case 'o': return { env: 0.85, open: 0.6, round: 0.8, spread: 0, fric: 0 };
    case 'u': case 'w': case 'q': return { env: 0.7, open: 0.25, round: 1, spread: 0, fric: 0 };
    case 'm': case 'b': case 'p': return { env: 0, open: 0, round: 0, spread: 0, fric: 0 };
    case 'f': case 'v': return { env: 0.35, open: 0.1, round: 0, spread: 0.1, fric: 1 };
    case ' ': case ',': case '.': case '!': case '?': case ';': case ':': case '-':
      return { env: 0.05, open: 0, round: 0, spread: 0, fric: 0 };
    default: return { env: 0.45, open: 0.25, round: 0, spread: 0.2, fric: 0 };
  }
}

// ---- streaming LLM -----------------------------------------------------------------------
async function* sse(response) {
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line.startsWith('data:')) yield line.slice(5).trim();
    }
  }
}

async function* streamReply({ provider, key, model, messages, signal }) {
  let res;
  if (provider === 'anthropic') {
    res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST', signal,
      headers: {
        'content-type': 'application/json',
        'x-api-key': key,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({ model, max_tokens: 160, system: SYSTEM, messages, stream: true }),
    });
  } else {
    res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
      body: JSON.stringify({ model, max_tokens: 160, stream: true, messages: [{ role: 'system', content: SYSTEM }, ...messages] }),
    });
  }
  if (!res.ok) {
    let msg = `${res.status}`;
    try { const j = await res.json(); msg += ': ' + (j.error?.message ?? JSON.stringify(j)); } catch {}
    throw new Error(msg);
  }
  for await (const data of sse(res)) {
    if (data === '[DONE]') return;
    let j;
    try { j = JSON.parse(data); } catch { continue; }
    const t = provider === 'anthropic'
      ? (j.type === 'content_block_delta' ? j.delta?.text : null)
      : j.choices?.[0]?.delta?.content;
    if (j.type === 'error') throw new Error(j.error?.message ?? 'stream error');
    if (t) yield t;
  }
}

// ---- the conversation --------------------------------------------------------------------
export function createTalker({ getCues, onUpdate }) {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const s = {
    active: false, state: 'off',            // off | listening | thinking | speaking
    history: [],                            // [{role, content}]
    transcript: [],                         // [{who, text}] for the page
    heard: '', interim: '', sendTimer: 0, userStartedAt: 0,
    queue: [], speakingText: '', speakStart: 0, charMs: 70, abort: null,
    lastSpoken: '', spokeEndAt: 0,          // to ignore its own voice echoing into the mic
    mouth: { env: 0, open: 0, round: 0, spread: 0, fric: 0 },
    timing: null,                           // {userEnd, firstToken, firstSound}
    error: '', voice: null, provider: 'openai', key: '', model: '',
  };
  let rec = null;
  const update = () => onUpdate?.(s);

  // --- speech recognition (stays on the whole time, so you can interrupt)
  function startRecognition() {
    rec = new Recognition();
    rec.lang = 'en-US';
    rec.continuous = true;
    rec.interimResults = true;
    rec.onresult = (e) => {
      let interim = '', final = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) final += r[0].transcript; else interim += r[0].transcript;
      }
      const words = (interim || final).trim().split(/\s+/).filter(Boolean);
      // Just finished speaking? Ignore its own words arriving late from the speakers.
      if (s.state === 'listening' && performance.now() - s.spokeEndAt < 2000 && words.length && isEcho(words, s.lastSpoken)) return;
      if (s.state === 'speaking' || s.state === 'thinking') {
        // Barge-in: real speech from the person (not the avatar's own voice leaking into the mic).
        if (words.length >= 2 && !isEcho(words)) interrupt();
        else return;
      }
      if (!s.userStartedAt && words.length) s.userStartedAt = performance.now();
      s.interim = interim;
      if (final) s.heard = (s.heard + ' ' + final).trim();
      // Send after a short silence, so a pause mid-thought doesn't cut you off.
      clearTimeout(s.sendTimer);
      if (s.heard && !interim) s.sendTimer = setTimeout(sendTurn, 900);
      update();
    };
    rec.onerror = (e) => {
      if (e.error === 'not-allowed') { s.error = 'Microphone permission was blocked for speech recognition.'; stop(); }
      else if (e.error !== 'no-speech' && e.error !== 'aborted') { s.error = 'speech recognition: ' + e.error; update(); }
    };
    rec.onend = () => { if (s.active) try { rec.start(); } catch {} };   // Chrome stops after a while; keep it going
    rec.start();
  }

  // Is the "speech" just the avatar hearing itself through the speakers?
  function isEcho(words, reference = s.speakingText) {
    if (!reference) return false;
    const spoken = new Set(reference.toLowerCase().replace(/[^a-z' ]/g, ' ').split(/\s+/));
    const hits = words.filter((w) => spoken.has(w.toLowerCase().replace(/[^a-z']/g, ''))).length;
    return hits / words.length > 0.5;
  }

  async function sendTurn() {
    const text = s.heard.trim();
    if (!text || !s.active) return;
    const cues = getCues?.(s.userStartedAt ? (performance.now() - s.userStartedAt) / 1000 : 0, text) ?? '';
    s.heard = ''; s.interim = ''; s.userStartedAt = 0;
    s.transcript.push({ who: 'you', text });
    s.history.push({ role: 'user', content: cues ? `${text}\n${cues}` : text });
    s.history = s.history.slice(-16);
    s.timing = { userEnd: performance.now() - 900, firstToken: 0, firstSound: 0 };
    s.state = 'thinking';
    s.lastSpoken = '';
    update();

    const reply = { who: 'memoji', text: '' };
    s.transcript.push(reply);
    s.abort = new AbortController();
    let pending = '';
    try {
      for await (const t of streamReply({ provider: s.provider, key: s.key, model: s.model, messages: s.history, signal: s.abort.signal })) {
        if (!s.timing.firstToken) s.timing.firstToken = performance.now();
        reply.text += t;
        pending += t;
        // Speak each sentence as soon as it's complete: don't wait for the whole reply.
        let m;
        while ((m = pending.match(/^(.*?[.!?])(\s+|$)/s)) && m[2] !== '' ) {
          enqueue(m[1]);
          pending = pending.slice(m[0].length);
        }
        update();
      }
      if (pending.trim()) enqueue(pending.trim());
      s.history.push({ role: 'assistant', content: reply.text });
    } catch (err) {
      if (err.name !== 'AbortError') {
        s.error = 'AI request failed (' + err.message + ').';
        s.state = 'listening';
      }
    }
    s.abort = null;
    update();
  }

  // --- speaking
  function enqueue(sentence) {
    s.queue.push(sentence);
    if (s.state !== 'speaking') speakNext();
  }
  function speakNext() {
    const text = s.queue.shift();
    if (!text) {
      if (s.state === 'speaking') s.spokeEndAt = performance.now();
      s.state = s.active ? 'listening' : 'off'; s.speakingText = ''; update(); return;
    }
    const u = new SpeechSynthesisUtterance(text);
    if (s.voice) u.voice = s.voice;
    u.rate = 1.03;
    u.onstart = () => {
      s.speakingText = text;
      s.lastSpoken = (s.lastSpoken + ' ' + text).slice(-400);
      s.speakStart = performance.now();
      s.charMs = 1000 / (14 * u.rate);                      // ~14 characters per second
      if (s.timing && !s.timing.firstSound) s.timing.firstSound = performance.now();
      s.state = 'speaking';
      update();
    };
    // When the voice reports word positions, re-sync the mouth to them.
    u.onboundary = (e) => { if (e.name === 'word') s.speakStart = performance.now() - e.charIndex * s.charMs; };
    u.onend = () => speakNext();
    u.onerror = () => speakNext();
    s.state = 'speaking';
    speechSynthesis.speak(u);
  }

  function interrupt() {
    s.abort?.abort();
    s.queue = [];
    speechSynthesis.cancel();
    const last = s.transcript[s.transcript.length - 1];
    if (last?.who === 'memoji' && last.text) last.text += ' …(interrupted)';
    s.speakingText = '';
    s.state = 'listening';
    update();
  }

  function start({ provider, key, model, voice }) {
    if (!Recognition) { s.error = 'This browser has no speech recognition. Please use Chrome or Edge.'; update(); return false; }
    Object.assign(s, { provider, key, model: model || defaultModel(provider), voice, active: true, state: 'listening', error: '' });
    startRecognition();
    update();
    return true;
  }
  function stop() {
    s.active = false;
    clearTimeout(s.sendTimer);
    interrupt();
    try { rec?.stop(); } catch {}
    s.state = 'off';
    update();
  }

  /** Mouth shape for this moment while the avatar is speaking, else null (the mic drives the mouth). */
  function mouth(now) {
    if (s.state !== 'speaking' || !s.speakingText) return null;
    const i = Math.floor((now - s.speakStart) / s.charMs);
    const target = i < s.speakingText.length ? letterShape(s.speakingText, i) : letterShape(' ', 0);
    for (const k of Object.keys(s.mouth)) s.mouth[k] += (target[k] - s.mouth[k]) * 0.35;
    return s.mouth;
  }

  return { start, stop, interrupt, mouth, state: s, get speaking() { return s.state === 'speaking'; } };
}
