<p align="center">
  <img src="avatar.png" alt="My Memoji avatar" width="260">
</p>

<h1 align="center">Talking Memoji</h1>

<p align="center">
  <b>Audio-driven 3D lipsync in the browser.</b> I talk, and my Memoji lip-syncs with me.<br>
  A neural network I trained on my own voice turns sound into mouth movements, live.
</p>

<p align="center">
  <a href="https://k-sahi.github.io/LiveMemoji-audio-lipsync/"><b>▶ Live demo</b></a>
</p>

---

## How it works

```
 mic ─► 13 MFCCs every 33 ms ─► neural network ─► 27 mouth shapes ─► 3D Memoji face
          (sound features)       (trained on me)    (jaw, lips, smile…)  (mesh + blinks + waving hand)
```

1. **Hearing.** The browser's audio thread cuts the microphone into frames at exactly 30 per second.
   Each frame goes through an FFT and a mel filterbank into **13 MFCCs**, numbers that describe the shape of
   the sound (where your mouth puts the resonances) and mostly ignore pitch. Both written from scratch.
2. **Learning.** To get training data, I recorded myself talking while the webcam watched my face.
   **MediaPipe** measured 52 facial blendshapes (jaw open, lips rounded, smile…) for every audio frame,
   so the camera labelled the audio automatically. I recorded about 3 minutes.
3. **Predicting.** A small network (117 → 128 → 64 → 27) looks at 300 ms of sound (9 frames) and predicts
   the 27 jaw and mouth blendshapes. It trains in the browser with **TensorFlow.js** in under a minute.
   On speech it never saw, it scores **0.59** (R², where 0 = guessing the average and 1 = perfect).
4. **Animating.** MediaPipe finds 478 3D points on my Memoji picture. I connect them into a mesh
   (Delaunay triangulation) with a hole for the mouth, close the picture's smiling lips into a resting pose,
   and move the vertices in **Three.js**. The mouth shape comes from the voice's **formants** (the resonances
   of your mouth, found with LPC): a high F1 opens it ("ah"), a low F2 rounds it ("oo"), a high F2 spreads it ("ee").
   How far it opens follows your loudness against your own mic's background level, so it works on any
   microphone, and the network adds detail on top.
   **Real mouth motion:** on the Mouth capture page I recorded my own mouth making 7 shapes; the webcam
   measured how all 478 face points move for each (with head movement removed). Those motions are scaled to
   any picture's eye distance and tilt and blended by the voice, so any face moves like a real mouth.
   The inside of the mouth is clipped to the lips with a stencil mask. On top: blinking, eyebrows that lift
   on emphasis, head nods with speech, and a hand that waves (found with MediaPipe's hand tracker, cut out
   with a colour-keyed flood fill, and painted out of the background).

Everything runs locally in the browser. No audio is uploaded anywhere.

## Pages

| Page | What it does |
|---|---|
| [`index.html`](index.html) | The talking 3D Memoji |
| [`audio.html`](audio.html) | Live waveform, spectrum, mel filterbank and MFCC heatmaps |
| [`capture.html`](capture.html) | Captures how a real mouth moves (7 shapes, head motion removed) so any face can move the same way |
| [`face.html`](face.html) | Live face tracking with all 52 blendshapes, plus an avatar-picture checker |
| [`record.html`](record.html) | Records training data: voice features paired with face blendshapes |
| [`train.html`](train.html) | Trains the network, scores it, and tests it live with no camera |

## Code

| File | |
|---|---|
| `src/fft.js`, `src/mfcc.js`, `src/features.js` | FFT, mel filterbank and DCT from scratch: sound → 13 MFCCs |
| `src/audio.js`, `src/audio-worklet.js` | Mic capture on the audio thread at exactly 30 frames/s |
| `src/facemodel.js` | MediaPipe face and hand landmarker loaders |
| `src/record.js` | Pairs each audio frame with the face measured closest in time, estimates mic/camera delay |
| `src/dataset.js` | 9-frame windows, delay correction, validation split with no leakage between train and test |
| `src/train.js`, `src/predictor.js` | TensorFlow.js training (best-epoch selection) and streaming prediction |
| `src/avatar-geometry.js` | Landmarks → 3D mesh, mouth hole, lip closing, jaw/lip/smile deformation, blinks, brows |
| `src/avatar-hand.js` | Hand cut-out (flood fill) and background inpainting |
| `src/visemes.js`, `src/capture.js` | Real-mouth motion capture: similarity alignment, eye-distance normalisation, retargeting to any face |
| `src/formants.js` | LPC formant tracking (F1/F2) → open / round / spread mouth shapes, on any mic |
| `src/voice-envelope.js` | Jaw opening from loudness, relative to each mic's own noise floor and peaks |
| `src/avatar.js` | Puts it all together in Three.js |
| `src/main.js`, `src/face.js` | Page scripts for `audio.html` (live audio visualisations) and `face.html` (face tracking) |

## Things I learned the hard way

- **Timing matters.** My first recorder sampled audio with a timer on the main thread, competing with face
  tracking, so frames arrived 18–91 ms apart. Moving capture to an AudioWorklet made them exactly 33 ms.
- **Validate on unseen time, not random frames.** Neighbouring frames are near-copies, so a random split
  lets a model look great by memorising. I hold out the last 15% of every take instead.
- **The mic is part of the model.** A model trained on my laptop mic scored ~0 on another mic. Mean and
  variance normalisation didn't fix it (I tested both). More mics in the training data is the real fix.
- **Test with someone else.** The first person to try the live demo saw the mouth not move at all. On their mic
  the network's jaw predictions were tiny (trained on ~3 minutes of data, it plays safe and predicts values close
  to the average). Now the jaw follows loudness, which works on any mic, and the network adds the mouth shapes.
  The page also shows a mic meter and says when it can't hear you.

## Limitations

- The mouth shapes (rounding, smiles) are most accurate on the mic the network was trained on (my laptop's).
  On other mics the jaw still follows your voice, but the finer shapes are rougher.
- One front-facing picture contains only the front of the face, so only the face turns; the hand waves as a flat
  cut-out and its fingers don't bend.

## Run it locally

Browsers only allow the mic and camera on `https://` or `localhost`, so serve the folder:

```bash
python -m http.server 8000
# open http://localhost:8000
```

Built with MediaPipe Tasks Vision, TensorFlow.js and Three.js.
