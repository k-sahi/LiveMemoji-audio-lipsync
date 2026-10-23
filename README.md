# Audio-driven avatar lipsync

Real-time lipsync in the browser: **mic → MFCCs → MLP → ARKit blendshapes → my 3D avatar**.
I trained it on my own voice, using webcam face tracking (MediaPipe) as automatic labels.
The avatar is a 3D face mesh built from my iPhone Memoji. Memoji are animated with the same
52 ARKit blendshapes this model predicts.

> 🚧 Work in progress. Built step by step as a learning project.

## Progress
- [x] Step 1: Mic capture, waveform + spectrum visualization
- [x] Step 2: MFCC feature extraction (mel filterbank + DCT, written from scratch in `src/mfcc.js`)
- [x] Step 3: Webcam face tracking (MediaPipe FaceLandmarker, 478 landmarks + 52 blendshapes) and avatar-image checker
- [x] Step 4: Synced audio + face recording to JSON (AudioWorklet frame clock at exactly 30 Hz, own FFT, timestamp-matched face labels, mic/camera delay estimate)
- [x] Step 5: Train an MLP (117 → 128 → 64 → 27) in TensorFlow.js: time-based validation split, mic-delay correction, best-epoch selection, live test
- [x] Step 6: 3D avatar from my Memoji: MediaPipe landmarks → Delaunay mesh with a mouth hole, lip-closing rest pose, procedural jaw/lip/round/smile deformation, mouth interior, blinking, brow raises on emphasis, speech-driven head nods, and a waving hand (MediaPipe HandLandmarker → colour-keyed flood-fill mask → onion-peel inpainting of the background)
- [ ] Step 7: Deploy to GitHub Pages

## Run locally
The browser only allows mic and camera access on `https://` or `localhost`, so serve the folder instead of opening the file directly:

```bash
python -m http.server 8000
# open http://localhost:8000
```

Pages: `index.html` (audio → MFCCs), `face.html` (face tracking + avatar checker), `record.html` (record training data), `train.html` (train + live test), `avatar.html` (the 3D talking avatar).

## Known limitations
- **One mic.** The model was trained on recordings from my laptop mic and works best with it. Tested offline, a
  model trained on one mic scores ~0 on another, and mean/variance normalisation didn't fix it. Recording
  training takes on several mics is the next step.
- **One picture.** A single front-facing image only contains the front of the face, so the head turns about ±25°.
