// facemodel.js — shared MediaPipe loaders: FaceLandmarker (face.js, record.js, avatar.js)
// and HandLandmarker (avatar.js, to find the hand in the avatar picture).
import {
  FaceLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

export { FaceLandmarker, DrawingUtils };

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

let visionPromise;
const getVision = () => (visionPromise ??= FilesetResolver.forVisionTasks(WASM_URL));

/** @param {'VIDEO'|'IMAGE'} runningMode */
export async function createLandmarker(runningMode) {
  const vision = await getVision();
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode,
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
  try {
    return await FaceLandmarker.createFromOptions(vision, options('GPU'));
  } catch {
    // Some machines can't use the GPU path; CPU is slower but works everywhere.
    return await FaceLandmarker.createFromOptions(vision, options('CPU'));
  }
}

const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

/** Finds up to 2 hands: 21 points each (0 = wrist, 4 = thumb tip, 8 = index tip, …). */
export async function createHandLandmarker() {
  const vision = await getVision();
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
    runningMode: 'IMAGE',
    numHands: 2,
  });
  try {
    return await HandLandmarker.createFromOptions(vision, options('GPU'));
  } catch {
    return await HandLandmarker.createFromOptions(vision, options('CPU'));
  }
}
