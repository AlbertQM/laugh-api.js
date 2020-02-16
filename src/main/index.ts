import * as faceapi from "face-api.js";
// @ts-ignore as `Meyda` doesn't have type defs
import * as Meyda from "meyda";
import { LayersModel, loadLayersModel } from "@tensorflow/tfjs";

const MODELS_PATH = "./models";
// How long should the "You laughed!" caption stay on screen after a detection.
const DETECTION_CAPTION_SCREEN_TIME_MS = 1000;

// Media Element containing the A/V feed
const video = document.getElementById("video") as HTMLVideoElement;
const predictionEl = document.getElementById("prediction");
// Audio setup
const audioContext = new AudioContext();
let source: MediaStreamAudioSourceNode | null = null;
// Laugh-audio model
type Model = LayersModel;
let model: Model | null = null;

function init() {
  const isAudioReady = !!audioContext && typeof Meyda !== "undefined";
  const isVideoReady = !!video;
  if (isAudioReady && isVideoReady) {
    video.addEventListener("play", () => {
      const analyzer = Meyda.createMeydaAnalyzer({
        audioContext: audioContext,
        source,
        bufferSize: 512,
        numberOfMFCCCoefficients: 40,
        featureExtractors: [
          "mfcc",
          "energy",
          "zcr",
          "spectralCentroid",
          "spectralFlatness"
        ],
        callback: ({
          mfcc,
          energy,
          zcr,
          spectralFlatness,
          spectralCentroid
        }: {
          zcr: number[];
          spectralCentroid: number[];
          spectralFlatness: number[];
          mfcc: number[];
          energy: number;
        }) => {
          faceapi.tf.tidy(() => {
            const isTalking = energy > 0.5;
            if (!isTalking) {
              return;
            }
            // If we detect voice activity, use the model to make predictions
            faceapi
              .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
              .withFaceLandmarks()
              .withFaceExpressions()
              .then((detections): any => {
                if (!detections[0]) {
                  return;
                }
                const {
                  expressions: { happy }
                } = detections[0];
                const features = mfcc
                  .concat(zcr)
                  .concat(spectralFlatness)
                  .concat(spectralCentroid);
                const mfccTensor = faceapi.tf.tensor(features, [1, 43]);
                const prediction = model!.predict(
                  mfccTensor
                ) as faceapi.tf.Tensor;
                const [laugh, filler] = prediction.dataSync();
                const max = Math.max(laugh, filler);
                const isLaughingAudio = max === laugh;
                const isLaughingVideo = happy === 1;
                if (isLaughingAudio && isLaughingVideo) {
                  predictionEl!.innerHTML = "You laughed!";
                  setTimeout(() => {
                    predictionEl!.innerHTML = "";
                  }, DETECTION_CAPTION_SCREEN_TIME_MS);
                }
              });
          });
        }
      });
      analyzer.start();
    });
  }
}

function startAV() {
  audioContext.resume();

  const isAVReady = video && audioContext;
  if (!isAVReady) {
    return;
  }

  // Init both models
  init();
  loadAudioModel();

  navigator.getUserMedia(
    { video: {}, audio: {} },
    stream => {
      video.srcObject = stream;
      source = audioContext.createMediaStreamSource(stream);
    },
    err => console.error(err)
  );
}

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_PATH),
  faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_PATH),
  faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_PATH),
  faceapi.nets.faceExpressionNet.loadFromUri(MODELS_PATH)
]).then(startAV);

const loadAudioModel = async () => {
  model = await loadLayersModel("./models/model.json");
};
