"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const faceapi = require("face-api.js");
// @ts-ignore as `Meyda` doesn't have type defs
const Meyda = require("meyda");
const tfjs_1 = require("@tensorflow/tfjs");
const MODELS_PATH = "./models";
// How long should the "You laughed!" caption stay on screen after a detection.
const DETECTION_CAPTION_SCREEN_TIME_MS = 1000;
const isMobileVersion = window.location.pathname === "/mobile.html";
// Media Element containing the A/V feed
const videoEl = document.getElementById("video");
const predictionEl = document.getElementById("prediction");
const audioConfidenceEl = document.getElementById("confidenceAudio");
const videoConfidenceEl = document.getElementById("confidenceVideo");
const initialMessage = document.getElementById("initialMessage");
// Audio setup
const audioContext = new AudioContext();
let source = // When using audio from live webcam feed
 null;
let model = null;
const tinyFaceDetector = new faceapi.TinyFaceDetectorOptions();
/** Takes AudioFeatures as input and predicts laughter using both
 *  audio and video model.
 */
function makePrediction({ mfcc, energy, zcr, spectralFlatness, spectralCentroid }) {
    faceapi.tf.tidy(() => {
        if (isMobileVersion && energy < 0.5) {
            return;
        }
        faceapi
            .detectAllFaces(videoEl, tinyFaceDetector)
            .withFaceExpressions()
            .then((detections) => {
            if (!detections[0]) {
                return;
            }
            const { expressions: { happy } } = detections[0];
            const features = mfcc
                .concat(zcr)
                .concat(spectralFlatness)
                .concat(spectralCentroid);
            const mfccTensor = faceapi.tf.tensor(features, [1, 43]);
            const prediction = model.predict(mfccTensor);
            const [laugh, speech, silence] = prediction.dataSync();
            const isLaughingAudio = laugh > 0.65;
            const isLaughingVideo = happy > 0.65;
            // Update the progress bars on screen based on how confident
            // the models are
            audioConfidenceEl.value = Number(laugh.toFixed(3)) * 100;
            videoConfidenceEl.value = Number(happy.toFixed(3)) * 100;
            // Remove the "Loading.." message as we're ready to show predictions
            initialMessage.remove();
            // If we are not detecting sound, there is no chance of
            // laugh. This check could be done before the prediction to
            // save computation. However, we want to show the "video confidence"
            // update in real-time. This might change in the future.
            // TODO: Move this check as early as possible to save computation
            // once we don't need to show the video confidence on screen anymore.
            const isTalking = energy > 0.5;
            if (!isTalking) {
                return;
            }
            if (isLaughingAudio && isLaughingVideo) {
                predictionEl.innerHTML = "You laughed!";
                // Persist the results in the UI for a few seconds, then
                // clear it.
                setTimeout(() => {
                    predictionEl.innerHTML = "";
                    audioConfidenceEl.value = 0;
                    videoConfidenceEl.value = 0;
                }, DETECTION_CAPTION_SCREEN_TIME_MS);
            }
        });
    });
}
function startPredicting() {
    const isAudioReady = audioContext.state === "running" && typeof Meyda !== "undefined";
    const isVideoReady = !!videoEl;
    const computePreduction = () => {
        const analyzer = Meyda.createMeydaAnalyzer({
            audioContext: audioContext,
            source,
            bufferSize: 4096,
            numberOfMFCCCoefficients: 40,
            featureExtractors: [
                "mfcc",
                "energy",
                "zcr",
                "spectralCentroid",
                "spectralFlatness"
            ],
            callback: ({ mfcc, energy, zcr, spectralFlatness, spectralCentroid }) => makePrediction({
                mfcc,
                energy,
                zcr,
                spectralFlatness,
                spectralCentroid
            })
        });
        analyzer.start();
        videoEl.removeEventListener("play", computePreduction);
    };
    if (isAudioReady && isVideoReady) {
        videoEl.addEventListener("play", computePreduction);
    }
}
function handleBeginInteraction() {
    if (audioContext.state !== "running") {
        audioContext.resume();
    }
    initialMessage.innerHTML = "Loading models..";
    Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_PATH),
        faceapi.nets.faceExpressionNet.loadFromUri(MODELS_PATH)
    ]).then(startAV);
}
// Chrome 70 or above requires users gestures to enable WebAudio API.
// We need to resume the audio context after users made an action.
window.addEventListener("pointerdown", handleBeginInteraction);
const loadAudioModel = async () => {
    model = await tfjs_1.loadLayersModel("./models/model.json");
};
function startAV() {
    window.removeEventListener("pointerdown", handleBeginInteraction);
    const isOfflineVideo = window.location.pathname === "/video.html";
    const isAVReady = videoEl && audioContext.state === "running";
    if (!isAVReady) {
        return;
    }
    loadAudioModel();
    startPredicting();
    navigator.getUserMedia({ video: {}, audio: {} }, stream => {
        if (!isOfflineVideo) {
            videoEl.srcObject = stream;
        }
        source = !isOfflineVideo
            ? audioContext.createMediaStreamSource(stream)
            : audioContext.createMediaElementSource(videoEl);
    }, err => console.error(err));
}
//# sourceMappingURL=index.js.map