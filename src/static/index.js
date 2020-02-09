"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const faceapi = require("face-api.js");
// @ts-ignore as `Meyda` doesn't have type defs
const Meyda = require("meyda");
const tfjs_1 = require("@tensorflow/tfjs");
const MODELS_PATH = "./models";
// Media Element containing the A/V feed
const video = document.getElementById("video");
const predictionEl = document.getElementById("prediction");
// Audio setup
const audioContext = new AudioContext();
let source = null;
let model = null;
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
                featureExtractors: ["mfcc", "energy"],
                callback: ({ mfcc, energy }) => {
                    faceapi.tf.tidy(() => {
                        let label = "in silence";
                        const isTalking = energy > 0.5;
                        if (!isTalking) {
                            predictionEl.innerHTML = label;
                            return;
                        }
                        // If we detect voice activity, use the model to make predictions
                        faceapi
                            .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                            .withFaceLandmarks()
                            .withFaceExpressions()
                            .then((detections) => {
                            if (!detections[0]) {
                                return;
                            }
                            const { expressions } = detections[0];
                            const bestGuess = Object.keys(expressions).reduce((a, b) => 
                            // @ts-ignore
                            expressions[a] > expressions[b] ? a : b);
                            const mfccTensor = faceapi.tf.tensor(mfcc, [1, 40]);
                            const prediction = model.predict(mfccTensor);
                            const [laugh, filler] = prediction.dataSync();
                            const max = Math.max(laugh, filler);
                            const isLaughingAudio = max === laugh;
                            const isLaughingVideo = bestGuess === "happy";
                            if (isLaughingAudio && isLaughingVideo) {
                                predictionEl.innerHTML = "laughing";
                            }
                            else {
                                predictionEl.innerHTML = "talking";
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
    navigator.getUserMedia({ video: {}, audio: {} }, stream => {
        video.srcObject = stream;
        source = audioContext.createMediaStreamSource(stream);
    }, err => console.error(err));
}
Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_PATH),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_PATH),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_PATH),
    faceapi.nets.faceExpressionNet.loadFromUri(MODELS_PATH)
]).then(startAV);
const loadAudioModel = async () => {
    model = await tfjs_1.loadLayersModel("./models/model.json");
};
//# sourceMappingURL=index.js.map