const MODELS_PATH = "../../ext/face-api.js/models";
// Media Element containing the A/V feed
const video = document.getElementById("video");
const predictionEl = document.getElementById("prediction");
function init() {
    if (video && predictionEl) {
        video.addEventListener("play", () => {
            setInterval(async () => {
                // @ts-ignore
                const detections = await faceapi
                    // @ts-ignore
                    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                    .withFaceLandmarks()
                    .withFaceExpressions();
                const { expressions } = detections[0];
                const bestGuess = Object.keys(expressions).reduce((a, b) => 
                // @ts-ignore
                expressions[a] > expressions[b] ? a : b);
                predictionEl.innerHTML = bestGuess;
            }, 100);
        });
    }
}
function startVideo() {
    if (!video) {
        return;
    }
    init();
    navigator.getUserMedia({ video: {} }, stream => (video.srcObject = stream), err => console.error(err));
}
Promise.all([
    // @ts-ignore
    faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_PATH),
    // @ts-ignore
    faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_PATH),
    // @ts-ignore
    faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_PATH),
    // @ts-ignore
    faceapi.nets.faceExpressionNet.loadFromUri(MODELS_PATH)
]).then(startVideo);
//# sourceMappingURL=index.js.map