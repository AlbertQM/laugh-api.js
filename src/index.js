const video = document.getElementById("video");
const predictionEl = document.getElementById("prediction");

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("../../ext/face-api.js/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("../../ext/face-api.js/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("../../ext/face-api.js/models"),
  faceapi.nets.faceExpressionNet.loadFromUri("../../ext/face-api.js/models")
]).then(startVideo);

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => (video.srcObject = stream),
    err => console.error(err)
  );
}

video.addEventListener("play", () => {
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();
    const { expressions } = detections[0];
    const bestGuess = Object.keys(expressions).reduce((a, b) =>
      expressions[a] > expressions[b] ? a : b
    );
    predictionEl.innerHTML = bestGuess;
  }, 100);
});
