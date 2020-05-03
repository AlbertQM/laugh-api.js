import Lold from "./Lold";

// How long should the "You laughed!" caption stay on screen after a detection.
const DETECTION_CAPTION_SCREEN_TIME_MS = 1000;
// How often to get predictions from API
const DETECTION_RATE_MS = 100;

const pageName = window.location.pathname.split("/").pop();
const isOfflineVideoVersion = pageName === "video.html";

// Media Element containing the A/V feed
const videoEl = document.getElementById("video") as HTMLVideoElement;
// On screen indicators of confidence.
const predictionEl = document.getElementById("prediction");
const audioConfidenceEl = document.getElementById(
  "confidenceAudio"
) as HTMLProgressElement;
const videoConfidenceEl = document.getElementById(
  "confidenceVideo"
) as HTMLProgressElement;

// Live webcam feed
let audioStream: MediaStream | null = null;

navigator.mediaDevices
  .getUserMedia({ video: {}, audio: {} })
  .then((stream) => {
    if (!isOfflineVideoVersion) {
      videoEl.srcObject = stream;
    }
    audioStream = stream;
    return new Promise((resolve) => (videoEl.onplay = resolve));
  })
  .then(async (_loadEvent) => {
    // Create an instance of Lold.js!
    const lold = new Lold(videoEl, audioStream!, {
      predictionMode: "multimodal",
      videoSourceType: isOfflineVideoVersion ? "video" : "webcam",
    });

    // Load all models and required (from face-api.js and lold.js audio model)
    await lold.loadModels();

    // Start predicting. Predictions run in the background and can be
    // accessed with getters (e.g. getMultimodalPrediction)
    lold.startMultimodalPrediction();

    // Get predictions at a set interval.
    setInterval(() => {
      let [audioConfidence, videoConfidence] = lold.getMultimodalPrediction();

      // If any prediction is null, set it to 0 so
      // it can be safely handled by the code below.
      if (!audioConfidence) {
        audioConfidence = 0;
      }
      if (!videoConfidence) {
        videoConfidence = 0;
      }

      audioConfidenceEl!.value = Number(audioConfidence.toFixed(3)) * 100;
      videoConfidenceEl!.value = Number(videoConfidence.toFixed(3)) * 100;

      const isLaughingAudio = audioConfidence > 0.65;
      const isLaughingVideo = videoConfidence > 0.65;

      if (isLaughingAudio && isLaughingVideo) {
        predictionEl!.innerHTML = "You laughed!";
        // Persist the results in the UI for a few seconds, then
        // clear it.
        setTimeout(() => {
          predictionEl!.innerHTML = "";
          audioConfidenceEl!.value = 0;
          videoConfidenceEl!.value = 0;
        }, DETECTION_CAPTION_SCREEN_TIME_MS);
      }
    }, DETECTION_RATE_MS);
  })
  .catch((err) => console.error(err));
