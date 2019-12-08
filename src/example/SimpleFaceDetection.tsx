import React, { useEffect, useRef, useCallback } from "react";
import * as faceapi from "face-api.js";

const MODELS_PATH = "/models";

export default function SimpleFaceDetection() {
  const video = useRef<HTMLVideoElement | null>(null);
  const predictionLabel = useRef<HTMLElement | null>(null);

  const startPredicting = useCallback(() => {
    console.log("about to predict");
    const { current: videoEl } = video;
    const { current: predictionEl } = predictionLabel;

    if (videoEl && predictionEl) {
      videoEl.addEventListener("play", () => {
        setInterval(async () => {
          const detections = await faceapi
            .detectAllFaces(videoEl, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceExpressions();
          const { expressions } = detections[0];
          const bestGuess = Object.keys(expressions).reduce((a, b) =>
            // @ts-ignore
            expressions[a] > expressions[b] ? a : b
          );
          predictionEl.innerHTML = bestGuess;
        }, 100);
      });
    }

    navigator.getUserMedia(
      { video: {} },
      stream => (((videoEl as unknown) as HTMLVideoElement).srcObject = stream),
      err => console.error(err)
    );
  }, []);

  // Load models
  useEffect(() => {
    console.log("run");
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_PATH),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_PATH),
      faceapi.nets.faceExpressionNet.loadFromUri(MODELS_PATH)
    ]).then(startPredicting);
  }, [startPredicting]);

  return (
    <div>
      <h1>
        You look <span ref={predictionLabel}></span>
      </h1>
      <video ref={video} width="720" height="560" muted autoPlay></video>
    </div>
  );
}
