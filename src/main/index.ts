import * as tf from "@tensorflow/tfjs";
// @ts-ignore as `Meyda` doesn't have type defs
import * as Meyda from "meyda";
import { Tensor } from "@tensorflow/tfjs-core";

type Model = tf.LayersModel;
let model: Model | null = null;
(async function loadModel() {
  model = await tf.loadLayersModel("./models/model.json");
  console.log("Model loaded.");
})();

const audioContext = new AudioContext();
let source: MediaStreamAudioSourceNode | null = null;
// Gain access to microphone
navigator.getUserMedia(
  { audio: {} },
  stream => {
    source = audioContext.createMediaStreamSource(stream);
  },
  err => console.error(err)
);

const pred = document.getElementById("prediction");
let isOn = false;
function getMfcc() {
  if (typeof Meyda === "undefined") {
    console.log("Meyda could not be found! Have you included it?");
  } else {
    const analyzer = Meyda.createMeydaAnalyzer({
      audioContext: audioContext,
      source,
      bufferSize: 512,
      numberOfMFCCCoefficients: 40,
      featureExtractors: ["mfcc", "energy"],
      callback: ({ mfcc, energy }: { mfcc: number[]; energy: number }) => {
        tf.tidy(() => {
          if (energy < 1) {
            pred!.innerHTML = "in silence";
            return;
          }
          const mfccTensor = (tf.tensor(mfcc, [1, 40]) as unknown) as Tensor;
          const prediction = model!.predict(mfccTensor) as Tensor;
          const [speech, laugh, filler] = prediction.dataSync();

          const max = Math.max(laugh, filler, speech);
          let label = "";
          if (max === laugh) label = "laugh";
          if (max === filler) label = "filler";
          if (max === speech) label = "speech";

          pred!.innerHTML = label;
        });
      }
    });
    if (!isOn) {
      analyzer.start();
    } else {
      analyzer.stop();
    }
    isOn = !isOn;
  }
}

const btn = document.getElementById("mike");
btn!.addEventListener("click", getMfcc);
