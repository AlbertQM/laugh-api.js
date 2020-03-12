import * as faceapi from "face-api.js";
// @ts-ignore as `Meyda` doesn't have type defs
import * as Meyda from "meyda";
import { LayersModel, loadLayersModel } from "@tensorflow/tfjs";

const MODELS_PATH = "./models";
const NUM_FEATURES = 43;

// Laugh-audio model
type Model = LayersModel;
let model: Model | null = null;

// Features extracted with Meyda and fed to the model
type AudioFeatures = {
  zcr: number;
  spectralCentroid: number;
  spectralFlatness: number;
  mfcc: number[];
  energy: number;
};

export default class Lold {
  /** Meyda analyser. Used to extract audio features */
  private analyser: any;
  private audioContext = new AudioContext();
  private audioModel: Model | null = null;
  private source: MediaStreamAudioSourceNode;
  private videoModelOptions = new faceapi.TinyFaceDetectorOptions();
  /** An array that contains the confidence (0 to 1) of the prediction
   * being laugh (audio model) or happy (laugh-api.js)
   */
  private predictions: Array<number | undefined> = [];

  constructor(videoSource: faceapi.TNetInput, audioStream: MediaStream) {
    this.source = this.audioContext.createMediaStreamSource(audioStream);

    this.analyser = Meyda.createMeydaAnalyzer({
      audioContext: this.audioContext,
      source: this.source,
      bufferSize: 4096,
      numberOfMFCCCoefficients: 40,
      featureExtractors: [
        "mfcc",
        "energy",
        "zcr",
        "spectralCentroid",
        "spectralFlatness"
      ],
      callback: async ({
        mfcc,
        energy,
        zcr,
        spectralFlatness,
        spectralCentroid
      }: AudioFeatures) => {
        const [
          audioConfidence,
          videoConfidence
        ] = await makeMultimodalPrediction(
          videoSource,
          this.videoModelOptions,
          this.audioModel,
          {
            mfcc,
            energy,
            zcr,
            spectralFlatness,
            spectralCentroid
          }
        );
        this.predictions = [audioConfidence, videoConfidence];
      }
    });
  }

  /** Load models and weights */
  public loadModels = async () => {
    this.audioModel = await loadLayersModel(`${MODELS_PATH}/model.json`);
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_PATH),
      faceapi.nets.faceExpressionNet.loadFromUri(MODELS_PATH)
    ]);
  };

  /** Start the Meyda analyser. This calls both models and updates
   * the predictions value.
   */
  public startMultimodalPrediction = async () => {
    this.analyser.start();
  };

  /** Stop the Meyda analyser. */
  public stopMultimodalPrediction = async () => {
    this.analyser.stop();
  };

  /** Get the current predictions. */
  public getMultimodalPrediction = () => {
    return this.predictions;
  };
}

/** Return laughter prediction/confidence using both
 *  audio and video (face-api.js) model.
 */
async function makeMultimodalPrediction(
  /** Source of the video signal. Usually an HTMLVideoElement */
  videoSource: faceapi.TNetInput,
  videoModelOptions: faceapi.TinyFaceDetectorOptions,
  audioModel: Model | null,
  { mfcc, energy, zcr, spectralFlatness, spectralCentroid }: AudioFeatures
): Promise<Array<number | undefined>> {
  const audioConfidence = makeAudioPrediction(audioModel, {
    mfcc,
    energy,
    zcr,
    spectralFlatness,
    spectralCentroid
  });

  const videoConfidence = await makeVideoPrediction(
    videoSource,
    videoModelOptions
  );

  return [audioConfidence, videoConfidence];
}

/** Call the audio model to detect laughter */
function makeAudioPrediction(
  audioModel: Model | null,
  { mfcc, energy, zcr, spectralFlatness, spectralCentroid }: AudioFeatures
) {
  if (!audioModel) {
    throw new Error("Audio model not found.");
  }

  return faceapi.tf.tidy(() => {
    // Return early if we're not detecting any audio signal.
    const isTalking = energy > 0.5;
    if (!isTalking) {
      return;
    }

    const features = mfcc.concat([zcr, spectralFlatness, spectralCentroid]);
    const mfccTensor = faceapi.tf.tensor(features, [1, NUM_FEATURES]);
    const prediction = audioModel.predict(mfccTensor) as faceapi.tf.Tensor;
    const [laugh, speech, silence] = prediction.dataSync();
    return laugh;
  });
}

/** Call face-api.js to detect emotions via video */
async function makeVideoPrediction(
  videoSource: faceapi.TNetInput,
  videoModelOptions: faceapi.TinyFaceDetectorOptions
) {
  const detections = await faceapi
    .detectAllFaces(videoSource, videoModelOptions)
    .withFaceExpressions();

  if (!detections[0]) {
    return;
  }

  const {
    // The emotion "Happiness" can be associated with
    // laughter, thus we take it into consideration
    expressions: { happy }
  } = detections[0];
  return happy;
}
