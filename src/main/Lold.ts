/**
 * @fileOverview Lold.js - multmodal Lightweight Online Laughter Detection
 * @author Alberto Morabito
 * @version 1.0.1
 */

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

// Extra configuration applied to each Lold.js instance
type LoldOptions = {
  predictionMode: PredictionMode;
  videoSourceType: "webcam" | "video";
};

/**
 * @class Main Lold API
 * @classdesc
 * This class provides an interface to
 * 1. A ML model that detects laughter via sound signal; this
 *    model was developed as part of this project.
 * 2. face-api.js model that detects happiness via video signal.
 *
 * These models are used singularly or in conjunction to detect laughter.
 *
 * @constructor
 * @param videoSource Source of the video signal. Usually an HTMLVideoElement
 * @param audioStream A stream of media content. E.g. audio stream from webcam
 * @param predictionMode Whether to detect laughter using only audio, video or both (multimodal). Default multimodal.
 * @param videoSourceType What kind of video signal is fed to the model. Default webcam.
 *
 * @public `loadModels` - Loads weights and models needed to give predictions
 * @public `startMultimodalPrediction` - Start the Meyda analyser
 * @public `stopMultimodalPrediction` - Stop the Meyda analyser
 * @public `getMultimodalPrediction` - Call both audio and video model for a
 * prediction and return their confidence value
 *
 */
export default class Lold {
  /** Meyda analyser. Used to extract audio features */
  private analyser: any;
  private audioContext = new AudioContext();
  private audioModel: Model | null = null;
  private source: MediaStreamAudioSourceNode | MediaElementAudioSourceNode;
  private videoModelOptions = new faceapi.TinyFaceDetectorOptions();
  /** An array that contains the confidence (0 to 1) of the prediction
   * being laugh (audio model) or happy (laugh-api.js)
   */
  private predictions: Array<number | undefined> = [];

  constructor(
    videoSource: HTMLVideoElement,
    audioStream: MediaStream,
    { predictionMode = "multimodal", videoSourceType = "webcam" }: LoldOptions
  ) {
    // Important that we create the media stream source here, to make sure
    // that it refers to the right audioContext box
    this.source =
      videoSourceType === "webcam"
        ? this.audioContext.createMediaStreamSource(audioStream)
        : this.audioContext.createMediaElementSource(videoSource);

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
        const [audioConfidence, videoConfidence] = await processMakePrediction(
          predictionMode,
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
    // Load face-api.js models
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

type PredictionMode = "audio" | "video" | "multimodal";
/**
 * Choose the modality in which to make the prediction.
 * A wrapper around existing unimodal/multimodal functions.
 */
async function processMakePrediction(
  predictionMode: PredictionMode,
  videoSource: faceapi.TNetInput,
  videoModelOptions: faceapi.TinyFaceDetectorOptions,
  audioModel: Model | null,
  { mfcc, energy, zcr, spectralFlatness, spectralCentroid }: AudioFeatures
): Promise<Array<number | undefined>> {
  switch (predictionMode) {
    case "audio":
      // Return [audioConfidence, undefined]
      return [
        makeAudioPrediction(audioModel, {
          mfcc,
          energy,
          zcr,
          spectralFlatness,
          spectralCentroid
        }),
        undefined
      ];

    case "video":
      // Return [undefined, videoConfidence]
      const videoConfidence = await makeVideoPrediction(
        videoSource,
        videoModelOptions
      );
      return [undefined, videoConfidence];

    case "multimodal":
      // Return [audioConfidence, videoConfidence]
      return await makeMultimodalPrediction(
        videoSource,
        videoModelOptions,
        audioModel,
        { mfcc, energy, zcr, spectralFlatness, spectralCentroid }
      );

    default:
      throw new Error(
        `Specified prediction mode "${predictionMode}" is not supported. Try using either "audio", "video" or "multimodal"`
      );
  }
}

/** Return laughter prediction/confidence using both
 *  audio and video (face-api.js) model.
 *
 * @param videoSource Source of the video signal. Usually an HTMLVideoElement
 * @param videoModelOptions Select which model face-api.js should use. Using TinyFaceDetector
 * as it's the most lightweight.
 * @param audioModel The audio model developer as part of Lold.js
 * @param AudioFeatures The features extracted with Meyda to feed to the audio model
 */
async function makeMultimodalPrediction(
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

/**
 * Call the audio model to detect laughter.
 * Returns the confidence of the model.
 */
function makeAudioPrediction(
  audioModel: Model | null,
  { mfcc, energy, zcr, spectralFlatness, spectralCentroid }: AudioFeatures
) {
  if (!audioModel) {
    throw new Error("Audio model not found.");
  }

  return faceapi.tf.tidy(() => {
    // Return early if we're not detecting any audio signal.
    // Computationally cheap implementation of a VAD.
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

/**
 * Call face-api.js to detect emotions via video.
 * Returns the confidence of the model.
 * */
async function makeVideoPrediction(
  videoSource: faceapi.TNetInput,
  videoModelOptions: faceapi.TinyFaceDetectorOptions
) {
  const detections = await faceapi
    .detectAllFaces(videoSource, videoModelOptions)
    .withFaceExpressions();

  // No faces detected
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
