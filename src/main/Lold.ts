/**
 * @fileOverview Lold.js - multmodal Lightweight Online Laughter Detection
 * @author Alberto Morabito
 * @version 1.0.1
 */

import * as faceapi from "face-api.js";
import * as tf from "@tensorflow/tfjs-core";
// @ts-ignore as `Meyda` doesn't have type defs
import * as Meyda from "meyda";
import { loadGraphModel, GraphModel } from "@tensorflow/tfjs-converter";

const MODELS_PATH = "./models";
const NUM_FEATURES = 43;

const YAMNET_LAUGHTER_CLASS_ID = 13;
const YAMNET_BABY_LAUGHTER_CLASS_ID = 14;
const YAMNET_BELLY_LAUGH_CLASS_ID = 17;

// Laugh-audio model
type AudioModel = GraphModel;
// Yamnet audio model
let model: AudioModel | null = null;

// Features extracted with Meyda and fed to the model
type AudioFeatures = {
  buffer: number[];
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
  private audioModel: AudioModel | null = null;
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
      featureExtractors: ["buffer"],
      callback: async ({ buffer }: AudioFeatures) => {
        const [audioConfidence, videoConfidence] = await processMakePrediction(
          predictionMode,
          videoSource,
          this.videoModelOptions,
          this.audioModel,
          {
            buffer,
          }
        );
        this.predictions = [audioConfidence, videoConfidence];
      },
    });
  }

  /** Load models and weights */
  public loadModels = async () => {
    this.audioModel = await loadGraphModel(`${MODELS_PATH}/yamnet_model.json`);

    // Load face-api.js models
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_PATH),
      faceapi.nets.faceExpressionNet.loadFromUri(MODELS_PATH),
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
  audioModel: AudioModel | null,
  { buffer }: AudioFeatures
): Promise<Array<number | undefined>> {
  switch (predictionMode) {
    case "audio":
      // Return [audioConfidence, undefined]
      return [makeYamnetPrediction(audioModel, buffer), undefined];

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
        { buffer }
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
  audioModel: AudioModel | null,
  { buffer }: AudioFeatures
): Promise<Array<number | undefined>> {
  const audioConfidence = makeYamnetPrediction(audioModel, buffer);

  const videoConfidence = await makeVideoPrediction(
    videoSource,
    videoModelOptions
  );

  return [audioConfidence as number, videoConfidence];
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
    expressions: { happy },
  } = detections[0];
  return happy;
}

function makeYamnetPrediction(
  yamnetModel: AudioModel | null,
  buffer: number[]
) {
  if (!yamnetModel) {
    throw new Error("Yamnet model not found.");
  }

  const [scores, _embeddings, _spectrogram] = yamnetModel.predict(
    tf.tensor(buffer)
  ) as tf.Tensor[];
  // List of classes available at https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
  const predictedClassId = scores.mean(0).argMax().dataSync()[0];
  console.log(classes[predictedClassId]);
  return predictedClassId === YAMNET_LAUGHTER_CLASS_ID ? 1 : 0;
}

const classes = `Speech
Child speech, kid speaking
Conversation
Narration, monologue
Babbling
Speech synthesizer
Shout
Bellow
Whoop
Yell
Children shouting
Screaming
Whispering
Laughter
Baby laughter
Giggle
Snicker
Belly laugh
Chuckle, chortle
Crying, sobbing
Baby cry, infant cry
Whimper
Wail, moan
Sigh
Singing
Choir
Yodeling
Chant
Mantra
Child singing
Synthetic singing
Rapping
Humming
Groan
Grunt
Whistling
Breathing
Wheeze
Snoring
Gasp
Pant
Snort
Cough
Throat clearing
Sneeze
Sniff
Run
Shuffle
Walk, footsteps
Chewing, mastication
Biting
Gargling
Stomach rumble
Burping, eructation
Hiccup
Fart
Hands
Finger snapping
Clapping
Heart sounds, heartbeat
Heart murmur
Cheering
Applause
Chatter
Crowd
Hubbub, speech noise, speech babble
Children playing
Animal
Domestic animals, pets
Dog
Bark
Yip
Howl
Bow-wow
Growling
Whimper (dog)
Cat
Purr
Meow
Hiss
Caterwaul
Livestock, farm animals, working animals
Horse
Clip-clop
Neigh, whinny
Cattle, bovinae
Moo
Cowbell
Pig
Oink
Goat
Bleat
Sheep
Fowl
Chicken, rooster
Cluck
Crowing, cock-a-doodle-doo
Turkey
Gobble
Duck
Quack
Goose
Honk
Wild animals
Roaring cats (lions, tigers)
Roar
Bird
Bird vocalization, bird call, bird song
Chirp, tweet
Squawk
Pigeon, dove
Coo
Crow
Caw
Owl
Hoot
Bird flight, flapping wings
Canidae, dogs, wolves
Rodents, rats, mice
Mouse
Patter
Insect
Cricket
Mosquito
Fly, housefly
Buzz
Bee, wasp, etc.
Frog
Croak
Snake
Rattle
Whale vocalization
Music
Musical instrument
Plucked string instrument
Guitar
Electric guitar
Bass guitar
Acoustic guitar
Steel guitar, slide guitar
Tapping (guitar technique)
Strum
Banjo
Sitar
Mandolin
Zither
Ukulele
Keyboard (musical)
Piano
Electric piano
Organ
Electronic organ
Hammond organ
Synthesizer
Sampler
Harpsichord
Percussion
Drum kit
Drum machine
Drum
Snare drum
Rimshot
Drum roll
Bass drum
Timpani
Tabla
Cymbal
Hi-hat
Wood block
Tambourine
Rattle (instrument)
Maraca
Gong
Tubular bells
Mallet percussion
Marimba, xylophone
Glockenspiel
Vibraphone
Steelpan
Orchestra
Brass instrument
French horn
Trumpet
Trombone
Bowed string instrument
String section
Violin, fiddle
Pizzicato
Cello
Double bass
Wind instrument, woodwind instrument
Flute
Saxophone
Clarinet
Harp
Bell
Church bell
Jingle bell
Bicycle bell
Tuning fork
Chime
Wind chime
Change ringing (campanology)
Harmonica
Accordion
Bagpipes
Didgeridoo
Shofar
Theremin
Singing bowl
Scratching (performance technique)
Pop music
Hip hop music
Beatboxing
Rock music
Heavy metal
Punk rock
Grunge
Progressive rock
Rock and roll
Psychedelic rock
Rhythm and blues
Soul music
Reggae
Country
Swing music
Bluegrass
Funk
Folk music
Middle Eastern music
Jazz
Disco
Classical music
Opera
Electronic music
House music
Techno
Dubstep
Drum and bass
Electronica
Electronic dance music
Ambient music
Trance music
Music of Latin America
Salsa music
Flamenco
Blues
Music for children
New-age music
Vocal music
A capella
Music of Africa
Afrobeat
Christian music
Gospel music
Music of Asia
Carnatic music
Music of Bollywood
Ska
Traditional music
Independent music
Song
Background music
Theme music
Jingle (music)
Soundtrack music
Lullaby
Video game music
Christmas music
Dance music
Wedding music
Happy music
Sad music
Tender music
Exciting music
Angry music
Scary music
Wind
Rustling leaves
Wind noise (microphone)
Thunderstorm
Thunder
Water
Rain
Raindrop
Rain on surface
Stream
Waterfall
Ocean
Waves, surf
Steam
Gurgling
Fire
Crackle
Vehicle
Boat, Water vehicle
Sailboat, sailing ship
Rowboat, canoe, kayak
Motorboat, speedboat
Ship
Motor vehicle (road)
Car
Vehicle horn, car horn, honking
Toot
Car alarm
Power windows, electric windows
Skidding
Tire squeal
Car passing by
Race car, auto racing
Truck
Air brake
Air horn, truck horn
Reversing beeps
Ice cream truck, ice cream van
Bus
Emergency vehicle
Police car (siren)
Ambulance (siren)
Fire engine, fire truck (siren)
Motorcycle
Traffic noise, roadway noise
Rail transport
Train
Train whistle
Train horn
Railroad car, train wagon
Train wheels squealing
Subway, metro, underground
Aircraft
Aircraft engine
Jet engine
Propeller, airscrew
Helicopter
Fixed-wing aircraft, airplane
Bicycle
Skateboard
Engine
Light engine (high frequency)
Dental drill, dentist's drill
Lawn mower
Chainsaw
Medium engine (mid frequency)
Heavy engine (low frequency)
Engine knocking
Engine starting
Idling
Accelerating, revving, vroom
Door
Doorbell
Ding-dong
Sliding door
Slam
Knock
Tap
Squeak
Cupboard open or close
Drawer open or close
Dishes, pots, and pans
Cutlery, silverware
Chopping (food)
Frying (food)
Microwave oven
Blender
Water tap, faucet
Sink (filling or washing)
Bathtub (filling or washing)
Hair dryer
Toilet flush
Toothbrush
Electric toothbrush
Vacuum cleaner
Zipper (clothing)
Keys jangling
Coin (dropping)
Scissors
Electric shaver, electric razor
Shuffling cards
Typing
Typewriter
Computer keyboard
Writing
Alarm
Telephone
Telephone bell ringing
Ringtone
Telephone dialing, DTMF
Dial tone
Busy signal
Alarm clock
Siren
Civil defense siren
Buzzer
Smoke detector, smoke alarm
Fire alarm
Foghorn
Whistle
Steam whistle
Mechanisms
Ratchet, pawl
Clock
Tick
Tick-tock
Gears
Pulleys
Sewing machine
Mechanical fan
Air conditioning
Cash register
Printer
Camera
Single-lens reflex camera
Tools
Hammer
Jackhammer
Sawing
Filing (rasp)
Sanding
Power tool
Drill
Explosion
Gunshot, gunfire
Machine gun
Fusillade
Artillery fire
Cap gun
Fireworks
Firecracker
Burst, pop
Eruption
Boom
Wood
Chop
Splinter
Crack
Glass
Chink, clink
Shatter
Liquid
Splash, splatter
Slosh
Squish
Drip
Pour
Trickle, dribble
Gush
Fill (with liquid)
Spray
Pump (liquid)
Stir
Boiling
Sonar
Arrow
Whoosh, swoosh, swish
Thump, thud
Thunk
Electronic tuner
Effects unit
Chorus effect
Basketball bounce
Bang
Slap, smack
Whack, thwack
Smash, crash
Breaking
Bouncing
Whip
Flap
Scratch
Scrape
Rub
Roll
Crushing
Crumpling, crinkling
Tearing
Beep, bleep
Ping
Ding
Clang
Squeal
Creak
Rustle
Whir
Clatter
Sizzle
Clicking
Clickety-clack
Rumble
Plop
Jingle, tinkle
Hum
Zing
Boing
Crunch
Silence
Sine wave
Harmonic
Chirp tone
Sound effect
Pulse
Inside, small room
Inside, large room or hall
Inside, public space
Outside, urban or manmade
Outside, rural or natural
Reverberation
Echo
Noise
Environmental noise
Static
Mains hum
Distortion
Sidetone
Cacophony
White noise
Pink noise
Throbbing
Vibration
Television
Radio
Field recording`.split("\n");
