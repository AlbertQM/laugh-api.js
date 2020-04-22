const fs = require("fs-extra");
const BASE_DIR = "../annotations/";

/**
 * This script reads in the annotations containing the results of
 * testing each model against the multimodal dataset, and computes
 * the various measures (e.g. accuracy, precision, etc).
 */

const DATASET_SIZE = 100;

fs.readdir(BASE_DIR)
  .then((fileList) => {
    fileList.forEach((file) => {
      const contents = fs.readFileSync(`${BASE_DIR}${file}`, {
        encoding: "utf8",
      });
      const rows = contents.split("\n");
      let audioResults = {};
      let videoResults = {};
      let multimodalResults = {};

      rows.forEach((row, idx) => {
        if (idx === 0 || idx > DATASET_SIZE) {
          return;
        }
        // Headers
        // 0    1           2           3           4           5                 6                 7
        // UUID, youtubeID, start(ms), duration(s), true label, video prediction, audio prediction, multimodal prediction
        const [
          _,
          __,
          ___,
          ____,
          // "laugh" or "other"
          label,
          videoPrediction,
          audioPrediction,
          multimodalPrediction,
        ] = row.split(",");

        audioResults = computeResults(label, audioPrediction, audioResults);
        videoResults = computeResults(label, videoPrediction, videoResults);
        multimodalResults = computeResults(
          label,
          multimodalPrediction.trim(),
          multimodalResults
        );
      });
      printResults("Audio", audioResults);
      printResults("Video", videoResults);
      printResults("Multimodal", multimodalResults);
    });
  })
  .catch((e) => {
    console.error("Whoooops, ", e);
  });

function computeResults(label, prediction, results) {
  const _results = { ...results };
  if (label === "laugh") {
    if (prediction === "laugh") {
      if (!_results.TP) {
        _results.TP = 0;
      }
      _results.TP += 1;
    } else {
      if (!_results.FN) {
        _results.FN = 0;
      }
      _results.FN += 1;
    }
  } else {
    if (prediction === "other") {
      if (!_results.TN) {
        _results.TN = 0;
      }
      _results.TN += 1;
    } else {
      if (!_results.FP) {
        _results.FP = 0;
      }
      _results.FP += 1;
    }
  }

  return _results;
}

function printResults(modality, results) {
  const { TP, FP, FN, TN } = results;
  const precision = TP / (TP + FP);
  const recall = TP / (TP + FN);
  const accuracy = (TP + TN) / DATASET_SIZE;
  const specificity = TN / (TN + FP);
  const fallout = FP / (FP + TN);
  const missRate = FN / (FN + TP);
  const f1Score = 2 * ((precision * recall) / (precision + recall));

  function getSanePercentage(value) {
    return (value * 100).toFixed(2);
  }

  console.log(`${modality} results:
TP ${TP} | FN ${FN}
FP ${FP} | TN ${TN}
Accuracy: ${getSanePercentage(accuracy)}%
Precision: ${getSanePercentage(precision)}%
Recall (TPR): ${getSanePercentage(recall)}%
Specificity (TNR): ${getSanePercentage(specificity)}%
Fall-out (FPR): ${getSanePercentage(fallout)}%
Miss rate (FNR): ${getSanePercentage(missRate)}%
F1-Score: ${getSanePercentage(f1Score)}%
`);
}
