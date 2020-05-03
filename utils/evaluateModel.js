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

      const labels = [];
      const audioConfidenceList = [];
      const videoConfidenceList = [];
      // Read the annotations and store the values we are interested in.
      rows.forEach((row, idx) => {
        if (idx === 0 || idx > DATASET_SIZE) {
          return;
        }

        // Headers
        const [
          _UUID,
          _youtubeID,
          _startMs,
          _durationS,
          // "laugh" or "other"
          label,
          _videoPrediction,
          _audioPrediction,
          _multimodalPrediction,
          audioConfidenceValue,
          videoConfidenceValue,
        ] = row.split(",");

        labels.push(label);
        audioConfidenceList.push(audioConfidenceValue);
        videoConfidenceList.push(videoConfidenceValue);
      });

      /**
       * Compute and store results in an obj.
       * Example results of video threshold 0.1 and audio threshold 0.5:
       *
       * {[0.1]: {
       *    [0.5]: {
       *      TP: 20, TN: ...
       *    }
       *  }
       * }
       *
       */
      let resultsByThreshold = {};
      let multimodalResults = {};
      for (
        let videoThreshold = 0.1;
        videoThreshold <= 1;
        videoThreshold += 0.1
      ) {
        for (
          let audioThreshold = 0.1;
          audioThreshold <= 1;
          audioThreshold += 0.1
        ) {
          for (idx = 0; idx < DATASET_SIZE; idx++) {
            // toFixed(1) because computers can't do maths.
            const audioThresholdSane = audioThreshold.toFixed(1);
            const videoThresholdSane = videoThreshold.toFixed(1);

            const label = labels[idx];
            const audioPrediction =
              audioConfidenceList[idx] > audioThresholdSane ? "laugh" : "other";
            const videoPrediction =
              videoConfidenceList[idx] > videoThresholdSane ? "laugh" : "other";
            const bothModelsDetectingLaugh =
              audioPrediction === "laugh" && videoPrediction === "laugh";
            const multimodalPrediction = bothModelsDetectingLaugh
              ? "laugh"
              : "other";

            multimodalResults = computeResults(
              label,
              multimodalPrediction,
              multimodalResults
            );
            if (!resultsByThreshold[videoThresholdSane]) {
              resultsByThreshold[videoThresholdSane] = {};
            }
            if (!resultsByThreshold[videoThresholdSane][audioThresholdSane]) {
              resultsByThreshold[videoThresholdSane][audioThresholdSane] = {};
            }
            resultsByThreshold[videoThresholdSane][
              audioThresholdSane
            ] = sanitizeResults(multimodalResults);
          }
          // Clear results before re-computing them with a different theshold.
          multimodalResults = {};
        }
      }

      printResults("Multimodal", resultsByThreshold[0.1][0.5]);
      const { accuracyList, f1ScoreList } = getMetricsLists(resultsByThreshold);
      console.log(f1ScoreList);
    });
  })
  .catch((e) => {
    console.error("Whoooops, ", e);
  });

function getMetricsLists(results) {
  const metricLists = {
    accuracyList: [],
    recallList: [],
    precisionList: [],
    specificityList: [],
    falloutList: [],
    missRateList: [],
    f1ScoreList: [],
  };
  for (let videoThreshold = 0.1; videoThreshold <= 1; videoThreshold += 0.1) {
    for (let audioThreshold = 0.1; audioThreshold <= 1; audioThreshold += 0.1) {
      // toFixed(1) because computers can't do maths.
      const audioThresholdSane = audioThreshold.toFixed(1);
      const videoThresholdSane = videoThreshold.toFixed(1);

      const {
        precision,
        recall,
        accuracy,
        specificity,
        fallout,
        missRate,
        f1Score,
        results: _,
      } = computeMetrics(results[videoThresholdSane][audioThresholdSane]);
      metricLists.accuracyList.push(accuracy);
      metricLists.precisionList.push(precision);
      metricLists.recallList.push(recall);
      metricLists.specificityList.push(specificity);
      metricLists.falloutList.push(fallout);
      metricLists.missRateList.push(missRate);
      metricLists.f1ScoreList.push(f1Score);
    }
  }
  return metricLists;
}

/**
 *
 * @param label The true label from the dataset.
 * @param prediction The prediction from the model, either "laugh" or "other"
 * @param results An object with TP, TN, FP, FN.
 */
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

function sanitizeResults(results) {
  const saneResults = { ...results };
  if (!saneResults.TP) {
    saneResults.TP = 0;
  }
  if (!saneResults.FP) {
    saneResults.FP = 0;
  }
  if (!saneResults.FN) {
    saneResults.FN = 0;
  }
  if (!saneResults.TN) {
    saneResults.TN = 0;
  }
  return saneResults;
}

/**
 * Given a confusion matrix object {TP, FN, TN, FP}, computes and retuns
 * precision, recall, accuracy, specificity, fallout, miss rate and F1 Score
 *
 * @param results A {TP, TN, FP, FN} object.
 */
function computeMetrics(results) {
  const { TP, FP, FN, TN } = results;

  let precision = TP / (TP + FP);
  let recall = TP / (TP + FN);
  let accuracy = (TP + TN) / DATASET_SIZE;
  let specificity = TN / (TN + FP);
  let fallout = FP / (FP + TN);
  let missRate = FN / (FN + TP);
  let f1Score = 2 * ((precision * recall) / (precision + recall));

  precision = isNaN(precision) ? 0 : precision;
  recall = isNaN(recall) ? 0 : recall;
  specificity = isNaN(specificity) ? 0 : specificity;
  fallout = isNaN(fallout) ? 0 : fallout;
  missRate = isNaN(missRate) ? 0 : missRate;
  f1Score = isNaN(f1Score) ? 0 : f1Score;

  return {
    precision,
    recall,
    accuracy,
    specificity,
    fallout,
    missRate,
    f1Score,
    results,
  };
}

function printResults(modality, results) {
  const {
    precision,
    recall,
    accuracy,
    specificity,
    fallout,
    missRate,
    f1Score,
    results: { TP, FN, FP, TN },
  } = computeMetrics(results);

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
