async function plot(pointsArray, featureName, predictedPointsArray = null) {
  const values = [pointsArray.slice(0, 1000)];
  const series = ["original"];
  if (Array.isArray(predictedPointsArray)) {
    values.push(predictedPointsArray);
    series.push("predicted");
  }

  tfvis.render.scatterplot(
    { name: `${featureName} vs House Price` },
    { values, series },
    {
      xLabel: featureName,
      yLabel: "Price",
      height: 300,
    }
  );
}

async function plotPredictionLine() {
  const [xs, ys] = tf.tidy(() => {
    const normalisedXs = tf.linspace(0, 1, 100);
    const normalisedYs = model.predict(normalisedXs.reshape([100, 1]));

    const xs = denormalise(
      normalisedXs,
      normalisedFeature.min,
      normalisedFeature.max
    );
    const ys = denormalise(
      normalisedYs,
      normalisedLabel.min,
      normalisedLabel.max
    );

    return [xs.dataSync(), ys.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, index) => {
    return { x: val, y: ys[index] };
  });

  await plot(points, "Square feet", predictedPoints);
}

function normalise(tensor, previousMin = null, previousMax = null) {
  const min = previousMin || tensor.min();
  const max = previousMax || tensor.max();
  const normalisedTensor = tensor.sub(min).div(max.sub(min));
  return {
    tensor: normalisedTensor,
    min,
    max,
  };
}

function denormalise(tensor, min, max) {
  const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
  return denormalisedTensor;
}

let model;
function createModel() {
  model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 1,
      useBias: true,
      activation: "linear",
      inputDim: 1,
    })
  );

  const optimizer = tf.train.sgd(0.1);
  model.compile({
    loss: "meanSquaredError",
    optimizer,
  });

  return model;
}

async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {
  const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
    { name: "Training Performance" },
    ["loss"]
  );

  return model.fit(trainingFeatureTensor, trainingLabelTensor, {
    batchSize: 32,
    epochs: 20,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd,
      onEpochBegin: async function () {
        await plotPredictionLine();
        const layer = model.getLayer(undefined, 0);
        tfvis.show.layer({ name: "Layer 1" }, layer);
      },
    },
  });
}

async function predict() {
  const predictionInput = parseInt(
    document.getElementById("prediction-input").value
  );
  if (isNaN(predictionInput)) {
    alert("Please enter a valid number");
  } else if (predictionInput < 200) {
    alert("Please enter a value above 200 sqft");
  } else {
    tf.tidy(() => {
      const inputTensor = tf.tensor1d([predictionInput]);
      const normalisedInput = normalise(
        inputTensor,
        normalisedFeature.min,
        normalisedFeature.max
      );
      const normalisedOutputTensor = model.predict(normalisedInput.tensor);
      const outputTensor = denormalise(
        normalisedOutputTensor,
        normalisedLabel.min,
        normalisedLabel.max
      );
      const outputValue = outputTensor.dataSync()[0];
      const outputValueRounded = (outputValue / 1000).toFixed(0) * 1000;
      document.getElementById("prediction-output").innerHTML =
        `The predicted house price is <br>` +
        `<span style="font-size: 2em">\$${outputValueRounded}</span>`;
    });
  }
}

const storageID = "kc-house-price-regression";
async function save() {
  const saveResults = await model.save(`localstorage://${storageID}`);
  document.getElementById(
    "model-status"
  ).innerHTML = `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`;
}

async function load() {
  const storageKey = `localstorage://${storageID}`;
  const models = await tf.io.listModels();
  const modelInfo = models[storageKey];
  if (modelInfo) {
    model = await tf.loadLayersModel(storageKey);

    tfvis.show.modelSummary({ name: "Model summary" }, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1" }, layer);

    await plotPredictionLine();

    document.getElementById(
      "model-status"
    ).innerHTML = `Trained (saved ${modelInfo.dateSaved})`;
    document.getElementById("predict-button").removeAttribute("disabled");
  } else {
    alert("Could not load: no saved model found");
  }
}

async function test() {
  const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
  const loss = (await lossTensor.dataSync())[0];
  console.log(`Testing set loss: ${loss}`);

  document.getElementById(
    "testing-status"
  ).innerHTML = `Testing set loss: ${loss.toPrecision(5)}`;
}

async function train() {
  // Disable all buttons and update status
  ["train", "test", "load", "predict", "save"].forEach((id) => {
    document
      .getElementById(`${id}-button`)
      .setAttribute("disabled", "disabled");
  });
  document.getElementById("model-status").innerHTML = "Training...";

  const model = createModel();
  tfvis.show.modelSummary({ name: "Model summary" }, model);
  const layer = model.getLayer(undefined, 0);
  tfvis.show.layer({ name: "Layer 1" }, layer);
  await plotPredictionLine();

  const result = await trainModel(
    model,
    trainingFeatureTensor,
    trainingLabelTensor
  );
  console.log(result);
  const trainingLoss = result.history.loss.pop();
  console.log(`Training set loss: ${trainingLoss}`);
  const validationLoss = result.history.val_loss.pop();
  console.log(`Validation set loss: ${validationLoss}`);

  document.getElementById("model-status").innerHTML =
    "Trained (unsaved)\n" +
    `Loss: ${trainingLoss.toPrecision(5)}\n` +
    `Validation loss: ${validationLoss.toPrecision(5)}`;
  document.getElementById("test-button").removeAttribute("disabled");
  document.getElementById("save-button").removeAttribute("disabled");
  document.getElementById("predict-button").removeAttribute("disabled");
}

async function plotParams(weight, bias) {
  model.getLayer(null, 0).setWeights([
    tf.tensor2d([[weight]]), // Kernel (input multiplier)
    tf.tensor1d([bias]), // Bias
  ]);
  await plotPredictionLine();
  const layer = model.getLayer(undefined, 0);
  tfvis.show.layer({ name: "Layer 1" }, layer);
}

async function toggleVisor() {
  tfvis.visor().toggle();
}

let points;
let normalisedFeature, normalisedLabel;
let trainingFeatureTensor,
  testingFeatureTensor,
  trainingLabelTensor,
  testingLabelTensor;
async function run() {
  // Ensure backend has initialized
  await tf.ready();

  // Import from CSV
  const houseSalesDataset = tf.data.csv(
    "http://127.0.0.1:5500/linear-regression/kc_house_data.csv"
  );

  // Extract x and y values to plot
  const pointsDataset = houseSalesDataset.map((record) => ({
    x: record.sqft_living,
    y: record.price,
  }));
  points = await pointsDataset.toArray();
  if (points.length % 2 !== 0) {
    // If odd number of elements
    points.pop(); // remove one element
  }
  tf.util.shuffle(points);
  plot(points, "Square feet");

  // Extract Features (inputs)
  const featureValues = points.map((p) => p.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

  // Extract Labels (outputs)
  const labelValues = points.map((p) => p.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  // Normalise features and labels
  normalisedFeature = normalise(featureTensor);
  normalisedLabel = normalise(labelTensor);
  featureTensor.dispose();
  labelTensor.dispose();

  [trainingFeatureTensor, testingFeatureTensor] = tf.split(
    normalisedFeature.tensor,
    2
  );
  [trainingLabelTensor, testingLabelTensor] = tf.split(
    normalisedLabel.tensor,
    2
  );

  // Update status and enable train button
  document.getElementById("model-status").innerHTML = "No model trained";
  document.getElementById("train-button").removeAttribute("disabled");
  document.getElementById("load-button").removeAttribute("disabled");
}

run();
