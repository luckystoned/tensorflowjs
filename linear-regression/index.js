async function plot(pointsArray, featureName) {
  tfvis.render.scatterplot(
    { name: `${featureName} vs House Price` },
    { values: [pointsArray], series: ["original"] },
    {
      xLabel: featureName,
      yLabel: "Price",
    }
  );
}

function normalise(tensor) {
  const min = tensor.min();
  const max = tensor.max();
  const normalisedTensor = tensor.sub(min).div(max.sub(min));
  return {
    tensor: normalisedTensor,
    min: min,
    max: max,
  };
}

function denormalise(tensor, min, max) {
  const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
  return denormalisedTensor;
}

function createModel() {
  const model = tf.sequential();

  // Add a single hidden layer
  model.add(
    tf.layers.dense({
      units: 1,
      useBias: true,
      activation: "linear",
      inputDim: 1,
    })
  );

  // Add an output layer
  model.add(
    tf.layers.dense({
      units: 1,
      useBias: true,
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
  const batchSize = 32;
  const epochs = 20;

  const { onEpochEnd } = tfvis.show.fitCallbacks(
    { name: "Training Performance" },
    ["loss"],
    { height: 200, callbacks: ["onEpochEnd", "onBatchEnd"] }
  );

  return await model.fit(trainingFeatureTensor, trainingLabelTensor, {
    batchSize,
    epochs,
    validationSplit: 0.2,
    shuffle: true,
    callbacks: {
      onEpochEnd,
    },
  });
}

async function run() {
  // Import the dataset
  const housesSalesDataset = tf.data.csv(
    "http://127.0.0.1:5500/linear-regression/kc_house_data.csv"
  );

  // Extract the data
  const pointsDataset = housesSalesDataset.map((record) => ({
    x: record.sqft_living,
    y: record.price,
  }));
  const points = await pointsDataset.toArray();
  if (points.length % 2 !== 0) {
    // Remove the last element if the length is odd
    points.pop();
  }
  tf.util.shuffle(points);

  plot(await points, "Saquare Feet");

  // Extract Features (inputs)
  const featureValues = await points.map((p) => p.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

  // Extract Labels (outputs)
  const labelValues = await points.map((p) => p.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  // Normalise the data
  const normalisedFeature = normalise(featureTensor);
  const normalisedLabel = normalise(labelTensor);

  const [trainingFeatureTensor, testingFeatureTensor] = tf.split(
    normalisedFeature.tensor,
    2
  );
  const [trainingLabelTensor, testingLabelTensor] = tf.split(
    normalisedLabel.tensor,
    2
  );

  const model = createModel();
  //Model Summary
  tfvis.show.modelSummary({ name: "Model Summary" }, model);
  const layer = model.getLayer(undefined, 0);
  tfvis.show.layer({ name: "Layer 1" }, layer);

  const result = await trainModel(
    model,
    trainingFeatureTensor,
    trainingLabelTensor
  );
  const trainingLoss = result.history.loss.pop();
  console.log(`Training Loss: ${trainingLoss}`);
  const validationLoss = result.history.val_loss.pop();
  console.log(`Validation Loss: ${validationLoss}`);

  const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
  const loss = await lossTensor.dataSync();
  console.log(`Testing Loss: ${loss}`);
}

run();
