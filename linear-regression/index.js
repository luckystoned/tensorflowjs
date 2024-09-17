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

  trainingFeatureTensor.print(true);
}

run();
