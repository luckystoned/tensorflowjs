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
async function run() {
  const housesSalesDataset = tf.data.csv(
    "http://127.0.0.1:5500/linear-regression/kc_house_data.csv"
  );
  const sampleDataset = housesSalesDataset.take(10);
  const dataAsArray = await sampleDataset.toArray();
  console.log(dataAsArray);

  const points = housesSalesDataset.map((record) => ({
    x: record.sqft_living,
    y: record.price,
  }));

  plot(await points.toArray(), "Saquare Feet");
}

run();
