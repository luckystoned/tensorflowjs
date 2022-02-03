const outputs = [];


function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  // Write code here to analyze stuff
  const testSetSize = 10;

  const [trainingSet, testSet] = splitDataset(outputs, testSetSize);

  // console.log('Data:', outputs);

  // console.log("trainingSet" ,trainingSet);
  // console.log("testSet" ,testSet);

  // let numberCorrect = 0;
  // for (let i = 0; i < testSet.length; i++) {
  //   const bucket = knn(trainingSet, testSet[i][0]);

  //   if (bucket === testSet[i][3]) {
  //     numberCorrect ++
  //   }
  // }

  _.range(1, 15).forEach(k => {
    const accurancy = _.chain(testSet)
      .filter(testPoint => knn(trainingSet, testPoint[0]) === testPoint[3], k)
      .size()
      .divide(testSetSize)
      .value();

    console.log(`For k of ${k} Accuracy: ${accurancy}%`);
  });

}

function knn (data, point, k) {
  return _.chain(data)
    .map(r => [distance(r[0], point), r[3]])
    .sortBy(r => r[0])
    .slice(0, k)
    .countBy(r => r[1])
    .toPairs()
    .sortBy(r => r[1])
    .last()
    .first()
    .parseInt()
    .value()
}

function distance(pointA, pointB) {
  return Math.abs(pointA - pointB)
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount);
  const trainigSet = _.slice(shuffled, testCount);

  return [trainigSet, testSet];
}