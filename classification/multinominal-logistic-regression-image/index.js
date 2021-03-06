require('@tensorflow/tfjs')
const tf = require('@tensorflow/tfjs')
const LogisticRegression = require('./logistic-regression')
const plot = require('node-remote-plot')
const _ = require('lodash')
const mnist = require('mnist-data')

function loadData() {
    const mnistData = mnist.training(0, 2000)

    const features = mnistData.images.values.map(imag => _.flatMap(imag))

    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0)
        row[label] = 1
        return row
    })

    return {features, labels: encodedLabels}
}

const {features, labels} = loadData()

const regression = new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 20,
    batchSize: 100
})

regression.train()

const testMnistData = mnist.testing(0, 100)
const testFeatures = testMnistData.images.values.map(imag => _.flatMap(imag))
const testEcodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0)
    row[label] = 1
    return row
})

const accuracy = regression.test(testFeatures, testEcodedLabels)
console.log("Accuracy: ", accuracy)

