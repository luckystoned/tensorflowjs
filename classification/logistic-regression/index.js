require('@tensorflow/tfjs')
const tf = require('@tensorflow/tfjs')
const _ = require('lodash')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./logistic-regression')
const plot = require('node-remote-plot')

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {

    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: value => value === 'TRUE' ? 1 : 0
    }
})

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.1,
    iterations: 100,
    batchSize: 50,
    decisionsBoundary: 0.5
})

regression.train()

console.log(regression.test(testFeatures, testLabels))


plot({
    x: regression.costHistory.reverse()
})