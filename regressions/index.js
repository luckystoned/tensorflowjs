require('@tensorflow/tfjs')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const linearRegression = require('./linear-regression')

let { features, labels, testFeatures, testLabels } = loadCSV('cars.csv', {
    
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
})

const regression = new linearRegression(features, labels, { 

    learningRate: 0.0001,
    iteration: 100 
})


regression.train()

console.log('m: ', regression.m, 'b: ', regression.b)