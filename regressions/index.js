require('@tensorflow/tfjs')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const linearRegression = require('./linear-regression')

let { features, labels, testFeatures, testLabels } = loadCSV('cars.csv', {
    
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
})

const regression = new linearRegression(features, labels, { 

    learningRate: 0.00001,
    iteration: 1000
})


regression.train()

const r2 = regression.test(testFeatures, testLabels)

console.log('r2: ', r2)

