import numpy as np
from sklearn.naive_bayes import MultinomialNB
from preprocess import Preprocess



if __name__ == '__main__':

    trainProcessor = Preprocess('income-training.csv', 'income-test.csv', 'output.csv')
    data = trainProcessor.sanitize()

    trainData = trainProcessor.process(data[0])

    testData = data[1]


    trainY = trainData.iloc[:, 13:14].values
    trainX = trainData.iloc[:, 0:13].values

    testY = testData.iloc[:, 13:14].values
    testX = testData.iloc[:, 0:13].values

    michaelBayes = MultinomialNB()
    model = michaelBayes.fit(trainX, trainY.ravel())
    print (michaelBayes.score(testX, testY))
