import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
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

    destinyTree = DecisionTreeClassifier(random_state=5)

    model = destinyTree.fit(trainX, trainY.ravel())
    print (destinyTree.score(testX, testY))
