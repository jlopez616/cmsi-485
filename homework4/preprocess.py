import csv
import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import *
from sklearn.compose import ColumnTransformer

class Preprocess():
    def __init__ (self, firstInput, secondInput, output):
        self.firstInput = firstInput
        self.secondInput = secondInput
        self.output = output
        self.headers = ['Age', 'Work_Class', 'Education', 'Education-num',
        'Marital_Status', 'Occupation_Code', 'Relationship', 'Race',
        'Sex', 'Capital Gain', 'Capital Loss', 'Hours_Per_Week', 'Native_Country', 'Income_Class']
        self.enc =  OrdinalEncoder()

    def process(self, df):
           age_bins = [0, 22, 65, np.inf]
           df['Age'] = pd.cut(df['Age'], age_bins, labels=[0, 1, 2], include_lowest = True).cat.codes
           return df


    def combine(self, document, removeMissing):
        with open(document, mode = 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            newRows = []
            for row in csv_reader:
                hasMissing = False
                for each in row:
                    value = each.replace(" ", "")
                    if (value == "?" and removeMissing):
                         hasMissing = True
                         break
                if (not hasMissing):
                    newRows.append(row)
            return newRows


    def sanitize(self):
        with open(self.output, mode = 'w') as new_file:
            csv_writer = csv.writer(new_file, delimiter=',')
            rowsA = self.combine(self.firstInput, False)

            splitPoint = len(rowsA) + 1
            for rows in rowsA:
                csv_writer.writerow(rows)

            rowsB = self.combine(self.secondInput, True)
            for rows in rowsB:
                csv_writer.writerow(rows)

        df = pd.read_csv(self.output, header = None, names = self.headers)

        for header in self.headers:
            if (not header == 'age'):
                df[header] = df[header].astype('category').cat.codes

        firstDf, secondDf = df.iloc[:splitPoint, :], df.iloc[splitPoint:, :]

        return (firstDf, secondDf)
