from data_loader import concatenateTable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fitDegredationModels(concatenatedTable):
    outputDict = {}
    compounds = concatenatedTable['CompoundLabel'].unique()
    for compound in compounds:
        compound_data = concatenatedTable[concatenatedTable['CompoundLabel'] == compound]
        X = compound_data['TyreLife'].values.reshape(-1, 1)
        y = compound_data['FuelCorrectedLapTime'].values
        poly = PolynomialFeatures(degree=2)
        model = LinearRegression()

        X_poly = poly.fit_transform(X)

        outputDict[compound] = (poly, model.fit(X_poly, y))

    return outputDict

def evaluateModels(trainingDictionary, testingTable):
    compounds = trainingDictionary.keys()
    for compound in compounds:
        compound_data = testingTable[testingTable['CompoundLabel'] == compound]
        if compound_data.empty:
            print("The " + compound + " tyres were not used!")
            continue
        else:
            compound_model = trainingDictionary[compound]
            p = compound_model[0]
            inter = compound_data['TyreLife'].values.reshape(-1, 1)
            X_p = p.transform(inter)
            m = compound_model[1]
            prediction = m.predict(X_p)
            true_values = compound_data['FuelCorrectedLapTime'].values
            mae = mean_absolute_error(true_values, prediction)
            mse = mean_squared_error(true_values, prediction)
            print(mae)
            print(mse)

trYears = [2022, 2023]
teYears = [2025]
resultTable1 = concatenateTable(trYears, teYears, 'Bahrain', 'R')
trainTable = resultTable1[0]
testTable = resultTable1[1]

trainDict = fitDegredationModels(trainTable)

evaluateModels(trainDict, testTable)