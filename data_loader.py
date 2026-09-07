import fastf1
import pandas as pd 
import numpy as np

fastf1.Cache.enable_cache('f1_cache')

def cleanSession(Year, Circuit, Session):
    session = fastf1.get_session(Year, Circuit, Session)
    session.load()

    laps = session.laps
    filtered_laps = laps[(laps.TrackStatus == '1') & (laps.IsAccurate == True) & (laps.PitOutTime.isnull()) & (laps.PitInTime.isnull())].copy()
    filtered_laps["LapTimeSeconds"] = filtered_laps.LapTime.dt.total_seconds()
    filtered_laps["FuelCorrectedLapTime"] = filtered_laps.LapTimeSeconds + (1.5 * 0.03 * filtered_laps.LapNumber)
    filtered_laps["Year"] = Year
    filtered_laps['CompoundLabel'] = np.where((filtered_laps['Compound'] == 'HARD') & (filtered_laps['Year'] == 2022), '2022 HARD', filtered_laps['Compound'])
    cleanTable = filtered_laps[['Year', 'Driver', 'LapNumber', 'FuelCorrectedLapTime', 'TyreLife', 'Compound', 'CompoundLabel', 'Stint']]
    return cleanTable

def concatenateTable(TrainYears, TestYears, Circuit, Session):
    trainTable = loadYears(TrainYears, Circuit, Session)
    testTable = loadYears(TestYears, Circuit, Session)
    return trainTable, testTable

def loadYears(Years, Circuit, Session):
    outputTable = []
    for i in Years:
        currentSession = cleanSession(i, Circuit, Session)
        outputTable.append(currentSession)
    loadTable = pd.concat(outputTable)
    return loadTable