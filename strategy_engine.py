import numpy as np
import pandas as pd
from degradation_model import fitDegredationModels
from data_loader import concatenateTable
import itertools

def simulateStrategy(strategyDict, modelsDict, pitTime):
    pitlaps = strategyDict['pit_laps']
    totallaps = strategyDict['total_laps']
    stint_starts = [1]
    stint_ends = []
    for pitlap in pitlaps:
        stint_starts.append(pitlap + 1)
        stint_ends.append(pitlap)
    stint_ends.append(totallaps)
    numstints = len(stint_starts)
    stint_lengths = []
    for i in range(numstints):
        stint_lengths.append(stint_ends[i] - stint_starts[i] + 1)
    stintTyreLives = []
    for stintlen in stint_lengths:
        stintTyreLives.append(np.arange(1, stintlen + 1))
    comps = strategyDict['compounds']
    totalraceTime = 0
    for compound, stintTyreLife in zip(comps, stintTyreLives):
        comp_model = modelsDict[compound]
        p = comp_model[0]
        inter = stintTyreLife.reshape(-1, 1)
        X_p = p.transform(inter)
        m = comp_model[1]
        predictions = m.predict(X_p)
        totalraceTime = totalraceTime + np.sum(predictions)
    totalraceTime = totalraceTime + (len(pitlaps) * pitTime)
    return totalraceTime

def bruteForceSearch(modelsDict, totalLaps, minStint, pitDelta, numStops):
    loadTable = []
    pitlaps = []
    possible_laps = range(minStint, totalLaps - minStint + 1)
    all_combos = itertools.combinations(possible_laps, numStops)
    for combo in all_combos:
        boundaries = [1] + list(combo) + [totalLaps]
        stintlengths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        if all(length >= minStint for length in stintlengths):
            pitlaps.append(list(combo))
    old_comp_keys = modelsDict.keys()
    new_comp_keys = [key for key in old_comp_keys if '2022' not in key]
    comp_permutations = itertools.permutations(new_comp_keys, numStops+1)
    for comp_perm in comp_permutations:
        for pitlap in pitlaps:
            stratDict = {
                'pit_laps': pitlap,
                'total_laps': totalLaps,
                'compounds': comp_perm
            }
            loadTable.append([comp_perm, pitlap, totalLaps, simulateStrategy(stratDict, modelsDict, pitDelta)])
    outputTable = pd.DataFrame(loadTable, columns = ['Compounds', 'Pit Lap', 'Total Laps', 'Total Race Time']).sort_values('Total Race Time')
    return outputTable  

def allStrategies(modelsDict, totalLaps, minStint, pitDelta):
    oneStopStrategy = bruteForceSearch(modelsDict, totalLaps, minStint, pitDelta, 1)
    twoStopStrategy = bruteForceSearch(modelsDict, totalLaps, minStint, pitDelta, 2)
    allstratTable = pd.concat([oneStopStrategy, twoStopStrategy], ignore_index=True)
    return allstratTable

def stintDetail(optimalStrat, modelsDict):
    loadTable = []
    compounds = optimalStrat['Compounds']
    totalLaps = optimalStrat['Total Laps']
    pitLaps = optimalStrat['Pit Lap']
    stintStarts = [1]
    stintEnds = []
    for pitLap in pitLaps:
        stintStarts.append(pitLap + 1)
        stintEnds.append(pitLap)
    stintEnds.append(totalLaps)
    for i in range(1, totalLaps+1):
        Lap = i
        for stintStart, stintEnd, compound in zip(stintStarts, stintEnds, compounds):
            if i >= stintStart and i <= stintEnd:
                Compound = compound
                TyreLife = (i+1) - stintStart
                compound_model = modelsDict[Compound]
                p = compound_model[0]
                inter = np.array(TyreLife).reshape(-1, 1)
                X_p = p.transform(inter)
                m = compound_model[1]
                predictedLapTime = m.predict(X_p)[0]
                if Lap == 1:
                    degradationRate = 0
                else:
                    degradationRate = predictedLapTime - previousLapTime
                previousLapTime = predictedLapTime
        loadTable.append([Lap, Compound, TyreLife, predictedLapTime, degradationRate])
    outputTable = pd.DataFrame(loadTable, columns = ['Lap', 'Compound', 'TyreLife', 'Predicted Lap Time', 'Degradation Rate'])
    return outputTable

trYears = [2022, 2023]
teYears = [2025]
resultTable1 = concatenateTable(trYears, teYears, 'Bahrain', 'R')
trainTable = resultTable1[0]
trainDict = fitDegredationModels(trainTable)

stratTable = allStrategies(trainDict, 57, 15, 20)
optimalStrategy = stratTable.loc[0]
optimalStratDetails = stintDetail(optimalStrategy, trainDict)
print(optimalStratDetails)