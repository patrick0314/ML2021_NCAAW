from ReadData import *
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

Team = Team() # [369, 2] = [TeamID, TeamName]
n = len(Team['TeamID'])
VS = np.zeros((n, n)) # [369, 369], [i, j] record the winning probability for team i playing with team j

CompactResult = CompactResult() # [1386, 8] = [Season, DayNum, WTeamID, LTeamID, LScore, WLoc, NumOT]
for i in range(len(CompactResult['Season'])):
    W = Team.index[Team['TeamID']==CompactResult['WTeamID'][i]].to_list()
    L = Team.index[Team['TeamID']==CompactResult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1

RegularCompactResult = RegularCompactResult() # [112183, 8] = [Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT]
for i in range(len(RegularCompactResult['Season'])):
    W = Team.index[Team['TeamID']==RegularCompactResult['WTeamID'][i]].to_list()
    L = Team.index[Team['TeamID']==RegularCompactResult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1

for i in range(VS.shape[0]):
    for j in range(VS.shape[1]):
        tmp = VS[i, j] + VS[j, i]
        if tmp != 0:
            VS[i, j] /= tmp

Output = Output()
for i in range(len(Output['ID'])):
    Output['ID'][i] = Output['ID'][i].split('_')

for i in range(len(Output['ID'])):
    W = Team.index[Team['TeamID']==int(Output['ID'][i][1])].to_list()
    L = Team.index[Team['TeamID']==int(Output['ID'][i][2])].to_list()
    Output['Pred'][i] = VS[W[0], L[0]]

Output.to_csv('output.csv')

