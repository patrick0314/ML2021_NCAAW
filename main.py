from ReadData import *
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

team = Team() # [369, 2] = [TeamID, TeamName]
n = len(team['TeamID'])
VS = np.zeros((n, n)) # [369, 369], [i, j] record the winning probability for team i playing with team j

compactresult = CompactResult() # [1386, 8] = [Season, DayNum, WteamID, LteamID, LScore, WLoc, NumOT]
for i in range(len(compactresult['Season'])):
    W = team.index[team['TeamID']==compactresult['WTeamID'][i]].to_list()
    L = team.index[team['TeamID']==compactresult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1

regularcompactresult = RegularCompactResult() # [112183, 8] = [Season, DayNum, WteamID, WScore, LteamID, LScore, WLoc, NumOT]
for i in range(len(regularcompactresult['Season'])):
    W = team.index[team['TeamID']==regularcompactresult['WTeamID'][i]].to_list()
    L = team.index[team['TeamID']==regularcompactresult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1

for i in range(VS.shape[0]):
    for j in range(VS.shape[1]):
        tmp = VS[i, j] + VS[j, i]
        if tmp != 0:
            VS[i, j] /= tmp

output = Output()
for i in range(len(output['ID'])):
    output['ID'][i] = output['ID'][i].split('_')

for i in range(len(output['ID'])):
    W = team.index[team['TeamID']==int(output['ID'][i][1])].to_list()
    L = team.index[team['TeamID']==int(output['ID'][i][2])].to_list()
    output['Pred'][i] = VS[W[0], L[0]]

output1 = Output()
output1['Pred'] = output['Pred']
output1.to_csv('output.csv', index=0)
