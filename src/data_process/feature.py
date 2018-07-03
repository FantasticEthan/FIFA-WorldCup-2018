import pandas as pd
import sys
sys.path.append('../')
"""
Created on Tue June 26 12:09:29 2018

@author: ethan
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.labelEncoder import loadLabelEncoder,saveLabelEncoder

# Load Data
data_EUR_2012 = pd.read_csv("../../data/data_EUR_2012.csv", encoding='utf-8')
data_EUR_2016 = pd.read_csv("../../data/data_EUR_2016.csv", encoding='utf-8')

data_WC_2010 = pd.read_csv("../../data/data_WC_2010.csv", encoding='utf-8')
data_WC_2014 = pd.read_csv("../../data/data_WC_2014.csv", encoding='utf-8')

data_WC_2018 = pd.read_csv("../../data/data_WC_2018.csv", encoding='utf-8')

data_all = pd.concat([data_WC_2010, data_EUR_2012, data_WC_2014, data_EUR_2016])

data_all.drop(['home_team','host','match_type','abnormal'],axis=1,inplace=True)
# data_WC_2018.drop(['host'],axis=1,inplace=True)
data_all = pd.concat([data_all,data_WC_2018],sort=False)
data_all.drop(['id','date'],axis=1,inplace=True)
data_all.reset_index(inplace=True,drop=True)

#change 'win' and 'loss' to 'draw' if goal_diff is 0
data_all.loc[(data_all.goal_diff==0),'result'] = 'draw'

#change goal num
# over_three = lambda x : str(3) if abs(x)>=3 else str(int(abs(x)))
# data_['goal_diff'] = data_['goal_diff'].map(over_three)
# data_['result'] = data_['result']+data_['goal_diff']
# data_.loc[(data_.result =='draw1'),'result'] = 'draw0'

#-----create dictionary for result,tournament,country---------
le_result = saveLabelEncoder(data_all['result'],'../../data/LE/result.npy')
# Load Label Encoder and combine result,tourament,team_1,team_2,home_team
le_result = loadLabelEncoder('../../data/LE/result.npy')
data_all['result'] = le_result.transform(data_all['result'])
le_tour = loadLabelEncoder('../../data/LE/tournament.npy')
data_all['tournament'] = le_tour.transform(data_all['tournament'])
le_country = loadLabelEncoder('../../data/LE/country.npy')


data_all['team_1'] = le_country.transform(data_all['team_1'])
data_all['team_2'] = le_country.transform(data_all['team_2'])
# data_all['home_team'] = le_country.transform(data_all['home_team'])

# Add HOME team {home_tea:1,else:0}
# same_ht = (data_all.team_1 == data_all.home_team)
# data_all.loc[same_ht,'home_team'] = 1
# data_all.loc[-same_ht,'home_team'] = 0
print(data_all)
data_all.to_csv('../../result/process_data/train.csv',index=False)
