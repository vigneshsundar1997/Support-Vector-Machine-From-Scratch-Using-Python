import pandas as pd
import numpy as np  

data=pd.read_csv("dating-full.csv",nrows=6500)

columnsToStrip = ["race", "race_o", "field"]

#columns for which one hot encoding is done
columns = ['gender','race','race_o','field']
numberOfCellsStripped=0
temp_data = data[columnsToStrip].copy()

#the dictionary for printing the required one hot encoded labels
printValue = {"gender" : "female",
              "race" : "Black/African American",
              "race_o" : "Other",
              "field" : "economics"
              }

encodedDictionary={}

#1.(i) starts here
for stripColumn in columnsToStrip:
    data[stripColumn] = data[stripColumn].str.strip("'")


#1.(ii) starts here
data['field'] = data['field'].str.lower()

#1.(iv) starts here
preference_scores_of_participant = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive','pref_o_sincere', 'pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']

temp_data['total'] = 0

for column in preference_scores_of_participant:
    temp_data['total'] += data[column] 

for column in preference_scores_of_participant:
    data[column] = data[column] / temp_data['total']

temp_data['total'] = 0

for column in preference_scores_of_partner:
    temp_data['total'] += data[column] 

for column in preference_scores_of_partner:
    data[column] = data[column] / temp_data['total']


for column in columns:
    column_unique_set = set(data[column].value_counts().keys())
    column_dictionary = {}
    index=0

    for key in sorted(column_unique_set):
        column_dictionary[key]=index
        index=index+1

    encodedVectorDictonary = {}

    for key,value in column_dictionary.items():
        encodedVector = [0 for _ in range(len(column_dictionary)-1)]
        if value < len(column_dictionary)-1:
            encodedVector[value]=1
        
        encodedVectorDictonary[key]=encodedVector
    
    encodedDictionary[column] = encodedVectorDictonary


for key,value in printValue.items():
    print('Mapped vector for' , value, 'in column ' + key + ':' , ''.join(str(encodedDictionary[key][value]).split(',')) + '.')


one_hot_gender = pd.get_dummies(data['gender'],prefix='gender',columns=['gender'])
one_hot_race = pd.get_dummies(data['race'],prefix='race',columns=['race'])
one_hot_race_o = pd.get_dummies(data['race_o'],prefix='race_o',columns=['race_o'])
one_hot_field = pd.get_dummies(data['field'],prefix='field',columns=['field'])

one_hot_gender = one_hot_gender.iloc[:, :-1]
one_hot_race = one_hot_race.iloc[:, :-1]
one_hot_race_o = one_hot_race_o.iloc[:, :-1]
one_hot_field = one_hot_field.iloc[:, :-1]

data = data.drop(columns=['gender', 'race', 'race_o' ,'field'])

data = pd.concat([data,one_hot_gender],axis=1)
data = pd.concat([data,one_hot_race],axis=1)
data = pd.concat([data,one_hot_race_o],axis=1)
data = pd.concat([data,one_hot_field],axis=1)

#Sampling the data into required fractions and random state
test=data.sample(frac=0.2,random_state=25)
train=data.drop(test.index)

train.to_csv('trainingSet.csv',index=False)
test.to_csv('testSet.csv',index=False)