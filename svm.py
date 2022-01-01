from tkinter.filedialog import test
import pandas as pd
import numpy as np
from sys import argv
import warnings
warnings.filterwarnings("ignore")

#Split the input data into features set and decision set
def split_features_outcome(data):
    features = data.drop(['decision'],axis=1)
    decision = data['decision']
    return features,decision

#svm model which takes trainingSet and testSet. Trains the model, calculates the accuracy of the trainingSet and the testSet and returns them.
def svm(trainingSet,testSet):
    #split the training dataset
    features,decision = split_features_outcome(trainingSet)
    #replace decision 0 with -1 as svm works on -1 and 1
    decision=decision.replace(0,-1)

    lambdaValue=0.01
    stepSize=0.5

    shape = (1,features.shape[1]+1)

    #create an initial weight vector with zeros of size (1,261) i.e. (1,number of features+1)
    weight = np.zeros(shape)

    features_np_array = features.to_numpy()
    columns_shape = (features.shape[0],1)

    #add a column of ones to the feature array for first parameter in the weight vector
    feature_array = np.concatenate((np.ones(columns_shape),features_np_array),axis=1)
    decision_array = decision.to_numpy().reshape(-1,1)

    #iterate for 500 times
    for i in range(500):

        #initial y_hat by multiplying feature array and weight vector
        y_hat = np.dot(feature_array,weight.T)


        yDotYHat = decision_array * y_hat
        
        #this step is to help calculate deltaJI as 0 if y*y_hat is greater than or equal to zero
        temp_decision_array=np.where(yDotYHat>=1,0,decision_array)
        
        #calculate deltaJI as feature * decision if y*y_hat < 1
        deltaJI = np.dot(temp_decision_array.T,feature_array)

        #calculate deltaJ as lambda=0.01*weight - deltaJI. Divide the answer by the number of samples.
        deltaJ = (feature_array.shape[0]*lambdaValue*weight - deltaJI)/feature_array.shape[0]

        #calculate the new weight as old weight - step_size=0.05*deltaJ
        new_weight = weight - stepSize*deltaJ

        #if the new weight - weight is less than 1e-6, then we break the iterations and the new weight will be the learned weight.
        if(np.linalg.norm(new_weight-weight) < 1e-6):
            break
        else:
            weight=new_weight

    #make predictions using the new weight vector
    y_predict = np.dot(feature_array,new_weight.T)

    #identity the sign of each value to find if the label is -1 or 1
    y_predict = np.sign(y_predict)

    #calculate the accuracy
    training_accuracy = round(float(sum(decision_array==y_predict))/float(len(decision_array)),2)

    #split the features for test set
    features,decision = split_features_outcome(testSet)
    decision=decision.replace(0,-1)
    shape = (1,features.shape[1]+1)
    weight = np.zeros(shape)

    features_np_array = features.to_numpy()
    columns_shape = (features.shape[0],1)
    feature_array = np.concatenate((np.ones(columns_shape),features_np_array),axis=1)
    decision_array = decision.to_numpy().reshape(-1,1)

    #predict the outcomes using the learned weight from training
    y_predict = np.dot(feature_array,new_weight.T)

    y_predict = np.sign(y_predict)

    test_accuracy = round(float(sum(decision_array==y_predict))/float(len(decision_array)),2)

    return training_accuracy,test_accuracy

if __name__ == "__main__":
    trainingDataFileName = argv[1]
    testDataFileName = argv[2]
    
    data_train=pd.read_csv(trainingDataFileName)
    data_test=pd.read_csv(testDataFileName)
    training_accuracy,test_accuracy=svm(data_train,data_test)
    print('Training Accuracy SVM:', training_accuracy)
    print('Testing Accuracy LR:' ,test_accuracy)