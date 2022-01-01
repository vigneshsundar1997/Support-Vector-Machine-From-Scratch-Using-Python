import pandas as pd
import numpy as np
from svm import svm
from statistics import mean,stdev
import math
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

#sample the data based on the given random_state and t_frac
def sampleData(data,random_state_value,t_frac):
    return data.sample(frac=t_frac,random_state=random_state_value)

#read the trainingSet for lr and svm
trainDataSVM = pd.read_csv('trainingSet.csv')

#sample the training sets
trainDataSVM = sampleData(trainDataSVM,18,1)

#split the data into 10 sets of equal sizes
SVM_data_sets = np.array_split(trainDataSVM,10)

t_fracs = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]

svm_test_accuracy = []

svm_test_accuracy_list = []

svm_stdev_list = []

for t_frac in t_fracs:
    for index in range(10):
  
        #copy the dataframes to get the sIndex and sC for each of the ten sets
        temp_LRSVM=SVM_data_sets.copy()

        #sIndex is the set of the values from training beloning to index
        setIndexLRSVM=temp_LRSVM[index]
        del temp_LRSVM[index]

        #sC, the remaining values other than sIndex
        setCLRSVM = pd.concat(temp_LRSVM)

        #saample the data with random state 32 and the given t_frac
        train_set_LRSVM = sampleData(setCLRSVM,32,t_frac)

        #train svm and get the accuracy for test
        trainingAccuracy,testAccuracy = svm(train_set_LRSVM,setIndexLRSVM)
        svm_test_accuracy.append(testAccuracy)


    #calculate the mean, standard error of the 10 sets for each of the t_frac for svm
    svm_test_accuracy_list.append(mean(svm_test_accuracy))
    svm_stdev_list.append(stdev(svm_test_accuracy)/math.sqrt(10))
  
    svm_test_accuracy.clear()
  
#get the training set size for each of the fraction to be plotted in x-axis of the graph
plot_x_axis = [t_frac * setCLRSVM.shape[0] for t_frac in t_fracs]

plt.errorbar( plot_x_axis, svm_test_accuracy_list, yerr= svm_stdev_list ,label='SVM')

plt.xlabel('Training Dataset Size')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.title('Test Accuracy of LR,SVM and NBC and their standard errors')

plt.show()