# Support-Vector-Machine-From-Scratch-Using-Python
Implementation of Support Vector Machine from Scratch using Python on the Dating Dataset and the analysis of the hyper parameters.

The project folder contains 3 python files: 
1. preprocess.py
2. svm.py
3. cv.py

################

1. preprocess.py

This script contains the preprocessing steps like removing the quotes, converting to lower case, normalization, one-hot encoding and split the dataset. It makes use of dating-full.csv as the input. It outputs trainingSet.csv and testSet.csv.

Execution : python3 preprocess.py

2. svm.py

This script contains the training and testing of the models for Support Vector Machines. It takes in two arguments, the training file name, the test file.

Execution : python3 lr_svm.py trainingFileName testFileName

eg: 
Run command for SVM model
python3 lr_svm.py trainingSet.csv testSet.csv

3. cv.py

This script performs the ten fold validation for the model SVM. It also outputs a graph indicating the test accuracies of the model and their standard errors for different trainingSet sizes.

Execution : python3 cv.py
