sklearn_API
===========

An integrated machine learning API for [scikit-learn](http://scikit-learn.org/stable/).

Support function:
* train_valid.py: training + N-fold validation (multi-thread)
* train_pred.py: training + prediction
* build_data.py: generate svm-format data from data list
* merge_svmfeature.py: merge svm-format features

Support learning model(abbreviation):
* SVC(svm)
* Linear SVC(linearsvm)
* Linear model with Stochastic Gradient Descent(SGD)
* Logistic Regression(LR)
* Ridge Regression(ridge)
* Random Forest(RF)
* AdaBoost(adaboost)
* Gradient Boosting(GB)
* K-Nearest neighbors(KNN)
* Gaussian Naive Bayesian(GNB)
* Linear Discriminant Analysis(LDA)


Usage: 
-----
(Specific options for each model are described later)

Train by SVM with 5-fold cross validation and 8-thread processing:
```
  ./train_valid.py -i TRAIN_FILE -m svm -f 5 -th 8
```
Train and predict by SVM and output predicted label:
```
  ./train_pred.py -i TRAIN_FILE -t TEST_FILE -m svm -o PREDICT_FILE
```
Generate svm-format feature from list:
```
  ./build_data.py -l LIST -i INPUT_DIR -e EXT -o OUTPUT_FILE
```
Merge 3 svm-format features:
```
  ./merge_svmfeature.py -i FILE1 FILE2 FILE3 -o OUTPUT_FILE
```
Sample 100 training data and 50 testing data from a list:
```
  ./sample_list.py -i INPUT_LIST -n 100 500 -o TRAIN_LIST TEST_LIST
```

