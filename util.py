#!/usr/bin/python -u
import sys, os, argparse, time, traceback 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pickle import load, dump

####################    SUPPORT MODEL and KERNEL    ####################
models = ('SVM', 'LINEARSVM', 'RF', 'KNN', 'LR', 'ADABOOST', 'GNB', 'LDA', 'GB', 'RIDGE', 'SGD')
kernels = ('linear', 'poly', 'rbf', 'sigmoid')
base_estimators = ('DT', 'SVM', 'SGD')
########################################################################

def load_parser():
    
    # Set up parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-i'    , dest='train_filename', required=True  , help='Specify the input train file')
    parser.add_argument('-m'    , dest='model'         , required=True  , help='Specify the model name (choices: ' + ', '.join(models) + ')')
    parser.add_argument('-n'    , dest='normalized'    , type=int       , help='Normalization method [1: [-1, 1], 2:[0, 1], 3: standard normalization]')
    parser.add_argument('-k'    , dest='kernel'        , default="rbf"  , help='kernel in SVM (choices: ' + ', '.join(kernels) + ') [default = rbf]')
    parser.add_argument('-c'    , dest='C'                              , help='C in SVM, linearSVM and LR(Logistic Regression) [default = grid search]')
    parser.add_argument('-g'    , dest='gamma'                          , help='gamma in SVM [default = grid search]')
    parser.add_argument('-r'    , dest='coef0'                          , help='coef0 in poly and sigmoid kernel SVM [default = grid search]')
    parser.add_argument('-d'    , dest='degree'                         , help='degree in polynomial kernel SVM and KNN distance metric [default = grid search]')
    parser.add_argument('-ne'   , dest='n_estimators'                   , help='number of tree/estimators in RF(RandomForest) and Adaboost [default = grid search]')
    parser.add_argument('-nn'   , dest='n_neighbors'                    , help='number of neighbors in Adaboost and KNN [default = grid search]')
    parser.add_argument('-lr'   , dest='learning_rate'                  , help='learning rate for SGD and AdaBoost [default = grid search]')
    parser.add_argument('-be'   , dest='base_estimator', default="DT"   , help='Base estimator in Adaboost [default = Decision Tree]')
    parser.add_argument('-p'    , dest='penalty'                        , help='penalty function used in linearSVC and logistic regression (choice: l2, l1) [default = grid search]')
    parser.add_argument('-l'    , dest='loss'                           , help='loss function used in linearSVC (choice: l2, l1) [default = grid search]')
    parser.add_argument('-a'    , dest='alpha'                          , help='alpha in SGD [default = grid search]')
    parser.add_argument('-w'    , dest='weights'                        , help='weight function used in KNN (choice: uniform, distance) [default = distance]')
    parser.add_argument('-D'    , dest='dim'                            , help='First D dimension to use [default = all]')
    
    return parser


def check_options(opts):

    # check model
    if( opts.model not in models ):
        print "Unknown model name %s !" %opts.model
        print "Support model: " + ', '.join(models)
        traceback.print_stack()
        sys.exit(1)

    # check kernel
    if( opts.kernel != None ):
        if( opts.kernel not in kernels ):
            print "Unknown kernel name %s !" %opts.kernel
            print "Support kernel: " + ', '.join(kernels)
            traceback.print_stack()
            sys.exit(1)

    # check base estimator in adaboost
    if( opts.base_estimator != None ):
        if( opts.base_estimator not in base_estimators ):
            print "Unkinown base estimator %s !" %opts.base_estimator
            print "Support base estimator: " + ", ".join(base_estimators)
            traceback.print_stack()
            sys.exit(1)


def load_scaler(scaler_fileName, X, method=1):
    
    try:
        # load scaler if existed
        with open(scaler_fileName, 'r') as iFile:
            print "Loading " + scaler_fileName + " ..."
            scaler = load(iFile)
    except:
        # build new scaler
        print "Scalar file is not exist, compute scaler and dump %s ..." %scaler_fileName
        if( method == 1 ):
            print "Normalizeing to [-1, 1] ..."
            scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
        elif( method == 2 ):
            print "Normalizeing to [0, 1] ..."
            scaler = MinMaxScaler().fit(X)
        elif( method == 3 ):
            print "Standard Normalizing ..."
            scaler = StandardScaler().fit(X)
        else:
            print "Error! Unknown normalization method (%d)!" %method
            print "Choice: 1 for [-1, 1], 2 for [0, 1], 3 for standard normalization"
            traceback.print_stack()
            sys.exit(1)

        with open(scaler_fileName, 'w') as oFile:
            dump(scaler, oFile)
    
    return scaler


def print_time(ts, te):

    if( te < ts ):
        # swap if inputs are inverted
        ts, te = te, ts

    t = te - ts
    
    second = t % 60
    minute = int(t/60)
    if( minute > 60 ):
        hour = round(minute/60)
        minute = minute % 60
    else:
        hour = 0

    if( hour > 24 ):
        day = round(hour/24)
        hour = hour % 24
    else:
        day = 0

    print "Elapsed time is %f secs (%d Day, %d Hour, %d Min, %f Sec)" %(t, day, hour, minute, second)

    return t

