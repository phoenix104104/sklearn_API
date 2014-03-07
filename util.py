#!/usr/bin/python -u
import sys, os, argparse, time, traceback 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pickle import load, dump

models = ('SVM', 'LINEARSVM', 'RF', 'KNN', 'LR', 'ADABOOST', 'GNB', 'LDA', 'GB', 'RIDGE', 'SGD')
kernels = ('linear', 'poly', 'rbf', 'sigmoid')

def LoadParser():
    
    # Set up parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-i'    , dest='train_filename', required=True  , help='Specify the input train file')
    parser.add_argument('-m'    , dest='model'         , required=True  , help='Specify the model name (choices: ' + ', '.join(models) + ')')
    parser.add_argument('-n'    , dest='normalized'    , type=int       , help='Normalization method [1: [-1, 1], 2:[0, 1], 3: standard normalization]')
    parser.add_argument('-k'    , dest='kernel'        , default="rbf"  , help='kernel for SVC (choices: ' + ', '.join(kernels) + ') [default = rbf]')
    parser.add_argument('-c'    , dest='C'             , type=float     , help='C for SVC, linearSVC and logistic regression [default = grid search]')
    parser.add_argument('-g'    , dest='gamma'         , type=float     , help='gamma for rbf, poly, and sigmoid kernel SVC [default = grid search]')
    parser.add_argument('-r'    , dest='coef0'         , type=float     , help='coef0 for poly and sigmoid kernel SVC; reg for QDA [default = grid search]')
    parser.add_argument('-d'    , dest='degree'                         , help='degree for polynomial kernel SVC and KNN distance metric [default = grid search]')
    parser.add_argument('-ne'   , dest='n_estimators'                   , help='number of tree/estimators in RandomForest/Adaboost [default = grid search]')
    parser.add_argument('-nn'   , dest='n_neighbors'                    , help='number of neighbors in KNN [default = grid search]')
    parser.add_argument('-lr'   , dest='learning_rate' , type=float     , help='learning rate for AdaBoost/NeuralNetwork [default = grid search]')
    parser.add_argument('-be'   , dest='base_estimator'                 , help='Base estimator in Adaboost [default = Decision Tree]')
    parser.add_argument('-p'    , dest='penalty'                        , help='penalty function used in linearSVC and logistic regression (choice: l2, l1) [default = grid search]')
    parser.add_argument('-l'    , dest='loss'                           , help='loss function used in linearSVC (choice: l2, l1) [default = grid search]')
    parser.add_argument('-a'    , dest='alpha'                          , help='algorithm used in AdaBoost/KNN (choice: SAMME, SAMME.R for AdaBoost; audo, ball_tree, kd_tree for KNN) [default = SAMME.R for AdaBoost; auto for KNN]')
    parser.add_argument('-w'    , dest='weights'                        , help='weight function used in KNN (choice: uniform, distance) [default = distance]')
    parser.add_argument('-D'    , dest='dim'                            , help='First D dimension to use [default = all]')
    
    return parser

def CheckOpts(opts):

    # check model
    if( opts.model not in models ):
        print "Unknown model name %s !" %opts.model
        print "Support model: " + ', '.join(models)
        traceback.print_exc()
        sys.exit(1)

    # check kernel
    if( opts.kernel != None ):
        if( opts.kernel not in kernels ):
            print "Unknown kernel name %s !" %opts.kernel
            print "Support kernel: " + ', '.join(kernels)
            traceback.print_exc()
            sys.exit(1)


def LoadScaler(scaler_fileName, X, method=1):
    try:
        with open(scaler_fileName, 'r') as iFile:
            print "Loading " + scaler_fileName + " ..."
            scaler = load(iFile)
    except:
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
            traceback.exc()
            sys.exit(1)

        with open(scaler_fileName, 'w') as oFile:
            dump(scaler, oFile)
    
    return scaler


def BuildParameterGrid( argListDict, arg={} ):

    if len(argListDict) == 0 :
        return [arg]
    else:
        
        keys = argListDict.keys()
        argList = argListDict[ keys[0] ]
        
        grid = []

        smallDict = argListDict.copy()
        del smallDict[ keys[0] ]

        for param in argList:
            arg[ keys[0] ] = param
            arg_next = arg.copy()
            grid += buildParameterGrid(smallDict, arg_next)

        return grid
