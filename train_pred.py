#!/usr/bin/python -u

import sys, os, argparse, time, traceback 
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from pickle import dump, load
from util import load_parser, load_scaler, check_options, print_time
import numpy as np

def train(Classifier, arg, X, y, model_filename):
    
    clf = Classifier(**arg)
    clf.fit(X, y)

    if( model_filename != None ):
        with open(model_filename, 'w') as f:
            print "Saving model file %s ..." %model_filename
            dump(clf, f)

    return clf

def predict(clf, X, y, output_filename, prob_filename=None):

    y_pred = clf.predict(X)

    # calculate accuracy
    acc = float(sum(y == y_pred)) / len(y_pred)

    if( output_filename != None ):
        print "Saving predict file %s ..." %output_filename
        np.savetxt(output_filename, y_pred, '%d')
    
    if( prob_filename != None ):
        prob = clf.predict_proba(X)
        print "Saving probability file %s ..." %prob_filename
        np.savetxt(prob_filename, prob, '%.6e')
    
    return acc


def main():

    parser = load_parser()
    parser.add_argument('-t' , '--test'         , dest='test_filename' , required=True  , help='Specify the test file path')
    parser.add_argument('-o' , '--output'       , dest='output_filename'                , help='Specify the output predict file path [optional]')
    parser.add_argument('-om', '--output-model' , dest='model_filename'                 , help='Specify the output model file path [optional]')
    parser.add_argument('-op', '--output-prob'  , dest='prob_filename'                  , help='Specify the output probability file path [optional]')
    opts = parser.parse_args(sys.argv[1:]) 

    # pre-check options before loading data
    opts.model = opts.model.upper()
    opts.kernel = opts.kernel.lower()
    opts.base_estimator = opts.base_estimator.upper()
    check_options(opts) 
    

    # Loading training data
    print "Loading %s ..." %opts.train_filename
    x_train, y_train = load_svmlight_file(opts.train_filename)
    x_train = x_train.todense()
    (N, D) = x_train.shape
    print "training data dimension = (%d, %d)" %(N, D)


    # Loading testing data
    print "Loading %s ..." %opts.test_filename
    x_test, y_test = load_svmlight_file(opts.test_filename)
    x_test = x_test.todense()
    (N, D) = x_test.shape
    print "testing data dimension = (%d, %d)" %(N, D)


    # feature normalization
    if( opts.normalized ):      
        if( opts.normalized == 1 ):
            scaler_filename = opts.train_filename + '.scaler-11.pkl'
        elif( opts.normalized == 2 ):
            scaler_filename = opts.train_filename + '.scaler-01.pkl'
        elif( opts.normalized == 3 ):
            scaler_filename = opts.train_filename + '.scaler-std.pkl'
        else:
            print "Error! Unknown normalization method (%d)!" %opts.normalized
            print "Choice: 1 for [-1, 1], 2 for [0, 1], 3 for standard normalization"
            traceback.print_stack()
            sys.exit(1)

        scaler = load_scaler(scaler_filename, x_train, opts.normalized)
        x_train = scaler.transform(x_train)
        x_test  = scaler.transform(x_test)

    
    # dimension selection
    if( opts.dim != None ):
        opts.dim = int(opts.dim)
        if( opts.dim >= D ):
            print "Warning! Select dimension (%d) >= max data dimension (%d), use original dimension." %(opts.dim, D)
            opts.dim = D
        else:
            x_train = x_train[:, :opts.dim]
            x_test  = x_test[:, :opts.dim]
            (N, D) = x_train.shape
            print "Using first %d feature ..." %(opts.dim)

    
    if( opts.prob_filename != None ):
        outputProb = True
    else:
        outputProb = False


    # Train and predict
    print "Training and Predicting ..."

    if opts.model == 'SVM':

        arg = {'kernel': opts.kernel, 'probability': outputProb}
        
        if( opts.C == None ):
            arg['C'] = 1.0 
        else:
            arg['C'] = float(opts.C)
        
        if( opts.gamma == None ):
            arg['gamma'] = 1.0/D
        else:
            arg['gamma'] = float(opts.gamma)

############################################################
##                        RBF-SVM                         ##
############################################################
        if( opts.kernel == 'rbf' ):
            
            print 'Run %s-SVM with C = %f, gamma = %f' %(opts.kernel, arg['C'], arg['gamma'])
            clf = train(SVC, arg, x_train, y_train, opts.model_filename)
            acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
            print 'acc = %f' % acc

############################################################
##                    polynomial-SVM                      ##
############################################################
        elif( opts.kernel == 'poly' ):
            
            if( opts.coef0 == None ):
                arg['coef0'] = 0
            else:
                arg['coef0'] = float(opts.coef0)
            
            if( opts.degree == None ):
                arg['degree'] = 3
            else:
                arg['degree'] = int(opts.degree)
            
            
            print 'Run %s-SVM with C = %f, coef0 = %f, gamma = %f, degree = %d' %(opts.kernel, arg['C'], arg['coef0'], arg['gamma'], arg['degree'])
            clf = train(SVC, arg, x_train, y_train, opts.model_filename)
            acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
            print 'acc = %f' % acc
    
############################################################
##                    sigmoid-SVM                         ##
############################################################
        elif( opts.kernel == 'sigmoid' ):
            
            if( opts.coef0 == None ):
                arg['coef0'] = 0
            else:
                arg['coef0'] = float(opts.coef0)
            
            print 'Run %s-SVM with C = %f, coef0 = %f, gamma = %f' %(opts.kernel, arg['C'], arg['coef0'], arg['gamma'])
            clf = train(SVC, arg, x_train, y_train, opts.model_filename)
            acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
            print 'acc = %f' % acc

        else:
            print "Error! Unknown kernel %s!" %opts.kernel
            traceback.print_stack()
            sys.exit(1)

############################################################
##                     linear-SVM                         ##
############################################################
    elif opts.model == 'LINEARSVM':

        if( outputProb == True ):
            print "Warning! Probability output is not supported in LinearSVM!"
            outputProb = False

        arg = {}
        if( opts.penalty == None ):
            arg['penalty'] = 'l2'
        else:
            arg['penalty'] = opts.penalty

        if( opts.penalty == 'l1' ):
            arg['dual'] = False
        
        if( opts.loss == None ):
            arg['loss'] = 'l2'
        else:
            arg['loss'] = opts.loss

        if( opts.C == None ):  # run all C
            arg['C'] = 1.0 / D
        else:
            arg['C'] = float(opts.C)

        print 'Run Linear_SVM with C = %f, penalty = %s, loss = %s' %(arg['C'], arg['penalty'], arg['loss'])
        clf = train(LinearSVC, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename)
        print "acc = %f" %acc

############################################################
##                Linear model with SGD                   ##
############################################################
    elif opts.model == 'SGD':
        
        if( outputProb == True ):
            print "Warning! Probability output is not supported in SGD!"
            outputProb = False

        arg = {}
        if( opts.penalty == None ):
            arg['penalty'] = 'l2'
        else:
            arg['penalty'] = opts.penalty
        
        if( opts.loss == None ):
            arg['loss'] = 'hinge'
        else:
            arg['loss'] = opts.loss

        if( opts.alpha == None ):
            arg['alpha'] = 0.0001
        else:
            arg['alpha'] = float(opts.alpha)

        print 'Run Linear-SVM with alpha = %f, penalty = %s, loss = %s' %(arg['alpha'], arg['penalty'], arg['loss'])
        clf = train(SGDClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename)
        print "acc = %f" %acc

############################################################
##                     Random Forest                      ##
############################################################
    elif opts.model == 'RF':
        arg = {}
        if( opts.n_estimators == None ):
            arg['n_estimators'] = 100
        else:
            arg['n_estimators'] = int(opts.n_estimators)
    
        print 'Run RandomForest with n_estimators = %d' %(arg['n_estimators'])
        clf = train(RandomForestClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
        print "acc = %f" %acc

############################################################
##                        AdaBoost                        ##
############################################################
    elif opts.model == 'ADABOOST':
        arg = {}

        be_DT        = DecisionTreeClassifier()
        be_SVC       = SVC(probability=True)
        be_SGD_huber = SGDClassifier(loss='modified_huber')
        be_SGD_log   = SGDClassifier(loss='log')

        if( opts.base_estimator == None or opts.base_estimator == 'DT' ):
            be = [ be_DT ]
        elif( opts.base_estimator == 'SVM' ):
            be = [ be_SVC ]
        elif( opts.base_estimator == 'SGD' or opts.base_estimator == 'SGD-HUBER' ):
            be = [ be_SGD_huber ]
        elif( opts.base_estimator == 'SGD-LOG' ):
            be = [ be_SGD_log ]
        else:
            print "Unkinown base estimator %s !" %opts.base_estimator
            traceback.print_stack()
            sys.exit(1)

        if( opts.n_estimators == None ):
            arg['n_estimators'] = 100
        else:
            arg['n_estimators'] = int(opts.n_estimators)
        
        if( opts.learning_rate == None ):
            arg['learning_rate'] = 1.0
        else:
            arg['learning_rate'] = float(opts.learning_rate)
        
        print 'Run AdaBoost with n_estimators = %d, learning_rate = %f' %(arg['n_estimators'], arg['learning_rate'])
        clf = train(AdaBoostClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
        print "acc = %f" %acc

############################################################
##                    GradientBoost                       ##
############################################################
    elif opts.model == 'GB':
        arg = {}
        if( opts.n_estimators == None ):
            arg['n_estimators'] = 100
        else:
            arg['n_estimators'] = int(opts.n_estimators)
        
        if( opts.learning_rate == None ):
            arg['learning_rate'] = 0.1
        else:
            arg['learning_rate'] = float(opts.learning_rate)
        
        print 'Run GradientBoosting with n_estimators = %d, learning_rate = %f' %(arg['n_estimators'], arg['learning_rate'])
        clf = train(GradientBoostingClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
        print "acc = %f" %acc

############################################################
##                          KNN                           ##
############################################################
    elif opts.model == 'KNN':
        arg = {}        
        if( opts.n_neighbors == None ):
            arg['n_neighbors'] = 5
        else:
            arg['n_neighbors'] = int(opts.n_neighbors)
        
        if( opts.degree == None ):
            arg['p'] = 2
        else:
            arg['p'] = int(opts.degree)

        if( opts.weights == None ):
            arg['weights'] = 'distance'
        else:
            arg['weights'] = opts.weights

        print 'Run KNN with n_neighbors = %d, weights = %s, power of distance metric = %d' %(arg['n_neighbors'], arg['weights'], arg['p'])
        clf = train(KNeighborsClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
        print "acc = %f" %acc

############################################################
##                  Logistic Regression                   ##
############################################################
    elif opts.model == 'LR':
        arg = {}
        if( opts.penalty == None ):
            arg['penalty'] = 'l2'
        else:
            arg['penalty'] = opts.penalty
        
        if( opts.C == None ):  # run all C
            arg['C'] = 1.0
        else:
            arg['C'] = float(opts.C)

        if(arg['penalty'] == 'l2'):
            arg['dual'] = True

        print 'Run Logistic Regression with C = %f, penalty = %s' %(arg['C'], arg['penalty'])
        clf = train(LogisticRegression, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
        print "acc = %f" %acc
    
############################################################
##                    Ridge Regression                    ##
############################################################
    elif opts.model == 'RIDGE':
        arg = {}
        if( opts.alpha == None ):
            arg['alpha'] = 1.0
        else:
            arg['alpha'] = float(opts.alpha)

        print 'Run Ridge Regression with alpha = %f' %(arg['alpha'])
        clf = train(RidgeClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename, opts.prob_filename)
        print "acc = %f" %acc

############################################################
##                 Gaussian Naive Bayes                   ##
############################################################
    elif opts.model == 'GNB':

        print 'Run Gaussian Naive Bayes'
        clf = train(GaussianNB, {}, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename)
        print 'acc = %f' % acc

############################################################
##             Linear Discriminant Analysis               ##
############################################################
    elif opts.model == 'LDA':
        
        print 'Run Linear Discriminant Analysis'
        clf = train(LDA, {}, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.output_filename)
        print 'acc = %f' % acc

    else:
        sys.stderr.write('Error: invalid model %s\n' %opts.model)
        traceback.print_stack()
        sys.exit(1)


if __name__ == "__main__":
    ts = time.time()
    main()
    te = time.time()
    print_time(ts, te)

