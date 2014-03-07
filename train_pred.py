#!/usr/bin/python -u

import sys, os, argparse, time, traceback 
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.cross_validation import KFold
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import precision_recall_curve
from pickle import dump, load
from ml_final_util import LoadParser, LoadScaler, CheckOpts
import numpy as np

def train(Classifier, arg, X, y, model_filename):
    
    clf = Classifier(**arg)
    clf.fit(X, y)

    if( model_filename != None ):
        with open(model_filename, 'w') as f:
            print "Saving model file %s ..." %model_filename
            dump(clf, f)

    return clf

def predict(clf, X, y, pred_filename, prob_filename=None):

    y_pred = clf.predict(X)

    # calculate accuracy
    acc = float(sum(y == y_pred)) / len(y_pred)

    if( pred_filename != None ):
        print "Saving predict file %s ..." %pred_filename
        np.savetxt(pred_filename, y_pred, '%d')
    
    if( prob_filename != None ):
        prob = clf.predict_proba(X)
        print "Saving probability file %s ..." %prob_filename
        dump_svmlight_file(prob, y, prob_filename)
        #np.savetxt(prob_filename, prob, '%.6e')
    
    return acc


def main():

    parser = LoadParser()
    parser.add_argument('-t' , dest='test_filename' , required=True , help='Specify the test1 file')
    parser.add_argument('-om', dest='model_filename'                , help='Specify the output model file [optional]')
    parser.add_argument('-op', dest='pred_filename'                 , help='Specify the output predict file [optional]')
    parser.add_argument('-o' , dest='output_filename'               , help='Specify the output model/predict file [optional]')

    opts = parser.parse_args(sys.argv[1:]) 
    opts.model = opts.model.upper()
    # pre-check options before loading data
    
    CheckOpts(opts) 
   
    prob_filename = None
    if( opts.output_filename != None ):
        opts.model_filename = '../model/' + opts.output_filename + '.model'
        opts.pred_filename  = '../pred/'  + opts.output_filename + '.pred'
        prob_filename       = '../prob/'  + opts.output_filename + '.prob'


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
            traceback.print_exc()
            sys.exit(1)

        scaler = LoadScaler(scaler_filename, x_train, opts.normalized)
        x_train = scaler.transform(x_train)
        x_test  = scaler.transform(x_test)


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


    # Train and predict
    print "Training and Predicting ..."
    
    outputProb = True

    if opts.model == 'SVM':
    #   RBF-SVM
        if( opts.kernel == 'rbf' ):
            arg = {'kernel': 'rbf', 'probability': True}
            
            if( opts.C == None ):
                arg['C'] = 1.0 
            else:
                arg['C'] = opts.C
            
            if( opts.gamma == None ):
                arg['gamma'] = 1/D
            else:
                arg['gamma'] = opts.gamma
            
            print 'Run %s-SVM with C = %f, gamma = %f' %(opts.kernel, arg['C'], arg['gamma'])
            clf = train(SVC, arg, x_train, y_train, opts.model_filename)
            acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
            print 'acc = %f' % acc

    # polynomial-SVM
        elif( opts.kernel == 'poly' ):
            arg= {'kernel': 'poly', 'probability': True}

            if( opts.C == None ):
                arg['C'] = 1.0 
            else:
                arg['C'] = opts.C
            
            if( opts.gamma == None ):
                arg['gamma'] = 1/D
            else:
                arg['gamma'] = opts.gamma
            
            if( opts.coef0 == None ):
                arg['coef0'] = 0
            else:
                arg['coef0'] = opts.coef0
            
            if( opts.degree == None ):
                arg['degree'] = 3
            else:
                arg['degree'] = int(opts.degree)
            
            
            print 'Run %s-SVM with C = %f, coef0 = %f, gamma = %f, degree = %d' %(opts.kernel, arg['C'], arg['coef0'], arg['gamma'], arg['degree'])
            clf = train(SVC, arg, x_train, y_train, opts.model_filename)
            acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
            print 'acc = %f' % acc
    
    # sigmoid-SVM
        elif( opts.kernel == 'sigmoid' ):
            arg = {'kernel': 'sigmoid', 'probability': True}

            if( opts.C == None ):
                arg['C'] = 1.0 
            else:
                arg['C'] = opts.C
            
            if( opts.gamma == None ):
                arg['gamma'] = 1/D
            else:
                arg['gamma'] = opts.gamma
            
            if( opts.coef0 == None ):
                arg['coef0'] = 0
            else:
                arg['coef0'] = opts.coef0
            
            print 'Run %s-SVM with C = %f, coef0 = %f, gamma = %f' %(opts.kernel, arg['C'], arg['coef0'], arg['gamma'])
            clf = train(SVC, arg, x_train, y_train, opts.model_filename)
            acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
            print 'acc = %f' % acc

        else:
            print "Error! Unknown kernel %s!" %opts.kernel
            traceback.print_exc()
            sys.exit(1)

#   linearSVM
    elif opts.model == 'LINEARSVM':
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
            arg['C'] = opts.C

        print 'Run Linear_SVM with C = %f, penalty = %s, loss = %s' %(arg['C'], arg['penalty'], arg['loss'])
        clf = train(LinearSVC, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.pred_filename)
        print "acc = %f" %acc

#   linear SGD 
    elif opts.model == 'SGD':
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
        acc = predict(clf, x_test, y_test, opts.pred_filename)
        print "acc = %f" %acc

#   RandomForest 100 trees
    elif opts.model == 'RF':
        arg = {'n_jobs':10}
        if( opts.n_estimators == None ):
            arg['n_estimators'] = 100
        else:
            arg['n_estimators'] = int(opts.n_estimators)
    
        print 'Run RandomForest with n_estimators = %d' %(arg['n_estimators'])
        clf = train(RandomForestClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
        print "acc = %f" %acc

#   AdaBoost
    elif opts.model == 'ADABOOST':
        arg = {}
        if( opts.n_estimators == None ):
            arg['n_estimators'] = 100
        else:
            arg['n_estimators'] = int(opts.n_estimators)
        
        if( opts.learning_rate == None ):
            arg['learning_rate'] = 1.0
        else:
            arg['learning_rate'] = opts.learning_rate
        
        print 'Run AdaBoost with n_estimators = %d, learning_rate = %f' %(arg['n_estimators'], arg['learning_rate'])
        clf = train(AdaBoostClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
        print "acc = %f" %acc

#   GradientBoosting
    elif opts.model == 'GB':
        arg = {}
        if( opts.n_estimators == None ):
            arg['n_estimators'] = 100
        else:
            arg['n_estimators'] = int(opts.n_estimators)
        
        if( opts.learning_rate == None ):
            arg['learning_rate'] = 0.1
        else:
            arg['learning_rate'] = opts.learning_rate
        
        print 'Run GradientBoosting with n_estimators = %d, learning_rate = %f' %(arg['n_estimators'], arg['learning_rate'])
        clf = train(GradientBoostingClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
        print "acc = %f" %acc

#   KNN
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
        acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
        print "acc = %f" %acc


#   Logistic Regression
    elif opts.model == 'LR':
        arg = {}
        if( opts.penalty == None ):
            arg['penalty'] = 'l2'
        else:
            arg['penalty'] = opts.penalty
        
        if( opts.C == None ):  # run all C
            arg['C'] = 1.0
        else:
            arg['C'] = opts.C

        if(arg['penalty'] == 'l2'):
            arg['dual'] = True

        print 'Run Logistic Regression with C = %f, penalty = %s' %(arg['C'], arg['penalty'])
        clf = train(LogisticRegression, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
        print "acc = %f" %acc
    
#   Ridge Regression
    elif opts.model == 'RIDGE':
        arg = {}
        if( opts.alpha == None ):
            arg['alpha'] = 1.0
        else:
            arg['alpha'] = float(opts.alpha)

        print 'Run Ridge Regression with alpha = %f' %(arg['alpha'])
        clf = train(RidgeClassifier, arg, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.pred_filename, prob_filename)
        print "acc = %f" %acc

# Gaussian Naive Bayes
    elif opts.model == 'GNB':

        print 'Run Gaussian Naive Bayes'
        clf = train(GaussianNB, {}, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.pred_filename)
        print 'acc = %f' % acc

        #print clf.get_params()
        #print clf.theta_
        #print clf.sigma_
    
    elif opts.model == 'LDA':
        
        print 'Run Linear Discriminant Analysis'
        clf = train(LDA, {}, x_train, y_train, opts.model_filename)
        acc = predict(clf, x_test, y_test, opts.pred_filename)
        print 'acc = %f' % acc

    else:
        sys.stderr.write('Error: invalid model %s\n' %opts.model)




if __name__ == "__main__":
    ts = time.time()
    main()
    te = time.time()
    print "Elapsed time is %f sec." %(te - ts)
