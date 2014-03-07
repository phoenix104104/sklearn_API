#!/usr/bin/python -u

import sys, os, argparse, time, traceback, re 
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, RandomizedLogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.cross_validation import KFold
from sklearn.grid_search import ParameterGrid
from pickle import dump, load
from ml_final_util import LoadParser, LoadScaler, CheckOpts
from multiprocessing import Pool
import numpy as np

def parallel_cross_validation(Classifier, clf_name, arg_list, X, y, fold=5, thread=4):
    
    print "Run parallel %d-fold CV with %d thread..." %(fold, thread)
    p = Pool(thread)
    all_arg = []
    for arg in arg_list:
        all_arg.append( (Classifier, clf_name, arg, X, y, fold) )

    result = p.map_async(cross_validation, all_arg)
    res = result.get()
    (acc_max, arg_best) = max(res)
    return (acc_max, arg_best)


def cross_validation( (Classifier, clf_name, arg, X, y, fold) ):
    
    word = "Run %s with " %clf_name
    for key in arg:
        word += "%s = %s, " %(str(key), str(arg[key]) )
    word += "%d-fold CV ...\n" %fold
    print word
    
    clf = Classifier(**arg)
    (N, D) = X.shape
    kf = KFold(N, n_folds=fold)
    
    acc = 0

    for t_idx, v_idx in kf:
        x_train = X[t_idx, :]
        y_train = y[t_idx]
        x_valid = X[v_idx, :]
        y_valid = y[v_idx]

        clf.fit(x_train, y_train)
        y_pred = np.round( clf.predict(x_valid) )
        
        # calculate accuracy
        acc += float(sum(y_valid == y_pred)) / len(y_pred)

    acc /= fold
    print 'acc = %f' %acc
    return (acc, arg)

def parseGrid(grid_str, base=0, default_step=1):

    r = re.split('[\{\}:]', grid_str)
    s = filter(None, r)
    
    p_list = []
    if( len(s) > 3 ):
        sys.stderr.write('Error! Usage: {begin:end} or {begin:end:step}\n')
        sys.exit(1)
    elif( len(s) == 1 ):
        if( base == 0 ):
            p_list.append( int(s[0]) )
        else:
            p_list.append( base**int(s[0]) )
    else: 
        start = int(s[0])
        end   = int(s[1])
        if( len(s) == 3 ):
            step = int(s[2])
        else:
            step = default_step

        for i in range(start, end, step):
            if( base == 0 ):
                p_list.append(i)
            else:
                p_list.append( base**i )
    
    return p_list

def main():
    
    parser = LoadParser()
    parser.add_argument('-f'        , dest='fold'  , type=int, default=3, help='number of fold in cross_validation [default = 3]')
    parser.add_argument('-th'       , dest='thread', type=int, default=8, help='Number of thread to run in parallel [default = 8]')
    parser.add_argument('-log2c'    , dest='log2_C'                     , help='Grid search {begin:end:step} for log2_C [default = {0:7} for SVM, {-2:4} for LR]')
    parser.add_argument('-log2g'    , dest='log2_gamma'                 , help='Grid search {begin:end:step} for log2_gamma [default = {-2:3:2} for SVM]')
    parser.add_argument('-log2r'    , dest='log2_coef0'                 , help='Grid search {begin:end:step} for log2_coef0 [default = {-2:3:2} for SVM]')
    parser.add_argument('-log2lr'   , dest='log2_lr'                    , help='Grid search {begin:end:step} for log2_learning_rate [default = {-2:3:2}]')
    parser.add_argument('-log2a'    , dest='log2_alpha'                 , help='Grid search {begin:end:step} for log2_alpha [default = {-3:4:1}]')

    opts = parser.parse_args(sys.argv[1:])  
    opts.model = opts.model.upper()

    # pre-check options before loading data
    CheckOpts(opts) 

    # Loading training data
    print "Loading %s ..." %opts.train_filename
    x_train, y_train = load_svmlight_file(opts.train_filename)
    x_train = x_train.todense()
    (N, D) = x_train.shape
    print "training data dimension = (%d, %d)" %(N, D)


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
            traceback.exc()
            sys.exit(1)

        scaler = LoadScaler(scaler_filename, x_train, opts.normalized)
        x_train = scaler.transform(x_train)

    if( opts.dim == None ):
        dim_list = [D]
    else:
        dim_list = parseGrid(opts.dim, 0, 100)

    print dim_list
    
    x_train_all = x_train

    for dim in dim_list:
        if( dim > D ):
            print "Warning! Select dimension (%d) >= max data dimension (%d), use original dimension." %(dim, D)
            dim = D
        else:
            x_train = x_train_all[:, :dim]
            print "Using first %d feature ..." %(dim)


        # Train and predict
        print "Cross Validation..."
        
        if opts.model == 'SVM':

        #   RBF-SVM
            if( opts.kernel == 'rbf' ):
                c_list = []
                if( opts.C != None ):
                    c_list.append(opts.C)
                else:
                    if( opts.log2_C == None ): # default grid search
                        for i in range(0, 7):
                            c_list.append( 2**i )
                    else:
                        c_list = parseGrid(opts.log2_C, 2, 1) # base = 2, default step = 1

                gamma_list = []
                if( opts.gamma != None ):
                    gamma_list.append( opts.gamma )
                else:
                    if( opts.log2_gamma == None ):
                        for i in range(-4, 5, 2):
                            gamma_list.append( 2**i )
                    else:
                        gamma_list = parseGrid(opts.log2_gamma, 2, 2)
                
                arg_list = list( ParameterGrid( {'kernel': [opts.kernel], 'gamma': gamma_list, 'C': c_list} ) )

                (acc_max, arg_best) = parallel_cross_validation(SVC, 'SVM', arg_list, x_train, y_train, opts.fold, opts.thread)
                
                print "#####################################################################################"
                print "max_acc = %f --- C = %f, gamma = %f" %(acc_max, arg_best['C'], arg_best['gamma'])
                print "#####################################################################################"

        # polynomial-SVM
            elif( opts.kernel == 'poly' ):
                c_list = []
                if( opts.C != None ):
                    c_list.append(opts.C)
                else:
                    if( opts.log2_C == None ): # default grid search
                        for i in range(0, 7):
                            c_list.append( 2**i )
                    else:
                        c_list = parseGrid(opts.log2_C, 2, 1) # base = 2, default step = 1

                gamma_list = []
                if( opts.gamma != None ):
                    gamma_list.append( opts.gamma )
                else:
                    if( opts.log2_gamma == None ):
                        for i in range(-2, 3, 2):
                            gamma_list.append( 2**i )
                    else:
                        gamma_list = parseGrid(opts.log2_gamma, 2, 2)
                 
                coef0_list = []
                if( opts.coef0 != None ):
                    coef0_list.append( opts.coef0 )
                else:
                    if( opts.log2_coef0 == None ):
                        for i in range(-2, 3, 2):
                            coef0_list.append( 2**i )
                    else:
                        coef0_list.append(opts.log2_coef0, 2, 2)
                
                degree_list = []
                if( opts.degree == None ):
                    for i in range(1, 5):
                        degree_list.append(i)
                else:
                    degree_list = parseGrid(opts.degree, 0, 1)
                
                arg_list = list( ParameterGrid( {'kernel':[opts.kernel], 'degree': degree_list, 'coef0': coef0_list, 'gamma': gamma_list, 'C': c_list} ) )
                
                (acc_max, arg_best) = parallel_cross_validation(SVC, 'SVM', arg_list, x_train, y_train, opts.fold, opts.thread)
                
                print "#####################################################################################"
                print "max_acc = %f --- C = %f, coef0 = %f, gamma = %f, degree = %d" %(acc_max, arg_best['C'], arg_best['coef0'], arg_best['gamma'], arg_best['degree'])
                print "#####################################################################################"
        
        # sigmoid-SVM
            elif( opts.kernel == 'sigmoid' ):
                c_list = []
                if( opts.C != None ):
                    c_list.append(opts.C)
                else:
                    if( opts.log2_C == None ): # default grid search
                        for i in range(0, 7):
                            c_list.append( 2**i )
                    else:
                        c_list = parseGrid(opts.log2_C, 2, 1) # base = 2, default step = 1

                gamma_list = []
                if( opts.gamma != None ):
                    gamma_list.append( opts.gamma )
                else:
                    if( opts.log2_gamma == None ):
                        for i in range(-2, 3, 2):
                            gamma_list.append( 2**i )
                    else:
                        gamma_list = parseGrid(opts.log2_gamma, 2, 2)
                 
                coef0_list = []
                if( opts.coef0 != None ):
                    coef0_list.append( opts.coef0 )
                else:
                    if( opts.log2_coef0 == None ):
                        for i in range(-2, 3, 2):
                            coef0_list.append( 2**i )
                    else:
                        coef0_list.append(opts.log2_coef0, 2, 2)
                
                
                arg_list = list( ParameterGrid( {'kernel': [opts.kernel], 'coef0': coef0_list, 'gamma': gamma_list, 'C': c_list } ) )
                
                (acc_max, arg_best) = parallel_cross_validation(SVC, 'SVM', arg_list, x_train, y_train, opts.fold, opts.thread)
                
                print "#####################################################################################"
                print "max_acc = %f --- C = %f, coef0 = %f, gamma = %f" %(acc_max, arg_best['C'], arg_best['coef0'], arg_best['gamma'])
                print "#####################################################################################"
        

            else:
                print "Error! Unknown kernel %s!" %opts.kernel
                traceback.print_exc()
                sys.exit(1)

    #   SVC linear
        elif opts.model == 'LINEARSVM':
            
            penalty_list = []
            if( opts.penalty == None ):
                penalty_list.append('l2')
                penalty_list.append('l1')
            else:
                penalty_list.append( opts.penalty )
            
            loss_list = []
            if( opts.loss == None ):
                loss_list.append('l2')
                loss_list.append('l1')
            else:
                loss_list.append( opts.loss )

            c_list = []
            if( opts.C != None ):
                c_list.append(opts.C)
            else:
                if( opts.log2_C == None ): # default grid search
                    for i in range(0, 7):
                        c_list.append( 2**i )
                else:
                    c_list = parseGrid(opts.log2_C, 2, 1) # base = 2, default step = 1
        
            arg_list = list( ParameterGrid( {'penalty': penalty_list, 'loss': loss_list, 'C': c_list} ) )

            acc_max = -1
            arg_list_r = []
            for arg in arg_list:
                if( arg['penalty'] == 'l1' and arg['loss'] == 'l1' ):
                    # not support
                    continue

                if( arg['penalty'] == 'l1' ):
                    arg['dual'] = False

                arg_list_r.append(arg)

            (acc_max, arg_best) = parallel_cross_validation(LinearSVC, 'Linear-SVM', arg_list_r, x_train, y_train, opts.fold, opts.thread)
            
            print "#####################################################################################"
            print "max_acc = %f --- C = %f, penalty = %s, loss = %s" %(acc_max, arg_best['C'], arg_best['penalty'], arg_best['loss'])
            print "#####################################################################################"
            '''
            # Use SGD with hinge loss function to replace linearSVM
            alpha_list = []
            if( opts.alpha != None ):
                alpha_list.append(opts.alpha)
            else:
                if( opts.log2_alpha == None ): # default grid search
                    for i in range(-5, 3):
                        alpha_list.append( 2**i )
                else:
                    alpha_list = parseGrid(opts.log2_alpha, 2, 1) # base = 2, default step = 1

            loss_list = []
            loss_list.append('hinge')
            
            penalty_list = []
            if( opts.penalty == None ):
                penalty_list.append('l2')
                penalty_list.append('l1')
                penalty_list.append('elasticnet')
            else:
                penalty_list.append(opts.penalty)
            
            arg_list = list( ParameterGrid( {'alpha': alpha_list, 'loss':loss_list, 'penalty':penalty_list} ) )
            (acc_max, arg_best) = parallel_cross_validation(SGDClassifier, 'Linear-SVM', arg_list, x_train, y_train, opts.fold, opts.thread)
            
            print "#####################################################################################"
            print "max_acc = %f --- alpha = %f, penalty = %s" %(acc_max, arg_best['alpha'], arg_best['penalty'])
            print "#####################################################################################"
            '''
    # linear SGD
        elif opts.model == 'SGD':

            alpha_list = []
            if( opts.alpha != None ):
                alpha_list.append(opts.alpha)
            else:
                if( opts.log2_alpha == None ): # default grid search
                    for i in range(-5, 3):
                        alpha_list.append( 2**i )
                else:
                    alpha_list = parseGrid(opts.log2_alpha, 2, 1) # base = 2, default step = 1

            loss_list = []
            loss_list.append('hinge')
            loss_list.append('log')
            loss_list.append('modified_huber')
            loss_list.append('squared_hinge')
            loss_list.append('perceptron')
            loss_list.append('squared_loss')
            loss_list.append('huber')
            loss_list.append('epsilon_insensitive')
            loss_list.append('squared_epsilon_insensitive')
            
            penalty_list = []
            penalty_list.append('l2')
            penalty_list.append('l1')
            penalty_list.append('elasticnet')
            
            arg_list = list( ParameterGrid( {'alpha': alpha_list, 'loss':loss_list, 'penalty':penalty_list} ) )
            (acc_max, arg_best) = parallel_cross_validation(SGDClassifier, 'Linear-SGD', arg_list, x_train, y_train, opts.fold, opts.thread)
            
            print "#####################################################################################"
            print "max_acc = %f --- alpha = %f, loss = %s, penalty = %s" %(acc_max, arg_best['alpha'], arg_best['loss'], arg_best['penalty'])
            print "#####################################################################################"

    #   RandomForest
        elif opts.model == 'RF':
            arg = {'n_jobs':10}
            ne_list = []
            if( opts.n_estimators == None ):
                for i in range(5, 31, 5):
                    ne_list.append( 10*i )
            else:
                ne_list = parseGrid(opts.n_estimators, 0, 10)
            
            arg_list = list( ParameterGrid( {'n_estimators': ne_list} ) )
            (acc_max, arg_best) = parallel_cross_validation(RandomForestClassifier, 'Random Forest', arg_list, x_train, y_train, opts.fold, opts.thread)

            print "#####################################################################################"
            print "max_acc = %f --- n_estimators = %d" %(acc_max, arg_best['n_estimators'])
            print "#####################################################################################"


    #   AdaBoost
        elif opts.model == 'ADABOOST':
            be = [ DecisionTreeClassifier() ]
            #be = [SVC(probability=True)]
            #be = [SGDClassifier(loss='modified_huber')]
            ne_list = []
            if( opts.n_estimators == None ):
                for i in range(5, 31, 5):
                    ne_list.append( 10*i )
            else:
                ne_list = parseGrid(opts.n_estimators, 0, 10)
        
            lr_list = []
            if( opts.learning_rate != None ):
                lr_list.append(opts.learning_rate)
            else:
                if( opts.log2_lr == None ):
                    for i in range(-2, 2):
                        lr_list.append( 2**i )
                else:
                    lr_list = parseGrid(opts.log2_lr, 2, 1)

            arg_list = list( ParameterGrid( {'base_estimator': be, 'n_estimators': ne_list, 'learning_rate': lr_list} ) )
            (acc_max, arg_best) = parallel_cross_validation(AdaBoostClassifier, 'AdaBoost', arg_list, x_train, y_train, opts.fold, opts.thread)
            
            print "#####################################################################################"
            print "max_acc = %f --- base_estimator = %s, n_estimators = %d, learning_rate = %f" %(acc_max, arg_best['base_estimator'], arg_best['n_estimators'], arg_best['learning_rate'])
            print "#####################################################################################"


    #   GradientBoosting
        elif opts.model == 'GB':
            ne_list = []
            if( opts.n_estimators == None ):
                for i in range(2, 16, 2):
                    ne_list.append( 10*i )
            else:
                ne_list = parseGrid(opts.n_estimators, 0, 10)
        
            lr_list = []
            if( opts.learning_rate != None ):
                lr_list.append(opts.learning_rate)
            else:
                if( opts.log2_lr == None ):
                    for i in range(-2, 2):
                        lr_list.append( 2**i )
                else:
                    lr_list = parseGrid(opts.log2_lr, 2, 1)

            arg_list = []

            arg_list = list( ParameterGrid( {'n_estimators': ne_list, 'learning_rate': lr_list} ) )
            (acc_max, arg_best) = parallel_cross_validation(GradientBoostingClassifier, 'GradientBoosting', arg_list, x_train, y_train, opts.fold, opts.thread)

            print "#####################################################################################"
            print "max_acc = %f --- n_estimators = %d, learning_rate = %f" %(acc_max, arg_best['n_estimators'], arg_best['learning_rate'])
            print "#####################################################################################"

    #   KNN
        elif opts.model == 'KNN':
            
            nn_list = []
            if( opts.n_neighbors == None ):
                for i in range(5):
                    nn_list.append(5 + 10 * i)
            else:
                nn_list = parseGrid(opts.n_neighbors, 0, 10)
            
            p_list = []
            if( opts.degree == None ):
                p_list.append(1)
                p_list.append(2)
            else:
                p_list.append( opts.degree )

            weight_list = []
            if( opts.weights == None ):
                weight_list.append('distance')
                weight_list.append('uniform')
            else:
                weight_list.append( opts.weights ) 

            arg_list = list( ParameterGrid( {'n_neighbors': nn_list, 'p': p_list, 'weights': weight_list} ) )
            
            (acc_max, arg_best) = parallel_cross_validation(KNeighborsClassifier, 'KNN', arg_list, x_train, y_train, opts.fold, opts.thread)

            print "#####################################################################################"
            print "max_acc = %f --- n_neighbors = %d, weights = %s, p = %d" %(acc_max, arg_best['n_neighbors'], arg_best['weights'], arg_best['p'])
            print "#####################################################################################"


    #   Logistic Regression
        elif opts.model == 'LR':

            penalty_list = []
            if( opts.penalty == None ):
                penalty_list.append('l2')
                penalty_list.append('l1')
            else:
                penalty_list.append(opts.penalty)
            
            c_list = []
            if( opts.C != None ):
                c_list.append(opts.C)
            else:
                if( opts.log2_C == None ): # default grid search
                    for i in range(-2, 5):
                        c_list.append( 2**i )
                else:
                    c_list = parseGrid(opts.log2_C, 2, 1) # base = 2, default step = 1

            arg_list = list( ParameterGrid( {'penalty': penalty_list, 'C': c_list} ) )
        
            arg_list_r = []
            for arg in arg_list:
                if(arg['penalty'] == 'l2'):
                    arg['dual'] = True
                arg_list_r.append(arg)

            (acc_max, arg_best) = parallel_cross_validation(LogisticRegression, 'Logistic Regression', arg_list_r, x_train, y_train, opts.fold, opts.thread)

            print "#####################################################################################"
            print "max_acc = %f --- C = %f, penalty = %s" %(acc_max, arg_best['C'], arg_best['penalty'])
            print "#####################################################################################"

    #   Ridge Regression
        elif opts.model == 'RIDGE':

            alpha_list = []
            if( opts.alpha != None ):
                alpha_list.append(opts.alpha)
            else:
                if( opts.log2_alpha == None ): # default grid search
                    for i in range(-3, 4):
                        alpha_list.append( 2**i )
                else:
                    alpha_list = parseGrid(opts.log2_alpha, 2, 1) # base = 2, default step = 1

            arg_list = list( ParameterGrid( {'alpha': alpha_list} ) )
        
            (acc_max, arg_best) = parallel_cross_validation(RidgeClassifier, 'Ridge', arg_list, x_train, y_train, opts.fold, opts.thread)

            print "#####################################################################################"
            print "max_acc = %f --- alpha = %f" %(acc_max, arg_best['alpha'])
            print "#####################################################################################"


        elif opts.model == 'GNB':

            print 'Run Gaussian Naive Bayes (%d-fold CV)' %(opts.fold)
            (acc, arg) = cross_validation( (GaussianNB, 'GNB', {}, x_train, y_train, opts.fold) )

            print 'acc = %f' % acc
        
        elif opts.model == 'LDA':
            
            print 'Run Linear Discriminant Analysis (%d-fold CV)' %(opts.fold)
            (acc, arg) = cross_validation( (LDA, 'LNA', {}, x_train, y_train, opts.fold) )

            print "max_acc = %f " %(acc)

        else:
            sys.stderr.write('Error: invalid model ' + opts.model + '\n')




if __name__ == "__main__":
    ts = time.time()
    main()
    te = time.time()
    print "Elapsed time is %f sec." %(te - ts)
