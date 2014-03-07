#!/usr/bin/python -u
import sys, os, optparse 
import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

def main():
    if(len(sys.argv) < 4):
        sys.stderr.write('%s input1 input2 [...] output\n' % (sys.argv[0]))
        exit(1)
    all_X= []
    for fileName in sys.argv[1:-1]:
        print 'loading ' + fileName + ' ...'
        X, y = load_svmlight_file(fileName)
        print X.shape
        all_X.append(X.todense())
        
    X = np.concatenate(all_X, axis=1)
    print 'writing ' + sys.argv[-1] + ' ...'
    print X.shape
    dump_svmlight_file(X, y, sys.argv[-1])
    
if __name__ == '__main__':
    main()

