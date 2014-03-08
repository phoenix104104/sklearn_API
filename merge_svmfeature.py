#!/usr/bin/python -u
import sys, os, time, argparse
import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from util import print_time

def main():
    
    parser = argparse.ArgumentParser(description='Merge multiple svm format features.')
    parser.add_argument('-i', nargs='*', required=True, dest='input_filename' , help="Specify input file path (accept multiple inputs)")
    parser.add_argument('-o',            required=True, dest='output_filename', help="Specify output file path")
    opts = parser.parse_args(sys.argv[1:])

    all_X = []
    for fileName in opts.input_filename:
        print 'Loading ' + fileName + ' ...'
        X, y = load_svmlight_file(fileName)
        print X.shape
        all_X.append(X.todense())
        
    X = np.concatenate(all_X, axis=1)
    print 'Saving ' + opts.output_filename + ' ...'
    print X.shape
    dump_svmlight_file(X, y, opts.output_filename)
    
if __name__ == '__main__':
    ts = time.time()
    main()
    te = time.time()
    print_time(ts, te)

