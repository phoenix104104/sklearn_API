#!/usr/bin/python

import sys, os, argparse, time
from util import print_time

def read_feature(file_name):
    feature = open(file_name, "r").read().split()
    return feature

def feature2svmformat(label, feature):
    
    out = label.__str__()
    for index, value in enumerate(feature):
        if value != 0:
            out += (" " + (index+1).__str__() + ":" + value)

    out += "\n"
    return out

def merge_data(data_list, input_dir_path, file_ext):
    curr_dir = os.getcwd()
    os.chdir(input_dir_path)
    print "Read feature %s at %s ..." %(file_ext, input_dir_path)

    out = ""
    N = len(data_list)
    i = 0
    sys.stdout.write("000%")
    for label, file_name in data_list:
        feature = read_feature(file_name + '.' + file_ext)
        out += feature2svmformat(label, feature)
        
        i += 1
        percent = round(i/float(N)*100)
        sys.stdout.write("\r%03d%%" %percent)

    sys.stdout.write("\r100%\n")
    os.chdir(curr_dir)
    return out


def main():
    parser = argparse.ArgumentParser(description='read a data name list and build svm format feature')
    parser.add_argument("-l", "--list"       , dest="data_list_name"  , required=True, help="input data list (list format: label file_name)")
    parser.add_argument("-i", "--input-dir"  , dest="input_dir"       , required=True, help="input directory path")
    parser.add_argument("-o", "--output-file", dest="output_file_name", required=True, help="output file name")
    parser.add_argument("-e", "--file-ext"   , dest="file_ext"        , required=True, help="input file extension name")
    opts = parser.parse_args(sys.argv[1:])

    with open(opts.data_list_name) as f:
        print "Load %s ..." %opts.data_list_name
        data_list = [tuple(line.split()) for line in f]
    
    output_data = merge_data(data_list, opts.input_dir, opts.file_ext)
    with open(opts.output_file_name, "w") as f:
        print "Saving %s ..." %opts.output_file_name
        f.write(output_data)

    sys.stdout.write("finished!\n")



if __name__ == "__main__":
    ts = time.time()
    main()
    te = time.time()
    print_time(ts, te)


