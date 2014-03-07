#!/usr/bin/python

import sys, os
from optparse import OptionParser

def readfeature(filename):
    feature = open(filename,"r").read().split()
    return feature

def writeEntry(label,feature,output):
    output.write(label.__str__())
    for index, value in enumerate(feature):
        if value != 0:
            output.write(" " + (index+1).__str__() + ":" + value)
    output.write("\n")

def merge_data(fout, data, input_dir, file_ext):
    current_dir = os.getcwd()
    os.chdir(input_dir)
    i = 0
    for label, filename in data:
        feature = readfeature(filename+'.'+file_ext)
        writeEntry(label, feature, fout)
        i += 1
        if(i % 1000 == 0):
            sys.stderr.write('.') 
    sys.stderr.write('\n')
    os.chdir(current_dir)


def main():
    parser = OptionParser()
    parser.add_option("-l",dest="data_list",help="Specify data list")
    parser.add_option("-i",dest="input_dir",help="Specify input directory")
    parser.add_option("-o",dest="output_file", help="Specify output file")
    parser.add_option("-e","--file-ext",dest="file_ext",help="Specify input file extension name",default='vlad')
    
    (options,args) = parser.parse_args(sys.argv[1:])

    fout = open(options.output_file,"w")
    with open(options.data_list) as f:
        data = [tuple(line.split()) for line in f]
    merge_data(fout, data, options.input_dir, options.file_ext)
    fout.close()
    sys.stdout.write("finished!\n")

if __name__ == "__main__":
    main()
