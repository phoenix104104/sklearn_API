#!/usr/bin/python
import os, sys
import random
import argparse, traceback

def main():
    
    parser = argparse.ArgumentParser(description='Random sample unique elements from input list and seperate to multiple output lists (Used for sample training/testing dataset)')
    parser.add_argument('-i', dest='input_list' ,            required=True, help='input list name')
    parser.add_argument('-n', dest='sample_num' , nargs='*', required=True, help='number of sample (accept multiple inputs)')
    parser.add_argument('-o', dest='output_list', nargs='*', required=True, help='output list name (accept multiple inputs)')
    opts = parser.parse_args(sys.argv[1:])

    if( len(opts.sample_num) != len(opts.output_list) ):
        sys.stderr.write('Error! The length of sample number(%d) is not equal to output lists(%d)!' %( len(opts.sample_num), len(opts.output_list) ) )
        traceback.print_stack()
        exit(1)
    
    output_list_name = opts.output_list
    sample_num = [int(n) for n in opts.sample_num] 

    with open(opts.input_list) as f:
        print "Loading %s ..." %opts.input_list
        input_list = [line for line in f]

    N = len(input_list)
    nS = sum(sample_num)
    if (nS > N):
        sys.stderr.write("Error! Total number of sample (%d) > list length (%d)" %(nS, N) )
        traceback.print_stack()
        exit(1)

    sample_all = random.sample(input_list, nS)
    
    st = 0
    for i in range(len(sample_num)):
        ed = st + sample_num[i]
        sample_list = sample_all[st:ed]

        with open(output_list_name[i], 'w') as f:
            print "Saving %s ..." %output_list_name[i]
            f.writelines(sample_list)
        
        st = ed

if __name__ == "__main__":
    main()

