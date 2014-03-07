#!/usr/bin/python -u
import os

cmds = []

cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data/cent/ -e cent -o ../svmformat/P2-cent.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data/cent/ -e cent -o ../svmformat/P2-cent.test")
cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data/cl5/ -e cl5   -o ../svmformat/P2-cl5.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data/cl5/ -e cl5   -o ../svmformat/P2-cl5.test")
cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data/cl5_n/ -e cl5_n -o ../svmformat/P2-cl5_n.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data/cl5_n/ -e cl5_n -o ../svmformat/P2-cl5_n.test")
cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data/dist2cent/ -e dist2cent -o ../svmformat/P2-dist2cent.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data/dist2cent/ -e dist2cent -o ../svmformat/P2-dist2cent.test")
cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data/dist2cent_n/ -e dist2cent_n -o ../svmformat/P2-dist2cent_n.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data/dist2cent_n/ -e dist2cent_n -o ../svmformat/P2-dist2cent_n.test")
cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data/dist2line/ -e dist2line -o ../svmformat/P2-dist2line.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data/dist2line/ -e dist2line -o ../svmformat/P2-dist2line.test")
cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data/sum_dist2cent/ -e sum_dist2cent -o ../svmformat/P2-sum_dist2cent.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data/sum_dist2cent/ -e sum_dist2cent -o ../svmformat/P2-sum_dist2cent.test")
cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data//sum_dist2cent_n -e sum_dist2cent_n -o ../svmformat/P2-sum_dist2cent_n.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data//sum_dist2cent_n -e sum_dist2cent_n -o ../svmformat/P2-sum_dist2cent_n.test")
cmds.append("./buildData.py -l ../list/P2-train.txt -i ../data//sum_dist2line -e sum_dist2line -o ../svmformat/P2-sum_dist2line.train")
cmds.append("./buildData.py -l ../list/P2-test.txt  -i ../data//sum_dist2line -e sum_dist2line -o ../svmformat/P2-sum_dist2line.test")

for cmd in cmds:
    print cmd
    os.system(cmd)
