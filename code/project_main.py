#!/usr/bin/python

import sys, getopt, ntpath, os
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
from sklearn import linear_model
from myUtil import *

debug = False
dataType = np.float64


def main(argv):
    #### Input Parsing ####

    # Default options, use the command-line parameters to override
    filePath = 'data/iris.data'

    datadelimiter = ','
    # has_header = True
    has_header = False

    outputDir = 'output'

    def get_help(argv):
        print(argv[0] + ' -i <input_file> [-s]')
        print('    -s  -  skips the first line of files that have a header')

    try:
        opts, args = getopt.getopt(argv[1:], "hsi:", ["help", "ifile=", "skip-first"])
    except getopt.GetoptError:
        get_help(argv);
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            get_help(argv);
            sys.exit()

        elif opt in ("-i", "--ifile"):
            filePath = arg

        elif opt in ("-s", "--skip-first"):
            has_header = True


    #### Environment Setup ####
    fileName = ntpath.basename(filePath)

    try:
        i = fileName.rindex('.')
    except ValueError:
        i = len(fileName)

    outputFileBase = fileName[0:i]
    outputFileName = outputDir + '/' + outputFileBase + '_output'

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    print()
    print('Reading from "' + filePath + '"')
    print()

    #### Read/Parse Data ####
    content = np.genfromtxt(filePath,dtype=None,  delimiter=datadelimiter, skip_header=has_header)
    if debug:
        myshow(content, "content")
        print()


    N = content.shape[0];
    class_column = len(content[0]) - 1
    data_columns = [i for i in range(class_column)]

    try:
        class_type = content[0].dtype[class_column]
    except:
        class_type = content[0].dtype

    data = np.ndarray((N, class_column), dtype=dataType)
    classes = np.ndarray((N, 1), dtype=class_type)


    i = 0
    for d in content:
        for j in data_columns:
            data[i,j] = d[j]
        classes[i] = d[class_column]
        i+=1

    myshow(data, "data")
    myshow(classes, "classes")

    print("Done, exiting")

if __name__ == "__main__":
    main(sys.argv)
