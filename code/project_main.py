#!/usr/bin/python

from myUtil import *
from imputation import *

debug = False

def main(argv):

    #### Setup ####

    # Default options, use the command-line parameters to override
    filePath = 'data/iris.data'
    outputDir = 'output'

    delimiter = ','
    # has_header = True
    has_header = False

    manually_zero = True


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
    # fileName = ntpath.basename(filePath)
    # outputFileBase = get_name_without_ext(fileName)
    # outputFileName = outputDir + '/' + outputFileBase + '_output'
    # if not os.path.exists(outputDir):
    #     os.makedirs(outputDir)


    #### Read/Parse Data ####
    print()
    print('Reading from "' + filePath + '"')
    print()

    data, output = read_and_parse(filePath, header=has_header, delimiter=delimiter)
    # myshow(data, "data", maxlines=15)
    # myshow(output, "output")

    #### Process data ####
    old_data = data
    if manually_zero:
        data = clear_data_random(data, row_rate=0.4, col_rate=2, seed=0)

    myshow(data, "data", maxlines=15)
    myshow(output, "output")
    print("Done, exiting")


if __name__ == "__main__":
    main(sys.argv)
