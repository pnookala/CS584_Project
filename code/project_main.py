#!/usr/bin/python

from myUtil import *
from imputation import *
from sklearn import svm
import  itertools
import re
# from itertools import chain

debug = True
outputDir = 'output'

def main(argv):

    #### Setup ####

    # Default options for debugging, use the command-line parameters to override
    filePath = 'data/iris.data'
    delimiter = ','
    has_header = False
    row_random_rate = [1.0]
    col_random_rate = [0]
    # row_random_rate = np.arange(0,1.1,0.1)
    # col_random_rate = np.arange(0,4,1)
    r_seed = 0


    def get_help(argv):
        print(argv[0] + ' -i <input_file> [-s] [-r <row_rand>] [-c <col_rand>] [--seed]')
        print('    -s  -   skips the first line of files that have a header')
        print('    -r  -   Parameter for clearing data, the rate at which rows should be zeroed [0-1]')
        print('    -c  -   Parameter for clearing data, the number of columns to potentially clear [1-<#features>]')
        print('    -seed - The random seed to use in clearing the data')
        print()
        print('Both the -r and -c parameters will accept a triplet of values specifying a range (as [start,end,step])')
        print('for the given parameter. The resultant range must still lie within the arguments\' bounds')

    def parse_range_arg(arg, dtype):
        if ',' in arg:
            groups = re.findall(r"([-\d.]+)", arg)
            if len(groups) != 3:
                raise ValueError("Must specify a triplet of values (start,end,step)")
            start = float(groups[0])
            end = float(groups[1])
            step = float(groups[2])
            res = np.arange(start, end, step, dtype=dtype)
            myshow(res, "parse_range_result")
            return  res
        else:
            v = np.ndarray((1), dtype=dtype)
            v[0] =  float(arg)
            return v

    try:
        opts, args = getopt.getopt(argv[1:], "hsi:r:c:", ["help", "ifile=", "skip-first", "seed="])
    except getopt.GetoptError:
        get_help(argv);
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            get_help(argv);
            sys.exit()

        elif opt in ("-i", "--ifile"):
            filePath = arg
            if debug:
                print("Input file: " + filePath)

        elif opt in ("-s", "--skip-first"):
            has_header = True
            if debug:
                print("Has header: " + has_header)

        elif opt in ("-r"):
            row_random_rate = parse_range_arg(arg, np.float)
            row_random_rate = row_random_rate[(row_random_rate >= 0.0) & (row_random_rate <= 1.0)]
            if debug:
                print("row_rand: " + str(row_random_rate))

        elif opt in ("-c"):
            col_random_rate = parse_range_arg(arg, int)
            col_random_rate = col_random_rate[col_random_rate >= 0]
            if debug:
                print("col_rand: " + str(col_random_rate))

        elif opt in ("--seed"):
            r_seed = arg
            if debug:
                print("seed: " + str(arg))


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

    data, output, class_indices, classes = read_and_parse(filePath, header=has_header, delimiter=delimiter)
    # myshow(data, "data", maxlines=15)
    # myshow(output, "output")

    #### Process data ####
    for rrand in row_random_rate:
        for crand in col_random_rate:
            process_data(data, output, class_indices, classes, filePath, float(rrand), int(crand), r_seed )

    print("Done, exiting")


def process_data(data_source, _output, class_indices, classes, filePath, row_random_rate, col_random_rate, r_seed):
    manually_zero = row_random_rate > 0
    parameters = [["Input File", "Zero Data", "Row Random Rate", "Col Random Rate", "Rand Seed"],
                  [filePath, manually_zero, row_random_rate, col_random_rate, r_seed]]

    _data = data_source.copy()

    if manually_zero:
        _data = clear_data_random(_data, row_rate=row_random_rate, col_rate=col_random_rate, seed=r_seed)

    if debug:
        myshow(_data, "data", maxlines=15)
        myshow(_output, "output")
        myshow(classes, "classes")
        myshow(class_indices, "class_indices")

    processor = Imputation();
    processor.estimate_values(_data)

    # myshow(data, "imputed data", maxlines=15)
    # myshow(data - old_data, "difference", maxlines=15)

    perform_classification(_data, classes, class_indices, parameters)



def perform_classification(data, classes, class_indices, parameters):
    clf = svm.SVC()
    clf.fit(data, class_indices)
    print(clf._get_coef())

    # rows are actual, columns are predicted
    conf_mat = np.zeros((len(classes), len(classes)), dtype=np.int64)
    estimates = np.zeros(class_indices.shape, dtype=class_indices.dtype)
    for i in range(len(class_indices)):
        estimates[i] = clf.predict(np.reshape(data[i,:], (1,-1)))
        conf_mat[estimates[i], class_indices[i]] += 1

    myshow(estimates, "estimates")
    analyze_errors(conf_mat, classes, parameters)


def analyze_errors(conf_mat, classes, parameters):
    accuracy, recall, precision, f_measure = CalculateStatistics(conf_mat)

    print("Confusion Matrix:")
    print_confusion_matrix(conf_mat, ["class " + str(x) for x in classes])

    number_format = '{:8.3%}'
    print("Accuracy:  " + number_format.format(accuracy))
    print("Precision: " + number_format.format(precision))
    print("Recall:    " + number_format.format(recall))
    print("F-Measure: " + number_format.format(f_measure))
    # print(accuracy, recall, precision, f_measure )

    save_errors(accuracy, recall, precision, f_measure, parameters)


def save_errors(accuracy, recall, precision, f_measure, parameters):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    output_path = os.path.join(outputDir, "results.csv")
    new_file = not os.path.exists(output_path)

    headers = ("Accuracy", "Precision", "Recall", "F-Measure")
    with open(output_path, 'a') as f:
        myshow(parameters, "Parameters")
        if new_file:
            f.write(','.join(itertools.chain( parameters[0], headers)))
            f.write('\n')
        f.write(','.join(str(x) for x in itertools.chain( parameters[1], (accuracy, precision, recall, f_measure))))
        f.write('\n')
        f.flush()


    # parameters
    # outputDir


if __name__ == "__main__":
    main(sys.argv)
