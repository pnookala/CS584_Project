#!/usr/bin/python

from imputation import *
from sklearn import svm
import itertools
import re
from knnAlgorithm import *
from pylab import *
from numpy import *
from sklearn.preprocessing import Imputer

# debug = True
debug = False
outputDir = 'output'


def main(argv):
    #### Setup ####
    # Default options
    ignored_columns = None
    class_column = None
    impute_data = True
    use_sign = True  # Use ordering of imputed values
    filePath = None
    delimiter = ','
    has_header = False

    row_random_rate = [0]
    col_random_rate = [0]
    r_seed = None

    # Override defaults for debugging, use the command-line parameters otherwise
    # impute_data = False
    # use_sign = False

    filePath = 'data/iris.data'

    # filePath = 'data/hepatitis.data'
    # class_column = 0

    # filePath = 'data/soybean-large.data'
    # class_column = 0

    # filePath = 'data/echocardiogram.data'
    # class_column = 1
    # ignored_columns = [0, 10]

    # filePath = 'data/house-votes-84.data'
    # class_column = 0

    # filePath = 'data/bridges.data.version1.data'
    # class_column = 9
    # ignored_columns = [0,2]

    # filePath = 'data/water-treatment.data'
    # ignored_columns = [0]
    # class_column = None

    # row_random_rate = [.7]
    # col_random_rate = [2]
    # row_random_rate = np.arange(0, 1.0001, 0.1)
    # col_random_rate = np.arange(0, 8, 1)
    # r_seed = 0

    def get_help(argv):
        print(argv[0] + ' -i <input_file> [-l <class label column>] [-s] [-r <row_rand>] ' +
              '[-c <col_rand>] [--seed] [-I] [--skip-columns] [--no-sort]')
        print('    -l  -    The column containing the class label. Defaults to the last column [0-<# columns>]')
        print('    -s  -    skips the first line of files that have a header')
        print('    -r  -    Parameter for clearing data, the rate at which rows should be zeroed [0-1]')
        print('    -c  -    Parameter for clearing data, the number of columns to potentially clear [1-<#features>]')
        print("    -I  -    Don't impute values. Useful for getting the baseline accuracy of the classifier")
        print('    --seed - The random seed to use in clearing the data [Numeric]')
        print('    --skip-columns - Specify which columns should be ignored during the classification step (0-indexed)')
        print('    --no-sort - Tells the imputation algorithm to impute on the unordered set of missing values')
        print()
        print('Both the -r and -c parameters will accept a triplet of values specifying a range (as [start,end,step])')
        print('for the given parameter. The resultant range must still lie within the arguments\' bounds')
        print('The --skip-columns parameter accepts a comma-delimited set of columns')

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
            return res
        else:
            v = np.ndarray((1), dtype=dtype)
            v[0] = float(arg)
            return v

    def parse_seq_arg(arg, dtype):
        if ',' in arg:
            groups = re.findall(r"([-\d.]+)", arg)

            res = [dtype(x) for x in groups]
            myshow(res, "parse_seq_result")
            return res
        else:
            res = np.ndarray((1), dtype=dtype)
            res[0] = dtype(arg)
            return res

    try:
        opts, args = getopt.getopt(argv[1:], "hsi:r:c:l:I",
                                   ["help", "ifile=", "skip-first", "seed=", "skip-columns=", "--no-sort"])
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

        elif opt in ("-l"):
            class_column = int(arg)
            if debug:
                print("Class column: " + str(class_column))

        elif opt in ("-s", "--skip-first"):
            has_header = True
            if debug:
                print("Has header: " + str(has_header))

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
            r_seed = int(arg)
            if debug:
                print("seed: " + str(arg))

        elif opt in ("-I"):
            impute_data = False
            if debug:
                print("Impute: " + str(impute_data))

        elif opt in ("--skip-columns"):
            ignored_columns = parse_seq_arg(arg, int)
            if debug:
                print("Skipping columns: " + str(ignored_columns))

        elif opt in ("--no-sort"):
            use_sign = False
            if debug:
                print("Impute w Sort: " + str(use_sign))

    if filePath is None:
        get_help(argv)
        sys.exit(2)

    #### Environment Setup ####
    fileName = ntpath.basename(filePath)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    #### Read/Parse Data ####
    print()
    print('Reading from "' + filePath + '"')
    print()

    data, output, class_indices, classes = read_and_parse(filePath, class_column, ignored_columns, header=has_header,
                                                          delimiter=delimiter)

    # classes = np.unique(output, return_inverse=True)

    myshow(data, "data", maxlines=15)
    myshow(output, "output")
    if debug:
        myshow(classes, "classes")
        myshow(class_indices, "class_indices")

    statistics = []

    #### Process data ####
    for rrand in row_random_rate:
        for crand in col_random_rate:
            process_data(data, output, class_indices, classes, filePath, float(rrand), int(crand), impute_data, r_seed,
                         statistics, use_sign)

    print_statistics_plot(statistics, fileName)

    print("Done, exiting")


def process_data(data_source, _output, class_indices, classes, filePath, row_random_rate, col_random_rate, impute_data,
                 r_seed,
                 statistics, use_sign):
    manually_zero = row_random_rate > 0
    parameters = [["Input File", "Zero Data", "Row Random Rate", "Col Random Rate", "Rand Seed", "Impute"],
                  [filePath, manually_zero, row_random_rate, col_random_rate, r_seed, impute_data]]

    _data = data_source.copy()

    if manually_zero:
        _data = clear_data_random(_data, row_rate=row_random_rate, col_rate=col_random_rate, seed=r_seed)

    # if debug:
    #     myshow(_data, "data", maxlines=15)
    #     myshow(_output, "output")
    #     myshow(classes, "classes")
    #     myshow(class_indices, "class_indices")

    print("With Mean Imputation : ")
    perform_classification_with_mean_imputation(_data, classes, class_indices, parameters, statistics, row_random_rate,
                                                col_random_rate)

    processor = Imputation();
    print("With Knn Imputation : ")
    _data, meanChangeInValues = processor.estimate_values(_data, class_indices, impute_data, use_sign)

    # myshow(data, "imputed data", maxlines=15)
    # myshow(data - old_data, "difference", maxlines=15)

    perform_classification(_data, classes, class_indices, parameters, statistics, row_random_rate, col_random_rate)

    if impute_data:
        fileName = ntpath.basename(filePath)
        plot_save_meanvalues(fileName, meanChangeInValues)


def plot_save_meanvalues(fileName, meanChangeInValues):
    # Plot mean change in values
    plt.plot(meanChangeInValues, "-", linewidth=2.0)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Mean change in values', fontsize=12)
    # plt.title("Mean Change in Values")
    plt.savefig('output/MeanChangePlot_' + fileName + '.png')
    plt.close()

    print('', end='', flush=True)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    output_path = os.path.join(outputDir, "meanchangeresults.csv")
    new_file = not os.path.exists(output_path)

    headers = ("FileName", "Iteration", "Mean Change")
    with open(output_path, 'a') as f:
        if new_file:
            f.write(','.join(itertools.chain(headers)))
            f.write('\n')
        for k in range(len(meanChangeInValues)):
            f.write(','.join(str(x) for x in itertools.chain((fileName, k, meanChangeInValues[k]))))
            f.write('\n')
        f.flush()


def perform_classification(data, classes, class_indices, parameters, statistics, row_random_rate, col_random_rate):
    clf = svm.SVC()
    clf.fit(data, class_indices)
    print(clf._get_coef())

    # rows are actual, columns are predicted
    conf_mat = np.zeros((len(classes), len(classes)), dtype=np.int64)
    estimates = np.zeros(class_indices.shape, dtype=class_indices.dtype)
    for i in range(len(class_indices)):
        estimates[i] = clf.predict(np.reshape(data[i, :], (1, -1)))
        conf_mat[estimates[i], class_indices[i]] += 1

    # myshow(estimates, "estimates")
    analyze_errors(conf_mat, classes, parameters, statistics, row_random_rate, col_random_rate)


def perform_classification_with_mean_imputation(data, classes, class_indices, parameters, statistics, row_random_rate,
                                                col_random_rate):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data)
    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
    data_imputed = imp.transform(data)

    clf = svm.SVC()
    clf.fit(data_imputed, class_indices)
    print(clf._get_coef())

    # rows are actual, columns are predicted
    conf_mat = np.zeros((len(classes), len(classes)), dtype=np.int64)
    estimates = np.zeros(class_indices.shape, dtype=class_indices.dtype)
    for i in range(len(class_indices)):
        estimates[i] = clf.predict(np.reshape(data_imputed[i, :], (1, -1)))
        conf_mat[estimates[i], class_indices[i]] += 1

    # myshow(estimates, "estimates")
    analyze_errors(conf_mat, classes, parameters, statistics, row_random_rate, col_random_rate)


def analyze_errors(conf_mat, classes, parameters, statistics, row_random_rate, col_random_rate):
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

    statistics.append([row_random_rate, col_random_rate, accuracy, precision, recall, f_measure])


def save_errors(accuracy, recall, precision, f_measure, parameters):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    output_path = os.path.join(outputDir, "results.csv")
    new_file = not os.path.exists(output_path)

    headers = ("Accuracy", "Precision", "Recall", "F-Measure")
    with open(output_path, 'a') as f:
        # myshow(parameters, "Parameters")
        if new_file:
            f.write(','.join(itertools.chain(parameters[0], headers)))
            f.write('\n')
        f.write(','.join(str(x) for x in itertools.chain(parameters[1], (accuracy, precision, recall, f_measure))))
        f.write('\n')
        f.flush()


        # parameters
        # outputDir


def print_statistics_plot(statistics, fileName):
    row_vals = []
    col_vals = []
    for row in statistics:
        row_vals.append(row[0])
        col_vals.append(row[1])

    row_vals = np.unique(row_vals)
    col_vals = np.unique(col_vals)

    # myshow(row_vals)
    # myshow(col_vals)
    data = np.zeros((len(row_vals), len(col_vals), 4), dtype=np.float64)
    for row in statistics:
        r = np.where(row_vals == row[0])
        c = np.where(col_vals == row[1])
        data[r, c, 0:4] = row[2:6]

    fig, axes = plt.subplots(2, 2, figsize={16, 9}, subplot_kw={'xticks': [], 'yticks': []})
    for ax, indices, measurement in zip(axes.flat, range(0, 4, 1), ["Accuracy", "Precision", "Recall", "F-Measure"]):
        ax.imshow(1 - data[:, :, indices], cmap=plt.cm.jet, vmin=0., vmax=1.)
        ax.set_title(measurement)

    col_mult = math.ceil(len(col_vals) / 10)
    col_indices = np.arange(start=0, stop=len(col_vals), step=col_mult)

    row_mult = math.ceil(len(row_vals) / 11)
    row_indices = np.arange(start=0, stop=len(row_vals), step=row_mult)
    # myshow(row_indices)

    plt.setp(axes, xticks=col_indices, xticklabels=col_vals[col_indices], xlabel="Column Deletion Rate",
             yticks=row_indices, yticklabels=row_vals[row_indices], ylabel="Row Affected Rate")
    plt.tight_layout()

    # plt.show()
    plt.savefig(os.path.join(outputDir, "Estimation_statistics_plot-{0}.png".format(fileName)), bbox_inches='tight')


if __name__ == "__main__":
    main(sys.argv)
