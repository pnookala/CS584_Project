import sys, getopt, ntpath, os
import numpy as np
import math

dataType = np.float64


def read_and_parse(filepath, class_column=None, ignored_columns=None, header=False, delimiter=','):
    content = np.genfromtxt(filepath, dtype=None, delimiter=delimiter, skip_header=header)

    N = content.shape[0]
    M = len(content[0]) - 1
    data_columns = [i for i in range(M + 1)]

    if class_column == None:
        class_column = M

    if not ignored_columns is None:
        M -= len(ignored_columns)

    data_columns.remove(class_column)
    if not ignored_columns is None:
        for c in ignored_columns:
            data_columns.remove(c)

    try:
        class_type = content[0].dtype[class_column]
    except:
        class_type = content[0].dtype

    data = np.ndarray((N, M), dtype=dataType)
    output = np.ndarray((N, 1), dtype=class_type)

    i = 0
    for d in content:
        k = 0
        for j in data_columns:
            try:
                if str(d[j]) == "b'y'":
                    v = 1
                elif str(d[j]) == "b'n'":
                    v = 0
                else:
                    v = float(d[j])
            except ValueError:
                v = math.nan
            data[i, k] = v
            k += 1
        output[i] = d[class_column]
        i += 1

    classes = np.unique(output)
    class_indices = np.argwhere(output == classes)[:, 1]  # Create indices to the "classes" array for each row
    return data, output, class_indices, classes


def get_name_without_ext(name):
    try:
        i = name.rindex('.')
    except ValueError:
        i = len(name)
    return name[0:i]


def print_confusion_matrix(mat, labels, number_width=4, number_precision=0, title=''):
    row_title = "predicted "
    col_title = "actual"

    # Data calculations
    num_labels = len(labels)
    total = 0
    for i in range(num_labels):
        for j in range(num_labels):
            total += mat[i, j]
    summary_text = "Sum: " + str(total)

    # Calculate column widths and formats
    labelWidth = 0
    for l in labels:
        if labelWidth < len(l):
            labelWidth = len(l)

    if labelWidth < len(summary_text):
        labelWidth = len(summary_text)

    colWidth = max(labelWidth + 1, number_width, number_precision + 2)
    print(title)

    row_title_width = len(row_title)

    # label_format = "%" + str(labelWidth) + "s"
    label_format = "{:" + str(labelWidth) + "s}"
    header_format = "{:" + str(colWidth) + "s}"
    content_format = "{:" + str(colWidth) + "." + str(number_precision) + "f}"
    row_title_format = "{:" + str(row_title_width) + "s}"

    # Header part
    print(' ' * (row_title_width +
                 labelWidth + 1 +
                 int(colWidth * (-1 + num_labels / 2))), end=' ')
    print(col_title)

    print(' ' * row_title_width, end=' ')
    print(label_format.format(summary_text), end='|')
    for i in range(num_labels):
        print(header_format.format(str(labels[i])), end='|')
    print()

    # Body
    for i in range(num_labels + 1):
        # Room for "predicted" label
        if i == math.floor(num_labels / 2):
            print(row_title_format.format(row_title), end=' ')
        else:
            print(' ' * row_title_width, end=' ')

        # Line Separator
        print('-' * labelWidth, end='+')
        for j in range(num_labels):
            print('-' * colWidth, end='+')
        print('-' * colWidth, end='')
        print()

        if i >= num_labels:
            break

        # Room for "predicted" label
        print(' ' * row_title_width, end=' ')

        # Values
        print(label_format.format(str(labels[i])), end='|')
        for j in range(num_labels):
            print(content_format.format(mat[i, j]), end='|')
        # Sum of col values
        print(content_format.format(np.sum(mat[i, :])), end='')
        print()

    # Sum of row values (after label space)
    print(' ' * row_title_width, end=' ')
    # Values
    print(label_format.format(' '), end='|')
    for j in range(num_labels):
        print(content_format.format(np.sum(mat[:, j])), end='|')
    print()

    # Done!
    print()


def CalculateStatistics(conf_matrix):
    # print_confusion_matrix(conf_matrix, ["class " + str(x) for x in range(conf_matrix.shape[0])])

    num_classes = conf_matrix.shape[0]
    accuracy = np.sum(conf_matrix[range(num_classes), range(num_classes)]) / np.sum(conf_matrix)

    recalls = []
    for i in range(num_classes):
        s = np.sum(conf_matrix[range(num_classes), i])
        if s > 0:
            recalls.append(conf_matrix[i, i] / s)
        else:
            recalls.append(0)
    recall = np.average(recalls)

    precisions = []
    for i in range(num_classes):
        s = np.sum(conf_matrix[i, range(num_classes)])
        if s > 0:
            precisions.append(conf_matrix[i, i] / s)
        else:
            precisions.append(0)
    precision = np.average(precisions)

    f_measure = (2 * precision * recall) / (precision + recall)

    return accuracy, recall, precision, f_measure


# Just writes a few stats about the error to disk
def saveErrorReport(fName, line_array, header_array):
    noFileExists = not os.path.exists(fName)
    ef = open(fName, 'a')
    if noFileExists:
        ef.write(','.join(header_array))
        ef.write("\n")

    ef.write(','.join(
        # formatAsString(x)
        str(x)
        for x in line_array
    ))
    ef.write("\n")

    ef.close()


def formatAsString(x):
    if type(x) == str:
        return x
    if type(x) == int:
        return str(x)
    if (type(x) == float) | (type(x) == np.float64):
        return '{:.5f}'.format(x)
    return "({0} '{1}')".format(type(x), str(x))


def myshow(array, name='', maxlines=5):
    fLine = "'{:s}' ".format(name) if name != '' else ''
    fLine = "{:s}{:s}:".format(fLine, str(type(array)))
    print(fLine)

    if type(array) == np.ndarray:
        print("[{:s}] <{:s}>".format(", ".join(str(x) for x in array.shape), str(array.dtype)))
        # print("shape: " + str(array.shape))
        # print("dtype: " + str(array.dtype))
        l = 0;
        prefix = " " * 6
        for line in str(array).split('\n'):
            print(prefix + line)
            if l > maxlines:
                print(prefix + " ...")
                break
            l += 1

        try:
            print(prefix + "Sum:     " + str(np.sum(array)))
            print(prefix + "Min/Max: " + str(np.min(array)) + "/" + str(np.max(array)))
            print(prefix + "Average: " + str(np.average(array)))
        except:
            ''
            # else not a numeric range, skip

    else:
        print(repr(array))

    print(flush=True)


def memory():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def printMem():
    print("    Memory: " + str(memory() / 1024 / 1024) + " MB")
