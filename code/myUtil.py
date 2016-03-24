import sys, getopt, ntpath, os
import numpy as np
import math


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


# def myshow(array, name='', maxlines=5):
#     prefix = ""
#     if name != '':
#         fLine = name + " = "
#         print(fLine, end='')
#         prefix = " " * max(len(fLine), 8)
#
#     print(str(array.shape) + "    [dtype: " + str(array.dtype) + "]")
#     l = 0;
#     for line in str(array).split('\n'):
#         print(prefix + line)
#         l = l + 1
#         if l > maxlines:
#             print(prefix + " ...")
#             break


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
            l = l + 1
            if l > maxlines:
                print(prefix + " ...")
                break
    else:
        print(repr(array))

    print()

def memory():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def printMem():
    print("    Memory: " + str(memory() / 1024 / 1024) + " MB")
