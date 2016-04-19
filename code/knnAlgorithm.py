import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import operator
import csv
import random
import math
from myUtil import *
import warnings


def knnImputation(originalData, missing_indices, sorted_indices, impact_weight, n_classes):
    finalMeanValues = []
    if len(missing_indices) > 0:
        # Copy the data into another array which will contain the imputed data after each iteration.
        imputedData = originalData.copy()
        data = originalData.copy()
        changeinValues = 0.0
        oldMeanChange = 0.0

        k = 0
        max_iterations = 100
        for k in range(max_iterations):
            print("Iteration ", k, end=' ')
            for i in range(len(missing_indices)):
                CMIV_i = missing_indices[sorted_indices[i]]
                datapoint = data[CMIV_i[0]]

                nearest_n_neighbors = int(len(data) / n_classes)
                # nearest_n_neighbors =20
                neighbors, distances = getNearestNeighbors(data, datapoint, nearest_n_neighbors)

                # Imputed value is the weighted mean of the neighboring values weighted by the distance of the neighbors
                distanceWeights = np.ones(len(distances))
                maxDistance = max(distances)
                if maxDistance > 0:
                    for j in range(len(distances)):
                        distanceWeights[j] = 1 - (distances[j] / maxDistance)

                # neighbor_average = np.average(neighbors, axis=0)
                neighbor_average = np.average(neighbors, weights=distanceWeights, axis=0)
                imputedData[CMIV_i[0], CMIV_i[1]] = neighbor_average[CMIV_i[1]]

                # print("    Data[{0:3d}, {1:3d}]: {2:3.4f} => {3:3.4f}".format(
                #     CMIV_i[0],
                #     CMIV_i[1],
                #     data[CMIV_i[0], CMIV_i[1]],
                #     imputedData[CMIV_i[0], CMIV_i[1]]
                # ))
                changeinValues += abs(data[CMIV_i[0], CMIV_i[1]] - imputedData[CMIV_i[0], CMIV_i[1]])

            meanChange = changeinValues / len(missing_indices)
            print("Mean change in filled in values : ", meanChange)

            # Copy new data for next iteration
            data = imputedData.copy()
            meanDiff = abs(oldMeanChange - meanChange)
            print("Mean Difference : ", meanDiff)
            finalMeanValues.insert(k, meanDiff)
            if k > 1 and meanDiff <= 0.001:
                break;
            oldMeanChange = meanChange

        print('Total number of iterations to convergence : ' + str(k), flush=True)

        return data, finalMeanValues
    else:
        return originalData, finalMeanValues


def loadDataset(filename, split, train=[], test=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                train.append(dataset[x])
            else:
                test.append(dataset[x])


def euclideanDistance(x1, x2, length):
    distance = 0
    for x in range(length):
        distance += pow((x1[x] - x2[x]), 2)
    return math.sqrt(distance)


def getNearestNeighbors(data, datapoint, k):
    distances = []
    length = len(datapoint)
    for x in range(len(data)):
        dist = euclideanDistance(datapoint, data[x], length)
        distances.append((data[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    edist = []
    for x in range(k):
        neighbors.append(distances[x][0])
        edist.append(distances[x][1])
    return neighbors, edist


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def knnClassification(data, output, state):
    class_values = np.unique(output)
    n_neighbors = len(data) / len(class_values)

    # data = datasets.load_iris()
    X = data[:, :2]  # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    y = output

    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors)


    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 200),
                         np.arange(y_min, y_max, (y_max - y_min) / 200))

    with warnings.catch_warnings():  # https://docs.python.org/2/library/warnings.html#temporarily-suppressing-warnings
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", DeprecationWarning)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # myshow(np.c_[xx.ravel(), yy.ravel()], "np.c_")
    # print("Classification Accuracy : ", getAccuracy(y, Z))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification")

    plt.savefig('output/classification_' + str(state) + '.png')
    plt.close()
    print('', end='', flush=True)
