from __future__ import (absolute_import, division, print_function, unicode_literals)

from .matrix import Matrix
import math
import numpy as np

# this is an abstract class


class SupervisedLearner:

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        Before you call this method, you need to divide your data
        into a feature matrix and a label matrix.
        :type features: Matrix
        :type labels: Matrix
        """
        raise NotImplementedError()

    def predict(self, features, labels):
        """
        A feature vector goes in. A label vector comes out. (Some supervised
        learning algorithms only support one-dimensional label vectors. Some
        support multi-dimensional label vectors.)
        :type features: [float]
        :type labels: [float]
        """
        raise NotImplementedError

    def measure_accuracy(self, features, labels, confusion=None):
        """
        The model must be trained before you call this method. If the label is nominal,
        it returns the predictive accuracy. If the label is continuous, it returns
        the root mean squared error (RMSE). If confusion is non-NULL, and the
        output label is nominal, then confusion will hold stats for a confusion matrix.
        :type features: Matrix
        :type labels: Matrix
        :type confusion: Matrix
        :rtype float
        """

        if features.rows != labels.rows:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.cols != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.rows == 0:
            raise Exception("Expected at least one row")

        label_values_count = labels.value_count(0)
        if label_values_count == 0:
            # label is continuous
            pred = []
            sse = 0.0
            for i in range(features.rows):
                feat = features.row(i)
                targ = labels.row(i)
                pred.append(0.0)       # make sure the prediction is not biased by a previous prediction
                self.predict(feat, pred)
                delta = targ[0] - pred[0]
                sse += delta**2
            # return sse / features.rows    # for mse instead of rmse
            return math.sqrt(sse / features.rows)

        else:
            # label is nominal, so measure predictive accuracy
            if confusion:
                confusion.set_size(label_values_count, label_values_count)
                confusion.attr_names = [labels.attr_value(0, i) for i in range(label_values_count)]

            correct_count = 0
            prediction = []
            for i in range(features.rows):
                feat = features.row(i)
                targ = int(labels.get(i, 0))
                if targ >= label_values_count:
                    raise Exception("The label is out of range")
                self.predict(feat, prediction)
                pred = int(prediction[0])
                # print(targ, pred)
                if confusion:
                    confusion.set(targ, pred, confusion.get(targ, pred)+1)
                if pred == targ:
                    correct_count += 1
            return correct_count / features.rows

            running_sse = 0
            for i in range(features.rows):
                feat = features.row(i)
                targ = int(labels.get(i, 0))
                if targ >= label_values_count:
                    raise Exception("The label is out of range")
                self.predict(feat, prediction)
                converted_target, converted_prediction = self.convert_labels(labels, i, prediction)
                print(converted_target, converted_prediction)
                running_sse += self.calc_sum_square_error(converted_target, converted_prediction)
                #print(running_sse)
                pred = int(prediction[0])
                # pred = prediction.index(max(prediction[0]))
                if confusion:
                    confusion.set(targ, pred, confusion.get(targ, pred)+1)
                if pred == targ:
                    correct_count += 1
            #print("test mse: ", (running_sse / features.rows))
            return running_sse / features.rows

    def calc_sum_square_error(self, target, output):
        sum = 0
        for i in range(len(target)):
            sum += (target[i] - output[i]) ** 2
        print(sum)
        return sum

    def convert_labels(self, labels, row_index, result):
        if labels.value_count(0) <= 2:
            return labels.data[row_index]
        converted_labels = np.zeros(labels.value_count(0))
        converted_results = np.zeros(labels.value_count(0))
        col = int(labels.data[row_index][0])
        converted_labels[col] = 1
        converted_results[int(result[0])] = 1
        return converted_labels, converted_results
