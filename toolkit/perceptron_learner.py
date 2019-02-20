from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import copy
import csv

class PerceptronLearner(SupervisedLearner):
    """
    This model works to categorize nominal data.
    It can categorize into two or more categories
    """

    weights = []
    learning_rate = .1

    def __init__(self):
        pass

    # calculates the net output for the specified row for the perceptron
    def calculate_net(self, input, percep):
        net = 0
        # add the net for each input
        for c in range(len(input)):
            net = net + (input[c] * self.weights[percep][c])
        return net

    def update_weights(self, input, output, target, percep):
        change = np.zeros(len(input))
        # calculate the change in weight for each input
        for w in range(len(input)):
            change[w] = (target - output) * self.learning_rate * input[w]
        # update weight vector
        # print(change)
        np.add(self.weights[percep], change, self.weights[percep])

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # set up initial weight matrix, with a column for each perceptron to be trained
        self.weights = np.zeros((labels.value_count(0), features.cols + 1))
        total_epochs = 0

        graph = []

        for percep in range(labels.value_count(0)):
            print("training perceptron", percep)
            # for each possible output, train a perceptron for it
            improving = True
            prev_accuracy = 0.5
            epochs = 5
            # while there is improvement, or there has not been 5 epochs sense improvement
            while improving or epochs > 0:
                total_epochs = total_epochs + 1
                # print("epoch")
                # shuffle data
                features.shuffle(labels)
                # for each input vector/row
                for row_index in range(features.rows):
                    input = copy.deepcopy(features.data[row_index])
                    input.append(1)
                    # calculate the net output fot that row, for that perceptron
                    net = self.calculate_net(input, percep)
                    # translate net to output
                    output = 0
                    if net > 0:
                        output = 1
                    # calculate target for current perceptron and row
                    target = 0
                    if percep == labels.data[row_index][0]:
                        target = 1
                    # update weight's if output does not match target
                    if output != target:
                        self.update_weights(input, output, target, percep)

                # calculate the perceptron's current accuracy after each epoch
                accuracy = self.measure_accuracy(features, labels)
                graph.append(1 - accuracy)
                # print(total_epochs, 1-accuracy)
                # determine if additional epoch's are needed
                if accuracy > prev_accuracy:
                    improving = True
                    epochs = 5
                    prev_accuracy = accuracy
                else:
                    improving = False
                    epochs = epochs - 1
        with open("graph.csv", "wb") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(graph)
        print(graph)
        print("Total Epochs:", total_epochs)
        print("final weights:\n", self.weights[percep])

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        # make sure labels are clear
        del labels[:]
        best_percep = 0
        best_net = -100
        # run features on each perceptron
        for percep in range(len(self.weights)):
            net = self.calculate_net(features, percep)
            if net > best_net:
                best_net = net
                best_percep = percep
        # set the label to the best solution found
        labels += [best_percep]
        # print("prediction:", features, labels)
