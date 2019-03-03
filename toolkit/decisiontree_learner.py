from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import math
import copy


class DecisionTreeLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    class Node:
        features = []
        labels = []
        isOutput = False
        output_class = None
        children = []
        info = 0
        split_on = []
        current_split = -1
        learner = None

        def __init__(self, features, labels, splits, treeLearner):
            self.features = features
            self.labels = labels
            self.split_on = splits
            self.learner = treeLearner
            self.output_class = labels.most_common_value(0)
            self.calc_info()
            if self.info == 0:
                self.isOutput = True

        def calc_info(self):
            # print(self.learner.num_outputs)
            output_split = np.zeros(self.learner.num_outputs)
            # print(output_split)
            for i in range(len(self.features.data)):
                output_split[int(self.labels.data[i][0])] += 1
            for entry in output_split:
                frac = entry/len(self.labels.data)
                if frac != 0:
                    self.info += -frac * math.log(frac, 2)
            print("Info: ", self.info)
            return self.info

        def calc_info_subset(self, split_feature):
            num_categories = self.features.value_count(split_feature)
            print("num Categories: ", num_categories)
            output_split = [([0] * (self.learner.num_outputs + 1)) for t in range(num_categories)]
            #print(output_split)
            #print("number of features: ", len(self.features.data))
            for i in range(len(self.features.data)):
                #print("feature index: ", self.features.data[i][split_feature])
                output_split[int(self.features.data[i][split_feature])][0] += 1
                #print(output_split)
                #print("label index: ", self.labels.data[i][0])
                output_split[int(self.features.data[i][split_feature])][int(self.labels.data[i][0]) + 1] += 1
                #print(output_split)
            print(output_split)
            split_info = 0
            for entry in output_split:
                sub_info = 0
                for j in range(1, len(entry)):
                    if entry[0] != 0 and (entry[j]/entry[0]) != 0:
                        sub_info += (entry[j]/entry[0]) * math.log(entry[j]/entry[0], 2)
                split_info += -(entry[0]/len(self.features.data)) * sub_info
            print("Info for ", split_feature, ": ", split_info)
            return split_info

        def build_child_nodes(self, split_feature):
            num_categories = self.features.value_count(split_feature)
            new_sets = [None] * num_categories
            new_labels = [None] * num_categories
            new_split = copy.deepcopy(self.split_on)
            new_split.append(split_feature)
            for i in range(len(self.features.data)):
                f = int(self.features.data[i][split_feature])
                if new_sets[f] is None:
                    new_sets[f] = Matrix(self.features, i, 0, 1, self.features.cols)
                    new_labels[f] = Matrix(self.labels, i, 0, 1, self.labels.cols)
                else:
                    new_sets[f].data.append(self.features.data[i])
                    new_labels[f].data.append(self.labels.data[i])
            children = []
            for i in range(len(new_sets)):
                new_sets[i].print()
                new_labels[i].print()
                children.append(DecisionTreeLearner.Node(new_sets[i], new_labels[i], new_split, self.learner))
            return children

        def split_tree(self):
            best_gain = 0
            best_feature = -1
            for i in range(self.features.cols):
                if i not in self.split_on:
                    print("finding gain on ", i)
                    gain = self.info - self.calc_info_subset(i)
                    print("gain of ", i, ": ", gain)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = i
            if best_gain == 0 or best_feature == -1 or self.info == 0:
                self.isOutput = True
                print("found leaf node")
            else:
                print("Splitting on ", best_feature)
                self.current_split = best_feature
                self.children = self.build_child_nodes(int(best_feature))
                for child in self.children:
                    child.split_tree()

        def print_node(self):
            print("Node")
            print("split on: ", self.split_on)
            print("info: ", self.info)
            if self.isOutput:
                print("Leaf Node: ", self.output_class)
            else:
                print("Children: ")
                for child in self.children:
                    child.print_node()

        def traverse_tree(self, predict_features):
            if self.isOutput:
                return self.output_class
            else:
                return self.children[int(predict_features[self.current_split])].traverse_tree(predict_features)

    labels = []
    num_outputs = 0
    root = None

    def __init__(self):

        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.num_outputs = labels.value_count(0)
        self.labels = labels
        self.root = DecisionTreeLearner.Node(features, labels, [], self)
        self.root.split_tree()
        print("\nTree:")
        self.root.print_node()

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        print("predict feat: ", features)
        labels.append(self.root.traverse_tree(features))
        print("predict out", labels)
