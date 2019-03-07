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

        def __init__(self, features, labels, splits, treeLearner, parentNode=None):
            self.features = features
            self.labels = labels
            self.split_on = splits
            self.learner = treeLearner
            if labels is not None:
                self.output_class = labels.most_common_value(0)
                self.calc_info()
                if self.info == 0:
                    self.isOutput = True
            else:
                self.output_class = parentNode.output_class
                self.info = 0
                self.isOutput = True

        def calc_info(self):
            # print(self.learner.num_outputs)
            if self.features is None:
                self.info = 0
                return self.info
            output_split = np.zeros(self.learner.num_outputs)
            # print(output_split)
            for i in range(len(self.features.data)):
                output_split[int(self.labels.data[i][0])] += 1
            for entry in output_split:
                frac = entry/len(self.labels.data)
                if frac != 0:
                    self.info += -frac * math.log(frac, 2)
            # print("Info: ", self.info)
            return self.info

        def calc_info_subset(self, split_feature):
            num_categories = self.features.value_count(split_feature)
            # print("num Categories: ", num_categories)
            output_split = [([0] * (self.learner.num_outputs + 1)) for t in range(num_categories + 1)] # +1 for unknowns
            # print(output_split)
            # print("number of features: ", len(self.features.data))
            for i in range(len(self.features.data)):
                # print("feature index: ", self.features.data[i][split_feature])
                if self.features.data[i][split_feature] > num_categories:
                    # this is an unknown
                    category = num_categories
                else:
                    category = int(self.features.data[i][split_feature])
                output_split[category][0] += 1
                # print(output_split)
                # print("label index: ", self.labels.data[i][0])
                output_split[category][int(self.labels.data[i][0]) + 1] += 1
                # print(output_split)
            # print(output_split)
            split_info = 0
            for entry in output_split:
                sub_info = 0
                for j in range(1, len(entry)):
                    if entry[0] != 0 and (entry[j]/entry[0]) != 0:
                        sub_info += (entry[j]/entry[0]) * math.log(entry[j]/entry[0], 2)
                split_info += -(entry[0]/len(self.features.data)) * sub_info
            # print("Info for ", split_feature, ": ", split_info)
            return split_info

        def calc_laplacians(self, split_feature):
            num_categories = self.features.value_count(split_feature)
            output_split = [([0] * (self.learner.num_outputs + 1)) for t in range(num_categories + 1)]  # +1 for unknowns
            for i in range(len(self.features.data)):
                if self.features.data[i][split_feature] > num_categories:
                    # this is an unknown
                    category = num_categories
                else:
                    category = int(self.features.data[i][split_feature])
                output_split[category][0] += 1
                output_split[category][int(self.labels.data[i][0]) + 1] += 1

            split_info = 0
            for entry in output_split:
                for j in range(1, len(entry)):
                    if entry[0] != 0 and (entry[j] / entry[0]) != 0:
                        split_info += (entry[j] + 1 / (entry[0] + len(self.learner.features.data))) * (entry[0]/len(self.features.data))
            return split_info

        def build_child_nodes(self, split_feature):
            num_categories = self.features.value_count(split_feature)
            # print(num_categories)
            new_sets = [None] * (num_categories+1)
            new_labels = [None] * (num_categories+1)
            new_split = copy.deepcopy(self.split_on)
            new_split.append(split_feature)
            for i in range(len(self.features.data)):
                if self.features.data[i][split_feature] > num_categories:
                    f = num_categories
                else:
                    f = int(self.features.data[i][split_feature])
                if new_sets[f] is None:
                    new_sets[f] = Matrix(self.features, i, 0, 1, self.features.cols)
                    new_labels[f] = Matrix(self.labels, i, 0, 1, self.labels.cols)
                else:
                    new_sets[f].data.append(self.features.data[i])
                    new_labels[f].data.append(self.labels.data[i])
            children = []
            for i in range(len(new_sets)):
                if new_sets[i] is None:
                    children.append(DecisionTreeLearner.Node(None, None, new_split, self.learner, self))
                else:
                    # new_sets[i].print()
                    # new_labels[i].print()
                    children.append(DecisionTreeLearner.Node(new_sets[i], new_labels[i], new_split, self.learner, self))
            return children

        def split_tree_laplacian(self):
            best_gain = 0
            best_feature = -1
            for i in range(self.features.cols):
                if i not in self.split_on:
                    lap = self.calc_laplacians(i)
                    if lap > best_gain:
                        best_gain = lap
                        best_feature = i
            if best_feature == -1 or self.info == 0:
                self.isOutput = True
                # print("found leaf node")
            else:
                # print("Splitting on ", best_feature)
                self.current_split = best_feature
                self.children = self.build_child_nodes(int(best_feature))
                for child in self.children:
                    if not child.isOutput:
                        child.split_tree()

        def split_tree(self):
            best_gain = 0
            best_feature = -1
            for i in range(self.features.cols):
                if i not in self.split_on:
                    # print("finding gain on ", i)
                    gain = self.info - self.calc_info_subset(i)
                    # print("gain of ", i, ": ", gain)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = i
            if best_gain == 0 or best_feature == -1 or self.info == 0:
                self.isOutput = True
                # print("found leaf node")
            else:
                # print("Splitting on ", best_feature)
                self.current_split = best_feature
                self.children = self.build_child_nodes(int(best_feature))
                for child in self.children:
                    if not child.isOutput:
                        child.split_tree()

        def print_node(self, level):
            #print("Node: level ", level)
            #print("info: ", self.info)
            if self.isOutput:
                #print("Leaf Node: ", self.output_class)
                return 1, level
            else:
                #print("split on: ", self.features.attr_name(self.current_split))
                node_count = 1
                best_level = level
                #print("Children: ")
                for child in self.children:
                    sub_node_count, sub_level = child.print_node(level + 1)
                    node_count += sub_node_count
                    if sub_level > best_level:
                        best_level = sub_level
                return node_count, best_level

        def traverse_tree(self, predict_features):
            if self.isOutput:
                return self.output_class
            else:
                if predict_features[self.current_split] > len(self.children):
                    return self.children[-1].traverse_tree(predict_features)
                return self.children[int(predict_features[self.current_split])].traverse_tree(predict_features)

    labels = []
    features = []
    num_outputs = 0
    root = None
    node_count = 0
    leaf_count = 0
    max_level = 0

    def __init__(self):
        pass

    def prune_tree(self, validation_features, validation_labels):
        original_tree = copy.deepcopy(self.root)
        original_acc = SupervisedLearner.measure_accuracy(self, validation_features, validation_labels)
        best_acc = original_acc
        best_tree = self.root
        current_node = self.root
        node_queue = list()
        node_queue.append(self.root)
        while len(node_queue) > 0:
            while len(node_queue) > 0:
                node = node_queue.pop()
                if not node.isOutput:
                    for child in node.children:
                        node_queue.append(child)
                    node.isOutput = True
                    acc = SupervisedLearner.measure_accuracy(self, validation_features, validation_labels)
                    if acc > best_acc:
                        best_acc = acc
                        best_tree = copy.deepcopy(self.root)
                    node.isOutput = False
            if best_acc > original_acc:
                original_acc = best_acc
                self.root = best_tree
                node_queue.append(self.root)
        self.root = best_tree

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        features.shuffle(labels)
        original_features = copy.deepcopy(features)
        original_labels = copy.deepcopy(labels)
        val_size = int(features.rows / 3)
        uncropped_validation_features = Matrix(features, 0, 0, val_size, features.cols)
        validation_features = Matrix(features, 0, 0, val_size, features.cols)
        validation_labels = Matrix(labels, 0, 0, val_size, labels.cols)
        train_features = Matrix(features, val_size, 0, features.rows - val_size, features.cols)
        train_labels = Matrix(labels, val_size, 0, labels.rows - val_size, labels.cols)
        self.features = train_features

        self.num_outputs = labels.value_count(0)
        self.labels = labels
        self.root = DecisionTreeLearner.Node(train_features, train_labels, [], self)
        self.root.split_tree()
        # print("\nTree:")

        self.node_count, self.max_level = self.root.print_node(0)
        train_acc = SupervisedLearner.measure_accuracy(self, train_features, train_labels)
        print("\ntraining accuracy ", train_acc)
        val_acc = SupervisedLearner.measure_accuracy(self, validation_features, validation_labels)
        print("origianal validation accuracy ", val_acc)
        print("Original Nodes: ", self.node_count)
        print("original levels: ", self.max_level)
        self.prune_tree(validation_features, validation_labels)
        val_acc = SupervisedLearner.measure_accuracy(self, validation_features, validation_labels)
        print("pruned validation accuracy ", val_acc)
        self.node_count = 0
        self.max_level = 0
        self.node_count, self.max_level = self.root.print_node(0)
        print("pruned Nodes: ", self.node_count)
        print("pruned levels: ", self.max_level)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        # print("predict feat: ", features)
        labels.append(self.root.traverse_tree(features))
        # print("predict out", labels)
