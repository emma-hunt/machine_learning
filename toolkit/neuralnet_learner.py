from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import csv
import numpy as np
import copy


class NeuralNetLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    labels = []
    network = []
    c = 0.1 # learning rate
    a = 0.8 # alpha for momentum
    num_hidden_layers = 1
    num_nodes_per_layer = 4

    running_test_error = 0

    class Node:
        weights = []
        prev_change_weight = []
        net = 0
        isOutput = False
        activation = 0
        output = 0
        error = 0

        def __init__(self, num_inputs, is_output):
            self.weights = np.random.uniform(-1.0, 1.0, num_inputs + 1)
            self.prev_change_weight = np.zeros(num_inputs + 1)
            self.isOutput = is_output

        def set_weights(self, weight):
            self.weights = weight

        def calculate_net(self, inputs):
            self.net = 0
            #print(inputs)
            #print(self.weights)
            for i in range(len(inputs)):
                self.net = self.net + inputs[i] * self.weights[i]
            return self.net

        def calc_output(self, inputs):
            self.calculate_net(inputs)
            # print("Input", input)
            # print("Net", self.net)
            self.activation = 1 / (1 + np.exp(-self.net))
            # print("Activation", self.activation)
            if self.net > 0:
                self.output = 1
            else:
                self.output = 0
            # print("act: ", self.activation)
            return self.activation

        def set_error(self, error):
            self.error = error

        def update_weights(self, weight_change):
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] + weight_change[i] + (NeuralNetLearner.a * self.prev_change_weight[i])
            self.prev_change_weight = weight_change

    def __init__(self):
        pass

    # helper functions
    def calc_output_error(self, target, act):
        return (target - act) * act * (1 - act)

    def calc_hidden_error(self, prev_error, weights, act):
        error = act * (1 - act)
        sum = 0
        for k in range(len(prev_error)):
            sum = sum + (prev_error[k] * weights[k])
        return error * sum

    def calc_change_weights(self, inputs, error):
        # print("in calc weights")
        change = []
        for i in inputs:
            change.append(self.c * i * error)
        # print("change:", change)
        return change

    def convert_labels(self, labels, row_index):
        if labels.value_count(0) <= 2:
            return labels.data[row_index]
        converted = np.zeros(labels.value_count(0))
        #print(labels.data[row_index][0])
        col = int(labels.data[row_index][0])
        converted[col] = 1
        #print(converted)
        return converted

    def print_network(self):
        i = 0
        for layer in self.network:
            print("layer ", i)
            i = i + 1
            for n in layer:
                print(n.weights, n.isOutput)

    def calculate_output(self, features):
        inputs = copy.deepcopy(features)
        # np.insert(inputs, 0, 1)
        next_inputs = []
        for layer in self.network:
            # print("calculate input: ", inputs)
            for node in layer:
                n = node.calc_output(inputs)
                next_inputs.append(n)
            next_inputs.insert(0, 1)
            # print(next_inputs)
            inputs = next_inputs
            next_inputs = []
        output = []
        # print("Calculated output: ", inputs)
        for entry in range(1, len(inputs)):
            if entry > 0:
                output.append(1)
            else:
                output.append(0)
        return inputs[1:]

    def calc_sum_square_error(self, target, output):
        sum = 0
        for i in range(len(target)):
            sum += (target[i] - output[i]) ** 2
        return sum

    def test_back_propagate(self, labels):
        converted_labels = labels
        for layer_index in range(len(self.network)):
            layer_index = len(self.network) - 1 - layer_index
            # print("layer: ", layer_index)
            layer = self.network[layer_index]
            for n in range(len(layer)):
                # print("node: ", n)
                node = layer[n]
                # print(node.weights, node.isOutput)
                error = 0
                if node.isOutput:
                    error = self.calc_output_error(converted_labels[n], node.activation)
                else:
                    assert not node.isOutput
                    # print("hidden node!")
                    prev_error = []
                    prev_weights = []
                    for prev in self.network[layer_index + 1]:
                        prev_error.append(prev.error)
                        prev_weights.append(prev.weights[n + 1])
                    # print("PR: ", prev_error, " PW: ", prev_weights)
                    error = self.calc_hidden_error(prev_error, prev_weights, node.activation)
                #print(layer_index, n, "error:", error)
                node.set_error(error)

    def back_propagate(self, labels, row_index):
        converted_labels = self.convert_labels(labels, row_index)
        for layer_index in range(len(self.network)):
            layer_index = len(self.network) - 1 - layer_index
            # print("layer: ", layer_index)
            layer = self.network[layer_index]
            for n in range(len(layer)):
                # print("node: ", n)
                node = layer[n]
                # print(node.weights, node.isOutput)
                error = 0
                if node.isOutput:
                    # print("an output node!")
                    #if node.activation > 0:
                     #   error = self.calc_output_error(converted_labels[n], 1)
                    #else:
                     #   error = self.calc_output_error(converted_labels[n], 0)
                    error = self.calc_output_error(converted_labels[n], node.activation)
                else:
                    assert not node.isOutput
                    # print("hidden node!")
                    prev_error = []
                    prev_weights = []
                    for prev in self.network[layer_index + 1]:
                        prev_error.append(prev.error)
                        prev_weights.append(prev.weights[n + 1])
                    # print("PR: ", prev_error, " PW: ", prev_weights)
                    error = self.calc_hidden_error(prev_error, prev_weights, node.activation)
                #print(layer_index, n, "error:", error)
                node.set_error(error)

    def update_weights(self, inputs):
        for layer_index in range(len(self.network)):
            layer = self.network[layer_index]
            z = []
            if layer_index == 0:
                z = inputs
            else:
                z = [1]
                for prev in self.network[layer_index - 1]:
                    z.append(prev.activation)
            # print("prev out: ", z)
            for current_node in layer:
                # print("for layer ", layer_index, " node ", i)
                # print("error: ", current_node.error)
                current_node.update_weights(self.calc_change_weights(z, current_node.error))
                #print("Weight: ", current_node.weights)

    def build_test_network(self, features, labels):
        print("building test network")
        layer0 = [NeuralNetLearner.Node(2, False), NeuralNetLearner.Node(2, False)]
        layer0[0].set_weights([0.1, 0.2, -0.1])
        layer0[1].set_weights([-0.2, 0.3, -0.3])
        self.network.append(layer0)
        layer1 = [NeuralNetLearner.Node(2, False), NeuralNetLearner.Node(2, False)]
        layer1[0].set_weights([0.1, -0.2, -0.3])
        layer1[1].set_weights([0.2, -0.1, 0.3])
        self.network.append(layer1)
        layer2 = [NeuralNetLearner.Node(2, True), NeuralNetLearner.Node(2, True)]
        layer2[0].set_weights([0.2, -0.1, 0.3])
        layer2[1].set_weights([0.1, -0.2, -0.3])
        self.network.append(layer2)
        self.print_network()

    def build_network(self, features, labels):
        # build network
        self.num_nodes_per_layer = 2 * features.cols
        #print("node per layer: ", self.num_nodes_per_layer)
        #print("hidden: ", self.num_hidden_layers)
        #print("output node: ", labels.value_count(0))
        # input layers
        for h in range(self.num_hidden_layers):
            layer = []
            if h == 0:
                # first layer, takes inputs from features
                for j in range(self.num_nodes_per_layer):
                    layer.append(NeuralNetLearner.Node(features.cols, False))
            else:
                # additional hidden layers take inputs for each node in prev layer
                for j in range(self.num_nodes_per_layer):
                    layer.append(NeuralNetLearner.Node(self.num_nodes_per_layer, False))
            self.network.append(layer)
        output_layer = []
        if labels.value_count(0) <= 2:
            output_layer.append(NeuralNetLearner.Node(self.num_nodes_per_layer, True))
        else:
            for o in range(labels.value_count(0)):
                output_layer.append(NeuralNetLearner.Node(self.num_nodes_per_layer, True))
        self.network.append(output_layer)

    def test_function(self):
        features = np.array([0.3, 0.7])
        labels = np.array([0.1, 1.0])
        self.build_test_network(features, labels)
        epochs = 0
        while epochs < 3:
            print("epoch: ", epochs)
            # train
            epochs = epochs + 1
            # for each input vector/row
            for row_index in range(1):
                input = np.insert(features, 0, 1)
                print("forward propagating...")
                output = self.calculate_output(input)
                print("predicted output: ", output)
                # calculate error
                print("back propagating...")
                self.test_back_propagate(labels)
                print("descending gradient...")
                self.update_weights(input)
                self.print_network()

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """

        # self.test_function()
        # return

        features.shuffle(labels)
        original_features = copy.deepcopy(features)
        original_labels = copy.deepcopy(labels)
        print("original: ", original_features.rows, original_labels.rows)
        print("original cols: ", original_features.cols)
        val_size = int(features.rows/3)
        uncropped_validation_features = Matrix(features, 0, 0, val_size, features.cols)
        validation_features = Matrix(features, 0, 2, val_size, features.cols - 2)
        validation_labels = Matrix(labels, 0, 0, val_size, labels.cols)
        train_features = Matrix(features, val_size, 2, features.rows - val_size, features.cols - 2)
        train_labels = Matrix(labels, val_size, 0, labels.rows - val_size, labels.cols)
        print("val: ", validation_features.rows, validation_labels.rows)
        print("train: ", train_features.rows, train_labels.rows)
        print("train cols: ", train_features.cols)

        self.build_network(train_features, train_labels)

        train_mse = 0
        val_mse = 0
        train_mse_array = []
        validation_mse_array = []
        accuracy_array = []
        # train network
        epochs = 0
        total_epochs = 0
        b_error = 100
        b_train_error = 100
        running_error = 0
        bssf = None
        while epochs < 5:
            running_error = 0
            print("epoch: ", epochs)
            # set up data
            epochs += 1
            total_epochs += 1
            train_features.shuffle(train_labels)
            # for each input vector/row
            for row_index in range(train_features.rows):
                inputs = np.insert(train_features.data[row_index], 0, 1)
                target = self.convert_labels(train_labels, row_index)
                # print("master input: ", inputs)
                # print("forward propagating...")
                output = self.calculate_output(inputs)
                # print("predicted output: ", output)
                # print("predicted converted: ", output.index(max(output)))
                # print("expected output: ", labels.data[row_index][0])
                # print("converted output: ", target)
                running_error += self.calc_sum_square_error(target, output)
                # calculate error
                # print("back propagating...")
                self.back_propagate(train_labels, row_index)
                # print("descending gradient...")
                self.update_weights(inputs)
                # self.print_network()
            accuracy_array.append(self.measure_accuracy(uncropped_validation_features, validation_labels))
            train_mse = running_error/train_features.rows
            train_mse_array.append(train_mse)
            val_sse = 0
            for row_index in range(validation_features.rows):
                inputs = np.insert(validation_features.data[row_index], 0, 1)
                target = self.convert_labels(validation_labels, row_index)
                output = self.calculate_output(inputs)
                val_sse += self.calc_sum_square_error(target, output)
            #print("train: ", val_sse)
            val_mse = val_sse/validation_features.rows
            validation_mse_array.append(val_mse)
            if val_mse < b_error:
                b_error = val_mse
                b_train_error = train_mse
                bssf = copy.deepcopy(self.network)
                epochs = 0
            # check validation set
        self.network = bssf
        # self.print_network()
        print("total epochs: ", total_epochs)
        print("train mse: ", b_train_error)
        print("val mse: ", b_error)
        with open("neuralAcc.csv", "wb") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(train_mse_array)
            wr.writerow(validation_mse_array)
            wr.writerow(accuracy_array)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        features = features[2:]
        inputs = np.insert(features, 0, 1)
        output = self.calculate_output(inputs)
        # print("Features: ", features)
        # print("Output: ", output)
        if len(output) == 1:
            if output[0] > 0:
                labels.append(1)
            else:
                labels.append(0)
        else:
            max_index = output.index(max(output))
            labels.append(max_index)
        labels.append(output)
        # self.print_network()
