# Neural Network Should Comprise of Following Major Components
# 1) Node 2) Edge 3) Network

import math, random
from abc import abstractmethod


class major_components:

    @abstractmethod
    def initialize(self, initializing_array):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass


# ------------------------------------------------------------------

class edge:

    def __init__(self):
        self.weight=None
        self.from_node=None
        self.to_node=None

    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
# -------------------------------------------------------------------


class neuron:

    def __init__(self):

        self.value = None
        self.in_edge = []
        self.out_edge = []
        self.bias = None

    def initialize(self,initializing_array):
        new_edge = edge()
        new_edge.from_node = initializing_array
        new_edge.to_node = self
        self.in_edge.append(new_edge)
        initializing_array.out_edge.append(new_edge)


# --------------------------------------------------------------------

class network(major_components, neuron, edge):

    def __init__(self, *layers):
        self.layers = []
        self.nodes = []
        self.init_layers = layers
        self.training_data = [
            [[0, 0], [0]],
            [[0, 1], [1]],
            [[1, 0], [1]],
            [[1, 1], [0]]
        ]
        self.iteration=3000

    # Private Method
    def __init(self):
        for layer_id in range(0, len(self.init_layers)):
            self.layers.append([])
            for node_id in range(0, self.init_layers[layer_id]):
                new_node = neuron()
                self.nodes.append(new_node)
                self.layers[layer_id].append(new_node)
                if layer_id != 0:
                    for previous_layer_node in self.layers[layer_id-1]:
                        new_node.initialize(previous_layer_node)

    def initialize(self, initializing_array):
        self.__init()
        const = initializing_array
        if const==[]:pass
        for node in self.nodes:
            for edge in node.in_edge:
                if edge.weight == None:
                    edge.weight = random.uniform(-1.0,1.0)
            for edge in node.out_edge:
                if edge.weight == None:
                    edge.weight = random.uniform(-1.0,1.0)
            if node not in self.layers[0]:
                node.bias = random.uniform(-1.0,1.0)

    # Private Method
    def __back_propagation(self, X):
        for input_layer_node_id in range(0,len(self.layers[0])):
            self.layers[0][input_layer_node_id].value = X[input_layer_node_id]

        for hiddenlayer in self.layers[1:]:
            for hiddenlayer_node in hiddenlayer:
                tmp_sum = 0
                for edge in hiddenlayer_node.in_edge:
                    tmp_sum += edge.from_node.value*edge.weight
                tmp_sum += hiddenlayer_node.bias
                hiddenlayer_node.value = edge.sigmoid(tmp_sum)

    # Private Method
    def __back_propagation_error(self, Y):
        mse=0
        for output_node_id in range(0, len(self.layers[len(self.layers)-1])):
            mse += 0.5*(Y[output_node_id] - self.layers[len(self.layers)-1][output_node_id].value)**2
        return mse

    # Private Method
    def __back_propagation_delta(self, trainning_data_y):
        for layer_id in range(len(self.layers)-1,-1,-1):
            for node_id in range(0,len(self.layers[layer_id])):
                node = self.layers[layer_id][node_id]
                if layer_id == len(self.layers)-1:
                    node.delta = (trainning_data_y[node_id] - node.value)* node.value*(1- node.value)
                else:
                    node.delta = 0
                    for edge in node.out_edge:
                        node.delta += edge.weight*edge.to_node.delta
                    node.delta = node.delta*node.value*(1-node.value)

    # Private Method
    def __back_propagation_update_weight(self, Y, learning_rate):
        for layer_id in range(len(self.layers)-1,-1,-1):
            for node_id in range(0,len(self.layers[layer_id])):
                node = self.layers[layer_id][node_id]
                for edge in node.in_edge:
                    edge.weight += learning_rate*node.delta*edge.from_node.value
                if node.bias !=None:
                    node.bias += learning_rate*node.delta

    def train(self):
        for iter_id in range(0, self.iteration):
            iter_mse = 0
            for single_data in self.training_data:
                X = single_data[0]
                Y = single_data[1]
                self.__back_propagation(X)
                iter_mse += self.__back_propagation_error(Y)
                self.__back_propagation_delta(Y)
                self.__back_propagation_update_weight(Y, 0.5)
            print("iteration " + str(iter_id) +" mse:"+str(iter_mse))
        print("training done")

    # Private Method
    def predict(self):
        predict_data = []
        for single_data in self.training_data:
            X = single_data[0]
            predict_data_y = []
            self.__back_propagation(X)
            for node_id in range(0,len(self.layers[len(self.layers)-1])):
                predict_data_y.append(self.layers[len(self.layers)-1][node_id].value)
            predict_data.append([X,predict_data_y])
        return predict_data


if __name__ == '__main__':
    n = network(2,10,1)
    n.initialize([0,1,2])
    n.train()
    predict_data  = n.predict()

    for single_data in predict_data:
        print(single_data)







