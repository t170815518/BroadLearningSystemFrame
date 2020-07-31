'''
Define the RVFLNN (Random-Vector Functional-Link Neural Network) object.
Implementation of the paper Learning and generalization characteristics of the random vector Functional-link net(1994) by
Yoh-Han Pao, Gwang-HoonPark and Dejan J. Sobajic
'''
import numpy as np
from numpy.linalg import norm


class RVFLNN:
    '''
    RVFLNN structure
    Attributes:
        self.input_number: int, the number of input nodes
        self.enhance_number: int, the number of enhancement nodes
        self.W: np.ndarray, the weights of input patterns and enhancement nodes
    Methods:
        predict(self,input_data,target_value=None)
        train(self, train_data, target_values, epoch, lr=0.001,)
    '''
    def __init__(self,input_number,enhance_number):
        '''
        net initialization

        :param input_number: int, input nodes' number
        :param enhance_number: int, enhancement nodes' number
        '''
        self.input_nodes = np.empty(input_number)
        self.enhance_nodes = np.empty(enhance_number)

        self.input_number = input_number
        self.enhance_number = enhance_number

        self.W = np.reshape(np.random.rand(input_number+enhance_number), (1, -1))  # Weight initialization
        # Weight randomization
        self.bias = np.random.rand(enhance_number,1)
        self.random_coef = np.random.normal(scale=0.5, size=(enhance_number, input_number))  # to avoid saturation

    def __enhance__(self, input_vec):
        '''
        Calculate the values of enhancement nodes.
        :param input_vec: np.ndarray, the input vector fed into the network
        :return self._enhance_input, np.ndarray, the enhanced values
        Note: The default activation function is tanh
        '''
        input_vec = np.reshape(input_vec,(self.input_number,-1))

        self._enhance_input = np.tanh(np.dot(self.random_coef,input_vec)+self.bias)
        return self._enhance_input

    def predict(self,input_data,target_value=None):
        '''
        Based on input data fed, predict the result.
        :param input_data: np.ndarray
        :param target_value: np.ndarray
        :return: tuple (prediction result, error). error = target_value-prediction; when target_value=None, None is
        returned prediction, error; both are np.ndarray.
        '''
        input_layer = np.concatenate((np.reshape(input_data,(self.input_number, -1)),
                                      self.__enhance__(input_data)), axis=0)
        prediction = np.dot(self.W, input_layer)
        if target_value is not None:
            error = target_value - prediction
        else:
            error = None
        return prediction, error

    def train(self, train_data, target_values, epoch, lr=0.001,):
        '''
        Train the net via conjugate gradient.
        :param train_data: np.ndarray
        :param target_values: np.ndarray
        :param epoch: int
        :param lr: float; default=0.01
        '''
        size = train_data.shape[0]
        target_values = np.reshape(target_values,(1,-1))
        input_layer = np.concatenate((np.reshape(train_data,(self.input_number, -1)),
                                      self.__enhance__(train_data)), axis=0)
        # conjugate gradient
        r0 = np.reshape(-1/size*np.dot(target_values - np.dot(self.W,input_layer),np.transpose(input_layer)), (1, -1))
        s0 = r0
        self.W = self.W + lr*s0
        for i in range(1,epoch):
            r1 = np.reshape(-1/size*np.dot(target_values - np.dot(self.W,input_layer),np.transpose(input_layer)), (1, -1))
            s1 = -r1 + norm(r1)/norm(r0)*s0
            self.W = self.W + lr * s1
            r0 = r1
            s0 = s1



