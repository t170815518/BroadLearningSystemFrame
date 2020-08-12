import numpy as np
from numpy.linalg import pinv
from test import sin_generation,visualization
import datetime


def REIL(net,input_data,target_data,error_threshold=None,increment=3):
    '''
    Use Algorithm Rank-Expansion with Instant Learning to train RVFLNN net.

    :param net: RVFLNN object
    :param input_data: np.ndarray, the input vectors are row vectors
    :param target_data: np.ndarray, in one row
    :param error_threshold: float
    :param increment: int

    Ref: C. L. Philip Chen, J.Z.W.: ‘A Rapid Learning and Dynamic Stepwise Updating Algorithm for Flat Neural Networks
and the Application to Time-Series Prediction’, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS, 1999
    '''
    def error():
        err_array = net.predict(input_data, target_data)[1]
        return np.sum(np.abs(err_array)) / err_array.shape[1]
        # return np.sum(err_array**2)/err_array.shape[1]

    while (not error_threshold) or error() > error_threshold:
        net.enhance_number += increment
        net.W = np.concatenate([net.W,np.reshape(np.random.rand(increment), (1, -1))], axis=1)
        net.bias = np.concatenate([net.bias,np.random.rand(increment,1)],axis=0)
        net.random_coef = np.concatenate([net.random_coef,np.random.normal(scale=0.5, size=(increment,net.input_number))],
                                         axis=0)
        net.W = np.dot(target_data,pinv(net.input_layer(input_data)))
        if not error_threshold:
            break
    net.input_stored = input_data


def update_with_data(net,new_input,new_target):
    '''
    Update the weights of net when 1 new observation added
    :param net: RVFLNN object
    :param new_input: np.ndarray
    :param new_target: np.ndarray
    '''
    a = net.input_layer(new_input).reshape(1,-1)
    d_ = np.dot(a,pinv(net.input_layer().transpose()))
    c_ = np.dot(d_,net.input_layer().transpose())
    if (c_ < 10e-9).any(): # c' = 0
        b = np.dot(pinv(net.input_layer().transpose()),d_.transpose()) / (1 + np.dot(d_, d_.transpose()))
    else:
        b = pinv(c_)
    net.W = net.W + (new_target - np.dot(a,net.W.transpose()))*b


if __name__=='__main__':
    from RVFLNN import RVFLNN


    def test_REIL():
        # configuration
        train_size = 50
        test_size = 80
        time = datetime.datetime.now()
        # generate samples of sin function
        X_train, y_train = sin_generation(size=train_size)
        X_test, y_test = sin_generation(size=test_size)
        # train
        net = RVFLNN(1, 5)
        REIL(net,X_train,y_train,error_threshold=0.05, increment=10)
        prediction, error = net.predict(X_test, y_test)
        error = np.sum(np.abs(error)) / test_size
        # print results
        print("Error={}\nTime={}".format(error, datetime.datetime.now() - time))
        pred_train, _ = net.predict(X_train, y_train)
        visualization(X_train,y_train,X_test,y_test,pred_train,prediction)


    def test_incremental_node():
        # initial data
        X_train, y_train = sin_generation(size=10)
        X_test, y_test = sin_generation(size=80)
        net = RVFLNN(1, 5)
        REIL(net, X_train, y_train)
        prediction, error = net.predict(X_test, y_test)
        error = np.sum(np.abs(error)) / 80
        print("Error={}\n".format(error))

        X_train, y_train = sin_generation(size=1)
        # update_with_data(net, X_train, y_train)
        net.update_with_node()


    test_incremental_node()
