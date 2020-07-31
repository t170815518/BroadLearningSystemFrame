'''
Test RVFLNN with sin function
'''
import numpy as np
from RVFLNN import RVFLNN
import matplotlib.pyplot as plt
import datetime


def sin_generation(size=10):
    X = np.random.uniform(low=-6.28, high=6.28, size=size)
    Y = np.sin(X)
    return X, Y


def visualization():
    plt.scatter(X_train, y_train, color='b',label='Train')
    plt.scatter(X_test, y_test, color='r',label='Test')
    plt.scatter(X_test, prediction, color='g', label='Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


# configuration
train_size= 100
test_size = 80
epoch = 12300
time = datetime.datetime.now()
# generate samples of sin function
X_train, y_train = sin_generation(size=train_size)
X_test, y_test = sin_generation(size=test_size)
# train
net = RVFLNN(1,300)
net.train(X_train,y_train,epoch=epoch)
prediction, error = net.predict(X_test, y_test)
error = np.sum(np.abs(error))/test_size
# print results
print("Epoch={}\nError={}\nTime={}".format(epoch,error,datetime.datetime.now()-time))
visualization()
