import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func

no_iter = 1000  # number of iteration
no_train = 70# Your code here  # number of training data
no_test = 30# Yourcode here  # number of testing data
no_data = 100  # number of all data
assert(no_train + no_test == no_data)

cumulative_train_err = 0
cumulative_test_err = 0

for i in range(no_iter):
    X, y, w_f = gen_data(no_data)
    X_train, X_test = X[:, :no_train], X[:, no_train:]
    y_train, y_test = y[:, :no_train], y[:, no_train:]
    print(X_train)
    print(y_train)
    w_g = func(X_train, y_train)
    # Compute training, testing error
    # Your code here
    # Answer begin
    # e.g
    train_err = 0
    test_err = 0
    train_P,train_N = X_train.shape
    test_P,test_N = X_test.shape
    Xt = X_train.T
    for i in range(train_N):
        if (float(np.dot(Xt[i,:],w_g[0:train_P,0])+w_g[train_P,0])*y_train[0,i] <= 0):
            train_err += 1
    Xt = X_test.T
    print(train_err)
    for i in range(test_N):
        if (float(np.dot(Xt[i,:],w_g[0:test_P,0])+w_g[test_P,0])*y_test[0,i] <= 0):
            test_err += 1
    print(test_err)
    # Answer end
    cumulative_train_err += train_err
    cumulative_test_err += test_err

train_err = cumulative_train_err / no_iter
test_err = cumulative_test_err / no_iter

plot(X, y, w_f, w_g, "Classification")
print("Training error: %s" % train_err)
print("Testing error: %s" % test_err)

'''
filename = "result.txt"
with open(filename, 'a') as res:
    res.write(w_g)
    res.write("Training error: %s" % train_err)
    res.write("Testing error: %s" % test_err)
'''

