import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from tensorflow.keras import datasets
mnist_train, mnist_test = datasets.mnist.load_data()


class MNIST_wrapper:
    def __init__(self, mnist_part_1, mnist_part_2):
        self.data = np.concatenate((mnist_part_1[0], mnist_part_2[0]))
        self.data = self.data.reshape(len(self.data), 784)
        self.target = np.concatenate((mnist_part_1[1], mnist_part_2[1]))


mnist = MNIST_wrapper(mnist_train, mnist_test)

n = len(mnist.data)
N = 10000
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
# 1-of-K 표현으로 변환한다. (ex: 3 -> [0 0 0 1 0 0 0 0 0 0])
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

"""
모델을 설정한다
"""
n_in = len(X[0])  # 784
n_hidden = 200
n_out = len(Y[0])  # 10

model = Sequential()
model.add(Dense(input_dim=n_in, units=n_hidden))
model.add(Activation('sigmoid'))

model.add(Dense(units=n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

"""
모델을 학습시킨다
"""
epochs = 1000
batch_size = 10

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

"""
예측 정확도를 평가한다
"""
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
