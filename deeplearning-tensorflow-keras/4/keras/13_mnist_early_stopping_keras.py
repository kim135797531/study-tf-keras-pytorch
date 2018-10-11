import numpy as np
from keras.initializers import TruncatedNormal
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping

from tensorflow.keras import datasets
import matplotlib.pyplot as plt

mnist_train, mnist_test = datasets.mnist.load_data()


class MNIST_wrapper:
    def __init__(self, mnist_part_1, mnist_part_2):
        self.data = np.concatenate((mnist_part_1[0], mnist_part_2[0]))
        self.data = self.data.reshape(len(self.data), 784)
        self.target = np.concatenate((mnist_part_1[1], mnist_part_2[1]))


mnist = MNIST_wrapper(mnist_train, mnist_test)

n = len(mnist.data)
N = 30000
N_train, N_validation = 20000, 4000
indices = np.random.permutation(range(n))[:N]

X = mnist.data[indices]
y = mnist.target[indices]
# 1-of-K 표현으로 변환한다. (ex: 3 -> [0 0 0 1 0 0 0 0 0 0])
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

"""
모델을 설정한다
"""
n_in, n_hiddens, n_out = len(X[0]), [200, 200, 200], len(Y[0])
p_keep = 0.5
activation = 'relu'


model = Sequential()
for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
    model.add(Dense(input_dim=input_dim, units=n_hiddens[i], kernel_initializer=TruncatedNormal(stddev=0.01)))
    model.add(Activation(activation))
    model.add(Dropout(p_keep))

model.add(Dense(units=n_out, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])

"""
모델을 학습시킨다
"""
epochs = 50
batch_size = 200

# 10번의 에폭에서 계속해서 오차가 증가하면 학습을 끝낸다
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

hist = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                 epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

"""
예측 정확도를 평가한다
"""
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

plt.rc('font', family='serif')
fig = plt.figure()

plt.plot(range(len(val_acc)), val_acc, label='loss', color='black')
plt.xlabel('epochs')
plt.show()

loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
