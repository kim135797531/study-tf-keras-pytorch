import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def mask(T=200):
    mask = np.zeros(T)
    indices = np.random.permutation(np.arange(T))[:2]
    mask[indices] = 1
    return mask


def toy_problem(N=10, T=200):
    signals = np.random.uniform(low=0.0, high=1.0, size=(N, T))
    masks = np.zeros((N, T))
    for i in range(N):
        masks[i] = mask(T)

    data = np.zeros((N, T, 2))
    data[:, :, 0] = signals[:]
    data[:, :, 1] = masks[:]
    target = (signals * masks).sum(axis=1).reshape(N, 1)

    return data, target


"""
데이터를 생성한다
"""
N = 10000
T = 200
maxlen = T  # 시계열 데이터 한 개의 길이

X, Y = toy_problem(N=N, T=T)

# 데이터를 설정한다
N_train = int(N * 0.9)
N_validation = N - N_train
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)


"""
모델을 설정한다
"""
n_in, n_hidden, n_out = len(X[0][0]), 100, len(Y[0])


def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)


early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=1)

model = Sequential()
model.add(LSTM(input_shape=(maxlen, n_in), units=n_hidden, kernel_initializer=weight_variable))
model.add(Dense(units=n_out, kernel_initializer=weight_variable))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999))


"""
모델을 학습시킨다
"""
epochs = 1000
batch_size = 100

hist = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
          epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])


"""
예측 정확도를 평가한다
"""
loss = hist.history['loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(len(loss)), loss, label='loss', color='black')
plt.xlabel('epochs')
plt.show()
plt.savefig(__file__ + '.eps')
