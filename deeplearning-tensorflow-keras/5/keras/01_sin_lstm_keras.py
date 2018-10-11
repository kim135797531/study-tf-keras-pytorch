import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def sin(x, T=100):
    return np.sin(2.0 * np.pi * x/T)


def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2*T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise


"""
데이터를 생성한다
"""
T = 100
f = toy_problem(T)

length_of_sequences = 2 * T  # 시계열 전체 길이
maxlen = 25  # 시계열 데이터 한 개의 길이

data = []
target = []

for i in range(0, length_of_sequences - maxlen + 1):
    data.append(f[i:i + maxlen])
    target.append(f[i + maxlen])

X = np.array(data).reshape(len(data), maxlen, 1)
Y = np.array(target).reshape(len(data), 1)

# 데이터를 설정한다
N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)


"""
모델을 설정한다
"""
n_in, n_hidden, n_out = len(X[0][0]), 20, len(Y[0])


def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model = Sequential()
model.add(LSTM(input_shape=(maxlen, n_in), units=n_hidden, kernel_initializer=weight_variable))
model.add(Dense(units=n_out, kernel_initializer=weight_variable))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999))


"""
모델을 학습시킨다
"""
epochs = 500
batch_size = 10

model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
          epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])


"""
예측 정확도를 평가한다
"""
truncate = maxlen
Z = X[:1]  # 원 데이터의 처음 일부를 잘라내기

original = [f[i] for i in range(maxlen)]
predicted = [None for i in range(maxlen)]

for i in range(length_of_sequences - maxlen + 1):
    z_ = Z[-1:]
    y_ = model.predict(z_)
    sequence_ = np.concatenate(
        (z_.reshape(maxlen, n_in)[1:], y_), axis=0
    ).reshape(1, maxlen, n_in)
    Z = np.append(Z, sequence_, axis=0)
    predicted.append(y_.reshape(-1))


"""
그래프로 가시화
"""
plt.rc('font', family='serif')
plt.figure()
plt.ylim([-1.5, 1.5])
plt.plot(toy_problem(T, ampl=0), linestyle='dotted', color='#aaaaaa')
plt.plot(original, linestyle='dashed', color='black')
plt.plot(predicted, color='black')
plt.show()
