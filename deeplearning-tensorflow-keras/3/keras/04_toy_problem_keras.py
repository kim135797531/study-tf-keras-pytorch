from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from sklearn import datasets
from sklearn.model_selection import train_test_split


N = 300
X, y = datasets.make_moons(N, noise=0.3)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

num_hidden = 3

model = Sequential()
model.add(Dense(num_hidden, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=500, batch_size=20)

loss_and_metrics = model.evaluate(X_test, Y_test)

# 오차함수값, 예측 정확도
print(loss_and_metrics)
