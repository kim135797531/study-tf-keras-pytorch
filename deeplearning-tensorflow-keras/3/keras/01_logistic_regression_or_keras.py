import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(0)

# y = σ(w1x1 + w2x2 + b)
model = Sequential([
    Dense(input_dim=2, units=1),
    Activation('sigmoid')
])

model_alt = Sequential()
model_alt.add(Dense(input_dim=2, units=1))
model_alt.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

model.fit(X, Y, epochs=200, batch_size=1)

classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)