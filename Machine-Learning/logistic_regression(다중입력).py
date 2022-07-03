# 다중입력 로지스틱 회귀
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model = Sequential()
model.add(Dense(imput_dim=2, units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])

X = np.array([(0,0),(0,1),(1,0),(1,1)])
Y = np.array([0,0,0,1])
model.fit(X, Y, epochs=5000, verbose=0)

model.predict(X)
model.summary()
model.layers[0].weights
model.layers[0].get_weights()