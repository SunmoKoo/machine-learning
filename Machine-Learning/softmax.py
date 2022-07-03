# 소프트맥스(다중 분류 로지스틱 회귀)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("train data(count, row, column :" +str(X_train.shape))
print("test data(count, row, column :" +str(X_test.shape))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("train target(count) :" + str(y_train.shape))
print("test target (count) :" + str(y_test.shape))

print("sample from train :" + str(y_train[0]))
print("sample from test :" + str(y_test[0]))

input_dim = 784  # 28*28
X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)

print(X_train.shape)
print(y_train.shpae)
print(X_test.shape)
print(y_test.shape)

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(y_train[0])

model = Sequential()
model.add(Dense(input_dim=input_dim, units=10, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=2048, epochs=100, verbose=0

score = model.evaluate(X_test, y_test)
print('Test accuracy :', score[1])

model.summary()
model.layers[0].weights