# 선형회귀
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 데이터 만들기
X = np.linspace(0, 10, 10)
Y = X + np.random.randn(*X.shape)

# 데이터 조회
for x, y in zip(X,Y):
    PRINT((round(x,1),round(y,1)))

# 선형회귀 모델 만들기
model = Sequential()
model.add(Dense(input_dim=1, units=1, activation="linear", use_bias=False))  # y = ax 이면 true, ax+b 이면 false

# 학습방법 설정 / 경사하강법(Gradient descent)으로 평균제곱오차(MSE)를 줄이는 방법
sgd = optimizers.SGD(1r=0.01)
model.compile(optimizer='sgd', loss='mse')

# 최초 w값 조회
weights = model.layers[0].get_weights()
w = weights[0][0][0]
print('initial w is : ' + str(w))

# 선형회귀 모델 학습
model.fit(X,Y,batch_size=10, epochs=10, verbose=1)  # 10개의 데이터가 있으므로 batch_size는 10, epochs는 10번 반복 수행하여 모델 학습

plt.plot(X, Y, label='data')
plt.plot(X, w*X, label='prediction')
plt.legend()
plt.show()