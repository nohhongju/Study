# 지금까지 배운 기존의 Train & Test 시스템은 훈련 후 바로 실전을 치뤄서 실전에서 값이 많이 튀었다.
# 이걸 더 보완하기위해서 Train 후 validation(검증) 거치는 걸 1 epoch로 하여 계속반복하며 값을 수정 후
# 실전 Test를 치루는 방식으로 더 개선시킨다. 앞으로는 데이터를 받으면 train validation test 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 정제작업
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
'''Epoch 100/100
    10/10 [==============================] - 0s 6ms/step - loss: 0.0051 - val_loss: 0.0334
    1/1 [==============================] - 0s 74ms/step - loss: 0.0136'''

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  # loss :  0.013592620380222797

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)  # 17의 예측값 :  [[17.226255]]