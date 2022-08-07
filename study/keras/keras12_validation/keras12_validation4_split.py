# 지금까지 배운 기존의 Train & Test 시스템은 훈련 후 바로 실전을 치뤄서 실전에서 값이 많이 튀었다.
# 이걸 더 보완하기위해서 Train 후 validation(검증) 거치는 걸 1 epoch로 하여 계속반복하며 값을 수정 후
# 실전 Test를 치루는 방식으로 더 개선시킨다. 앞으로는 데이터를 받으면 train validation test 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split # 싸이킷런, 싸이킷런 

#1. 데이터 정제작업
x = np.array(range(1,17))
y = np.array(range(1,17))

# train_test_split로 나누시오 10 3 3 

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.625, random_state=66) 
x_val,x_test,y_val,y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=49)

# print(x_train)
# print(y_train)
# print(x_val,y_val)
# print(x_test)
# print(y_test)
 #확인작업


# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
'''Epoch 100/100
    10/10 [==============================] - 0s 5ms/step - loss: 6.7430e-07 - val_loss: 5.8546e-07
    1/1 [==============================] - 0s 83ms/step - loss: 7.6527e-07'''

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  # loss :  7.652688509551808e-07

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)  # 17의 예측값 :  [[16.999056]]