# 지금까지 배운 기존의 Train & Test 시스템은 훈련 후 바로 실전을 치뤄서 실전에서 값이 많이 튀었다.
# 이걸 더 보완하기위해서 Train 후 validation(검증) 거치는 걸 1 epoch로 하여 계속반복하며 값을 수정 후
# 여기서 validation에서 나오는 val_loss가 fit에 직접적으로 관여하지는 않지만 방향성을 제시해준다.
# 실전 Test를 치루는 방식으로 더 개선시킨다. 앞으로는 데이터를 받으면 train validation test 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 정제작업
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train = x[:10]  # [ 1  2  3  4  5  6  7  8  9 10]
y_train = y[:10]  # [ 1  2  3  4  5  6  7  8  9 10]
x_test = x[11:14]  # [12 13 14]
y_test = y[11:14]  # [12 13 14]
x_val =  x[-3:]  # [14 15 16]
y_val =  y[-3:]  # [14 15 16]

print(x_train,y_train,x_test,y_test,x_val,y_val)  # 슬라이싱 잘 되었나 확인.
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
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
''' validation_data: 각 세대의 끝에서 손실과 모델 측정항목을 평가할 (x_val, y_val) 튜플, 
    혹은 (x_val, y_val, val_sample_weights) 튜플. 
    모델은 이 데이터에 대해서는 학습되지 않습니다. 
    validation_data는 validation_split에 대해 우선순위를 갖습니다.
    -> 위에서 나누어 줬기 때문에 가져와서 쓴다.'''
'''Epoch 100/100
    10/10 [==============================] - 0s 5ms/step - loss: 3.0298e-12 - val_loss: 5.1538e-12
    1/1 [==============================] - 0s 80ms/step - loss: 1.0004e-11'''

#4. 평가, 예측
model.evaluate(x_test, y_test)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)  # 17의 예측값 :  [[17.]]