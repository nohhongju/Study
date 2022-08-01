# import tensorflow as tf
from tensorflow.keras.models import Sequential  # 시퀀셜 모델과
from tensorflow.keras.layers import Dense  # 덴스 레이어를 쓸수있다. 
import numpy as np

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3,5,4])  # x.shape = (5, 1)
y =  np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))  #아웃풋 인풋이 1개?

#3. 컴파일, 훈련 -> 컴퓨터가 알아듣게 훈련시킴 그게 컴파일 y = wx + b 최적의 weight값을 빼기 위한 최소의 mse값을 찾는다.
model.compile(loss='mse', optimizer='adam')  # 평균 제곱 에러 mse 이 값은 작을수록 좋다. optimizer='adam'은  mse값 감축시키는 역할. 85점 이상이면 쓸만하다.

model.fit(x, y, epochs=2800, batch_size=1)  # epochs 훈련량을 의미  batch_size 몇개씩 데이터를 넣을지 지정해줌. batch가 작을수록 값이정교해짐 시간 성능 속도?
#위의 데이터들로 훈련하겠다 fit. 

#4. 평가, 예측
loss = model.evaluate(x, y)  # 평가하다.
print('loss : ', loss)  # loss :  0.38014134764671326
result = model.predict([6])  # 새로운 x값을 predcit한 결과 
print('6의 예측값 : ', result)  # 6의 예측값 :  [[5.7219687]]
                                # 5의 예측값 :  [[4.8137674]]
