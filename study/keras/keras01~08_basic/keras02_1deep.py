# import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜 모델과
from tensorflow.keras.layers import Dense # 덴스 레이어를 쓸수있다. 
import numpy as np

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3])
y =  np.array([1,2,3])

#2. 모델구성 layer와 paramiter를 추가하여 deep러닝으로 만들어본다.
model = Sequential()
model.add(Dense(5, input_dim=1)) #아웃풋 인풋이 1개?  각 줄이 레이어 개념이다. 아웃풋 개수를 아래에서 바로 받아서 쓰므로
model.add(Dense(3)) # 아래줄에는 input개수를 안써줘도 된다.
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1)) 
# 모델층을 두껍게해서 다중신경망을 형성하여 그 뒤 컴파일하고 예측을 해보면 단일신경망일때에 비하여 훈련량epochs를 훨씬 줄여도 loss값을 구할수있다.

#3. 컴파일, 훈련      컴퓨터가 알아듣게 훈련시킴 그게 컴파일 y = wx + b 최적의 weight값을 빼기 위한 최소의 mse값을 찾는다.
model.compile(loss='mse', optimizer='adam') # 평균 제곱 에러 m s e 이 값은 작을수록 좋다. optimizer='adam'은  mse값 감축시키는 역할. 85점 이상이면 쓸만하다.

model.fit(x, y, epochs=200, batch_size=1) # epochs 훈련량을 의미  batch_size 몇개씩 데이터를 넣을지 지정해줌. batch가 작을수록 값이정교해짐 시간 성능 속도?
#위의 데이터들로 훈련하겠다 fit. 

#4. 평가, 예측
loss = model.evaluate(x, y) # 평가하다.
print('loss : ', loss)
result = model.predict([6]) # 새로운 x값을 predcit한 결과 
print('6의 예측값 : ', result)

# epochs=500, batch_size=1
# loss 값 : 3.907985046680551e-14  6의 예측값 : [[6.0000014]]
#           2.842170943040401e-14  [[6.0000005]]