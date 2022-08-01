# import tensorflow as tf
from tensorflow.keras.models import Sequential  # 시퀀셜 모델 -> 순차적으로 레이어 층을 더해주는 모델을 순차모델이라고 한다.
from tensorflow.keras.layers import Dense  # 덴스 레이어를 쓸수있다. 
import numpy as np

#1. 데이터 정제해서 값 도출 -> 학습 데이터
x =  np.array([1,2,3])  # x.shape = (3, 1) (행, 열)
y =  np.array([1,2,3])

#2. 모델구성
model = Sequential()  # 순차모델로 만든다.
model.add(Dense(1, input_dim=1))  # Dense로 만들어진 노드를 모델에 add하면 모델안에 있는 다른 노드들과 연결된다.(출력 노드개수, 열)

#3. 컴파일, 훈련 -> 컴퓨터가 알아듣게 훈련시킴 그게 컴파일 y = wx + b 최적의 weight값을 빼기 위한 최소의 mse값을 찾는다.
model.compile(loss='mse', optimizer='adam') # 평균 제곱 에러 mse 이 값은 작을수록 좋다. optimizer='adam'은  mse값 감축시키는 역할. 85점 이상이면 쓸만하다.

model.fit(x, y, epochs=4200, batch_size=1) # epochs 훈련량을 의미  batch_size 몇개씩 데이터를 넣을지 지정해줌. batch가 작을수록 값이정교해짐 시간 성능 속도? 배치사이즈가 작을 수록 시간이 많이 필요함
#위의 데이터들로 훈련하겠다 fit. 
# 두 데이터의 행이 같아야 한다.

#4. 평가, 예측
loss = model.evaluate(x, y) # 평가하다. -> 모델의 최종적인 accuracy(정확도)와 loss 값을 알 수 있다.
print('loss : ', loss)  # loss : 6.158037269129654e-14 -> loss는 예측값과 실제값이 차이나는 정도를 나타내는 지표이며 작을 수록 좋다.
# 전체 샘플의 개수들 중에서 얼마나 나의 알고리즘이 정답이라고 예측한 샘플이 포함되었는지의 비율을 의미한다.
result = model.predict([4]) # 새로운 x값을 predcit한 결과 
print('4의 예측값 : ', result)  # 4의 예측값 : [[3.9999993]]
                                # 5의 예측값 : [[4.992851]]