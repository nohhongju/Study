# 데이터를 train과 test로 나눠주는 이유
# fit에서 모델학습을 시킬때 모든 데이터를 다 학습시켜버리면 x = [1~10] y = [1~10]
# 실제로 원하는 미래의 데이터를 넣어봤을때 크게 오류가 날 수 있다 model.predict[11]
# 왜냐하면 컴퓨터는 모든 주어진 값으로만 훈련을 하고 실전을 해본적이 없기때문이다
# 이때문에 train과 test로 나누어 x_train [1~7] x_test[8~10]
# train으로 학습을 시키고 test로 실전같은 모의고사를 한번 미리해보면
# fit단계에서의 loss값과 evaluate의 loss값의 차이가 큰 걸 확인할 수 있다.
# 확인까지만 가능하고 그 이상은 뭐 할 수 없다? evaluate은 평가만 가능한거지 
# 여기서 나온 loss값과 fit 단게의 loss값들의 차이가 크다 하더라도 그 차이가 fit단계에 
# 적용되지는 않는다.
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])      # 훈련
x_test = np.array([8,9,10])             # 평가
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss) # 결과값에서 나온 loss?
result = model.predict([11])
print('[11]의 예측값 : ', result)