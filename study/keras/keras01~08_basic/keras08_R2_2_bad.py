#1. R2를 음수가 아닌 0.5 이하로 만들것 
#2. 데이터 건들지 마!!
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100이상
#6. 히든레이어의 노드는 10개 이상 1000개 이하
#7. train 70% 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt     # 그래프나 그림그릴때 많이 씀
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score     

#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))  

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(444, input_dim=1))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # mean squared error , 평균제곱오차    
model.fit(x_train,y_train,epochs=100, batch_size=1)

#4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 

r2 = r2_score(y_test,y_predict)
print('r2스코어 : ', r2)
# 모델링 노드를 많이 주다가 적게주는걸 반복함으로써 고의로 
# loss값을 계속 튀게하여 정확도를 크게 낮추었다.
# loss: 679.3102
# r2스코어 :  0.2231010336057584