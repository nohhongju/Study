from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score    # r2_score 구하는 공식? 및 작업을 다 해놓은걸  import해서 가져다쓴다 

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])  

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # mean squared error , 평균제곱오차    
model.fit(x,y,epochs=1000, batch_size=1)

#4. 평가, 예측 
loss = model.evaluate(x,y) # 평가해보는 단계. 이미 다 나와있는  w,b에 test데이터를 넣어보고 평가해본다.
print('loss : ', loss)  # loss :  0.380171000957489

y_predict = model.predict(x) #y의 예측값은 x의 테스트값에 wx + b 
print(y_predict)
'''[[1.2053602]
    [2.1087615]
    [3.0121627]
    [3.9155636]
    [4.8189645]]'''

r2 = r2_score(y,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)  # r2스코어 :  0.8099145029981514

# r2스코어 :  0.8096227565995434
# r2스코어 :  0.8099396914282224
# r2스코어 :  0.809914410101338
# r2스코어 :  0.8099800601232573
# r2스코어 :  0.8099735798526126