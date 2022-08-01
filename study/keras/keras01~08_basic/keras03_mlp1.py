import numpy as np
from regex import P
from tensorflow.keras import models 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 멀티 레이어 퍼섹트론?
#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])   
print(x.shape)
# (2, 10)
print(x)
'''[[ 1.   2.   3.   4.   5.   6.   7.   8.   9.  10. ] 
    [ 1.   1.1  1.2  1.3  1.4  1.5  1.6  1.5  1.4  1.3]]'''
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(y.shape)
# (10,)
x = x.T
# print(x)
'''[[ 1.   1. ]
    [ 2.   1.1]
    [ 3.   1.2]
    [ 4.   1.3]
    [ 5.   1.4]
    [ 6.   1.5]
    [ 7.   1.6]
    [ 8.   1.5]
    [ 9.   1.4]
    [10.   1.3]]'''
# x = x.reshape(10,2)  
# print(x.shape)
# # (10, 2)
# print(x)
'''[[ 1.   2. ]
    [ 3.   4. ]
    [ 5.   6. ]
    [ 7.   8. ]
    [ 9.  10. ]
    [ 1.   1.1]
    [ 1.2  1.3]
    [ 1.4  1.5]
    [ 1.6  1.5]
    [ 1.4  1.3]]'''
#x.T 이거 하나만으로 배열의 행과 열을 바꿔줌.
# x = np.transpose(x)   이거도 똑같은 기능이다 
# x = reshape(10,2)     변환은 되는데 print해서 확인해보면 짝이 다 깨져있다. 구조가 엉망이 되어있다.
# transpose 와 reshape의 차이점 transpose는 변환의 개념이고 , reshape는 데이터를 늘였다 줄였다하면서 형태를 바꿔주는 개념이다.
# reshape는그래서 원본이 보존된다.
# print(x.T)
# print(x.T.shape)


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))    # 벡터의 수와 같다. 컬럼,열,특성
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4 평가, 예측

loss = model.evaluate(x,y)
print('loss : ', loss)  # loss :  0.05460137873888016
y_predict = model.predict([[10,1.3]])
print('[10,1.3]의 예측값 : ', y_predict)  # [10,1.3]의 예측값 :  [[19.511517]]