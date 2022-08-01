import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array( [[1,2,3,4,5,6,7,8,9,10],
               [1,1.1,1.2,1.3,1.4,1.5,
                1.6,1.5,1.4,1.3],
               [10,9,8,7,6,5,4,3,2,1]]) 


#  1. 데이터 정제해서 값 도출
x = x.T
print(x)
'''[[ 1.   1.  10. ]
    [ 2.   1.1  9. ]
    [ 3.   1.2  8. ]
    [ 4.   1.3  7. ]
    [ 5.   1.4  6. ]
    [ 6.   1.5  5. ]
    [ 7.   1.6  4. ]
    [ 8.   1.5  3. ]
    [ 9.   1.4  2. ]
    [10.   1.3  1. ]] (10, 3)'''
print(x.T)
'''[[ 1.   2.   3.   4.   5.   6.   7.   8.   9.  10. ]
    [ 1.   1.1  1.2  1.3  1.4  1.5  1.6  1.5  1.4  1.3]
    [10.   9.   8.   7.   6.   5.   4.   3.   2.   1. ]] (3, 10)'''
#  x = np.transpose(x)   이거도 똑같은 기능이다 
y = np.array([11,12,13,14,15,16,17,18,19,20])   # (1, 10)

#  [[10, 1.3, 1]] 결과값 예측

# print(x.shape)
# print(x)           바귄거 결과확인

# 2. 모델구성 layer와 paramiter  추가.
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))  # 가중치가 없는 곳부터 
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(12)) 
model.add(Dense(1))

#  3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)


#  4. 평가 , 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)  # loss :  0.725322425365448
result = model.predict([[ 10, 1.3, 1]])
print('[ 10, 1.3, 1]의 예측값 : ', result)  # [ 10, 1.3, 1]의 예측값 :  [[20.210913]]