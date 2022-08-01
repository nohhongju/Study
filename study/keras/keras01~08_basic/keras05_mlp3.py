import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 정제
x = np.array([range(10), range(21,31), range(201,211)])
print(x)
'''[[  0   1   2   3   4   5   6   7   8   9]
    [ 21  22  23  24  25  26  27  28  29  30]
    [201 202 203 204 205 206 207 208 209 210]]'''
x = np.transpose(x)
print(x.shape)  # (10, 3)

y = np.array( [[1,2,3,4,5,6,7,8,9,10],
               [1,1.1,1.2,1.3,1.4,1.5,
                1.6,1.5,1.4,1.3],
               [10,9,8,7,6,5,4,3,2,1]]) 

y = np.transpose(y)
print(y)
'''[[ 1.   1.  10. ]
    [ 2.   1.1  9. ]
    [ 3.   1.2  8. ]
    [ 4.   1.3  7. ]
    [ 5.   1.4  6. ]
    [ 6.   1.5  5. ]
    [ 7.   1.6  4. ]
    [ 8.   1.5  3. ]
    [ 9.   1.4  2. ]
    [10.   1.3  1. ]]'''
print(y.shape)  # (10, 3)

# 2. 모델구성 layer와 paramiter  추가.
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(12)) 
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=1)


# 4. 평가 , 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)  # loss :  0.008016151376068592
result = model.predict([[ 9, 30, 210]])
print('[ 9, 30, 210]의 예측값 : ', result)  # [ 9, 30, 210]의 예측값 :  [[9.995297  1.5258816 1.1526216]]

# 실제값: 10, 1.3, 1

# epochs=500, batch=3  [ 9, 30, 210]의 예측값 :  [[10.028844    1.5168741   0.98748475]]
# epochs=500, batch=1  [ 9, 30, 210]의 예측값 :  [[10.143886    1.5950966   0.78990823]]
