import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 정제
x = np.array([range(10)])
print(x)
'''[[0 1 2 3 4 5 6 7 8 9]]'''
print(x.shape)  # (1, 10)
x = np.transpose(x)  #행의 개수를 맞춰준다. 
print(x)
'''[[0]
    [1]
    [2]
    [3]
    [4]
    [5]
    [6]
    [7]
    [8]
    [9]]'''
print(x.shape)  # (10, 1)

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
model.add(Dense(10, input_dim=1)) # 입력데이터(x)의 열을 넣는다.
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(12)) 
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)  # 훈련에 x,y값이 들어갔다. 훈련에 데이터의 70%정도 쓰고
# fit에 넣을 때는 두 데이터의 행이 같아야 한다.

# 4. 평가 , 예측
loss = model.evaluate(x, y)  # 훈련시킨 결과로 평가 예측을 했기때문에 성능이 좋을수밖에 없다? -> 훈련과 평가 구간을 나눠서 진행해본다.  평가에 나머지 데이터 30%를 써본다. 
print('loss : ', loss)  # loss :  0.006147603504359722
result = model.predict([11])  # 이건 이해 갔다. 이미 x값안에 9와 그에 해당하는 값이 다 들어있다 
print('[11]의 예측값 : ', result)  # [11]의 예측값 :  [[12.015717   1.6228578 -1.08259  ]]

# 실제값: 10, 1.3, 1
# epochs=2000,batch=1 [9]의 예측값 :  [[9.99683   1.5020151 1.0145528]]
# epochs=100, batch=1 [9]의 예측값 :  [[9.987495  1.5129123 1.0419812]]
# epochs=100, batch=1 [9]의 예측값 :  [[9.947364  1.2419622 1.0112629]]
#사실 이건 잘 나올수 밖에 없는 구조다. 이건 이미 훈련이 되어있는 데이터이다. 이미 답안지를 보고 답을 맞추는 느낌