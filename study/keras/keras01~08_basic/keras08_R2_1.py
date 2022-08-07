## R2가 무엇인지 찾아라! R2 scroe , R2 제곱, 
## 점수매기는 것. 선형회귀모델에 대한 적합도 측정값. 몇점짜리냐 이게 0~1사이 1점만점
## R2 score는 -값도 가질수 있다. 자세한 공식은 아직 내 레벨로는 알기에 부족하다.
## loss만 가지고 정확도를 보기에는 부족함이있어서 R2 score로 점수를 매긴다. 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt     # 그래프나 그림그릴때 많이 씀
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score    # r2_score 구하는 공식? 및 작업을 다 해놓은걸  import해서 가져다쓴다 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,9,8,12,13,17,12,14,21,14,11,19,23,25])  

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

'''
x_train 14개 x_test6개 y도 동일 

'''
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # mean squared error , 평균제곱오차    
model.fit(x_train,y_train,epochs=500, batch_size=1)

#4. 평가, 예측 
loss = model.evaluate(x_test,y_test) # 평가해보는 단계. 이미 다 나와있는  w,b에 test데이터를 넣어보고 평가해본다.
print('loss : ', loss)  # loss :  12.500420570373535

y_predict = model.predict(x_test) #y의 예측값은 x의 테스트값에 wx + b 
print(y_predict)
'''[[ 2.2432418]
    [19.47034  ]
    [ 5.6886616]
    [ 6.8371344]
    [ 4.540188 ]
    [11.431026 ]]'''
r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)  # r2스코어 :  0.12106412418345613
# 0~1사이의 값이 나오고, 1에 가까울 수로 적합하다. 실제값과 예측값을 비교하여 나온다.


''' plt.plot(x, y_predict, color='red') 가 되었을 때는 에러가 발생
    ValueError: x and y must have same first dimension, but have shapes (20,) and (6, 1)
    x축 데이터와 y축 데이터의 개수가 맞지 않으면 다음과 같은 에러가 발생합니다.'''
plt.scatter(x, y) # scatter 흩뿌리다 그림처럼 보여주다?
plt.plot(x_test, y_predict, color='red') # scatter 점찍다 plot 선을 보여준다 
# x가 아닌 x_test로 넣었을 때는 데이터 수가 같아 에러가 없이 선이 그려진다. - 눈으로 보기에는 적당에 해당된다.
plt.show() 