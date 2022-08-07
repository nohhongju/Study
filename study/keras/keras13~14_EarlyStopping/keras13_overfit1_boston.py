# 과적합 예제
# epochs를 많이 준다고해서 무조건 좋은게 아니다. 오히려 과적합이 걸려서 loss, val_loss값이 튈 수 있다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time

#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data
y = datasets.target
'''
print(x)    # x내용물 확인
print(y)    # y내용물 확인
print(x.shape) # x형태
print(y.shape) # y형태
print(datasets.feature_names) # 컬럼,열의 이름들
print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명 
'''

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

start = time.time()
hist = model.fit(x_train,y_train,epochs=10, batch_size=1,validation_split=0.25) 
# 여기서 나오는 값들을 hist에 담는다.
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')   #round는 반올림 함수이다. end값을 소수점 3자리까지만 반올림처리하겠다는 뜻
#4. 평가 , 예측
#loss = model.evaluate(x_test,y_test)
#print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)

# r2스코어 :  0.6577537076731692

print("-------------------------------------------")
print(hist)   # 자료형이 나온다.
print("-------------------------------------------")
print(hist.history)  # loss 값과 var_loss값이 dic형태로 저장되어 있다. epoch 값만큼의 개수가 저장되어 있다 ->> 1epoch당 값을 하나씩 다 저장한다.
print("-------------------------------------------")
print(hist.history['loss']) # hist.history에서 loss키 값의 value들을 출력해준다.
print("-------------------------------------------")
print(hist.history['val_loss']) # hist.history var_loss키 값의 value들을 출력해준다.
print("-------------------------------------------")


plt.figure(figsize=(9,6)) # 판 깔고 사이즈가 9,5이다.
plt.plot(hist.history['loss'], marker=".", c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 그림그렸을때 격자를 보여주게 하기 위해 , 모눈종이 역할?
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') # 그림그렸을때 나오는 설명? 정보들 표시되는 위치
plt.show()
