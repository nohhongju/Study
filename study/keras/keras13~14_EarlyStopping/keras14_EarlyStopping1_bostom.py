# 과적합 예제

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import EarlyStopping    # 웬만한 인공지능은 이미 다 구현이 되어있다. import해서 쓰면 된다. 


#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data
y = datasets.target

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

es = EarlyStopping  #정의를 해줘야 쓸수있다. 
es = EarlyStopping(monitor="val_loss", patience=10, mode='min', verbose=1,baseline=None, restore_best_weights=True)
# 멈추는 시점은 최소값.발견 직후 patience값에서 멈춘다. 
# 숙제.
# 단 이때 제공되는 weight의 값은 최저점 val_loss에서의 weight값일까 아니면 마지막 trainig이 끝난후의 weight값일까? 
# Tensorflow.org 접속후 EarlyStopping 검색 후 깃허브에 있는 문서의 설명을 보면 설정 파라미터가 나와있다.
# restore_best_weights 라는 옵션이 있는데 이 값에 True,False를 넣어줘서 저장할 weights 값을 선택할 수 있다.
# False일 경우 마지막training이 끝난 후의 weight값을 저장하고 True라면 training이 끝난 후 값이 가장 좋았을때의 weight로 복원한다.
# 또한 baseline 옵션을 사용하여 모델이 달성해야하는 최소한의 기준값을 선정할수도있다.

start = time.time()
hist = model.fit(x_train,y_train,epochs=10000, batch_size=1,validation_split=0.25, callbacks=[es]) 
end = time.time() - start
#print("걸린시간 : ", round(end, 3), '초')

#4. 평가 , 예측
loss = model.evaluate(x_test,y_test)
# print(hist.history['val_loss'])
print('loss : ', loss)

# y_predict = model.predict(x_test)
# print("최적의 로스값 : ", y_predict)

# r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
# print('r2스코어 : ', r2)


# plt.figure(figsize=(9,6)) # 판 깔고 사이즈가 9,5이다.
# plt.plot(hist.history['loss'], marker=".", c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid() # 그림그렸을때 격자를 보여주게 하기 위해 , 모눈종이 역할?
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right') # 그림그렸을때 나오는 설명? 정보들 표시되는 위치
# plt.show()
