# callbacks 함수의 EarlyStopping 기능을 이용하여 diabets 데이터 모델링해보기 
# 숙제 및 지금까지 배운내용 총 정리.

from tensorflow.keras.models import Sequential          # 신경망 모델링할 모델의 종류 Sequential
from tensorflow.keras.layers import Dense               # 모델링에서 사용할 레이어의 종류 Dense 
from sklearn.datasets import load_diabetes              # 싸이킷런 라이브러리의 datasets클래스의 diabets함수 불러옴
from sklearn.model_selection import train_test_split    # 데이터를 train과 test로 0.0~1.0 사이의 비율로 분할 및 랜덤분류 기능
from sklearn.metrics import r2_score                    # y_predict값과 y_test값을 비교하여 점수매김. 0.0~1.0 및 - 값도 나옴.
import matplotlib.pyplot as plt                         # 데이터를 시각화 시켜주는 기능.
import time
from tensorflow.keras.callbacks import EarlyStopping    # training 조기종료를 도와주는 기능 여러 옵션들이 있다.

#1. 데이터 로드 및 정제
datasets = load_diabetes()
x = datasets.data
y = datasets.target

#print(x.shape)  442행 10열 input_dim 값 10 칼럼,특성,피쳐가 10개.
#datasets.feature_names     컬럼,열의 값들 확인가능.
#datasets.DESCR             데이터 및 컬럼에 대한 설명

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=66)    #정제작업 

#2. 모델링  여러번해서 좋은 값 찾아야함.
# 모델링을 못해서 그런가 값이 잘 안나옴;
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# 컴파일이 정확히 뭘 해주는거지?            <------------------------------------------------------------------------------------  질문할거 
# compile은 정확히 뭘 해주는 기능이지? loss , optimizer는 정확히 뭘 의미하는걸까  fit 들어가기전에 준비 ,설계 해주는 느낌


es = EarlyStopping  # 정의를 해줘야 사용가능하다
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
# val_loss를 관측하고 50번안에 최저값이 갱신되지 않으면 훈련을 중단하고 가장 좋았을때의 "weights"값을 복원하여 기록(?)합니다.
# 컴파일해보면 마지막에 Restoring model weights from the end of the best epoch. 라는 메시지를 출력시켜준다. 안심할수 있다.
# baseline = None ,  모델이 달성해야하는 최소한의 기준값,정확도(?)을 선정합니다. 정확도를 측정하는 기준은? 0.0~1.0?? 이게 뭘 의미하는건지.  <-------------------------------------------   질문할거 
# True값 넣고 evaluate했을때  1.loss: 3447.219482421875  r2: 0.5668603667030361  2.loss: 3641.060302734375   r2: 0.5425044431962383   3.loss: 3614.39453125  r2: 0.5458550011751824
# False값 넣고 evalutae했을때 1.loss: 3701.3193359375    r2: 0.5349329860627352  2.loss: 3321.02294921875    r2: 0.5827168650569212   3.loss: 3441.029541015625   r2: 0.5676381480025825
# 왠지는 모르겠는데 그렇게 큰 차이는 없다. 
# 큰 차이가 없는 이유. EarlyStopping은 최적의 weights값을 복원해서 저장한다. <-- 기록하고 저장해서 evaluate 할때 최적의 값으로 계산한다.
# 값을 저장하려면 ModelCheckpoint 함수를 써야한다.


hist = model.fit(x_train,y_train,epochs=10000, batch_size=1, validation_split=0.111111, callbacks=[es]) # loss와 똑같이 관측하기 위해 일단 저장.
# callbacks함수의 설정은 [es]안에 들어있다는 뜻인가? [ ]로 감싸주는 이유는?    [] 리스트잖아. 2개이상 값을 넣어주려고 나중에 modelcheckpoint 같은거 쓰려구

# hist 안에 수 많은 정보들이 담긴다.
# print("-------------------------------------------")
# print(hist)   # 자료형이 나온다.
# print("-------------------------------------------")
# print(hist.history)  # loss 값과 var_loss값이 dic형태로 저장되어 있다. epoch 값만큼의 개수가 저장되어 있다 ->> 1epoch당 값을 하나씩 다 저장한다.
# print("-------------------------------------------")
# print(hist.history['loss']) # hist.history에서 loss키 값의 value들을 출력해준다.
# print("-------------------------------------------")
# print(hist.history['val_loss']) # hist.history var_loss키 값의 value들을 출력해준다.
# print("-------------------------------------------")


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)  # loss라는 이름안에 evaluate의 값들을 저장. loss가 따로있는게 아니다.
# 컴파일 단계에서 도출된 weight 및 bias값에 xy_test를 넣어서 평가만. 해보고 거기서 나온 loss값들(?)을 저장.


print('평가만 해본 후 나온 loss의 값 : ', loss)
# val_loss와 loss의 차이가 적을수록 validation data가 더 최적의 weights를 도출시켜줘서 실제로 평가해봐도 차이가 적게 나온다는 말이므로 차이가 적을수록 좋다.
#model.evaluate도  model.fit처럼 수많은 값들을 loss안에 담아주는 줄 알았다 근데 보려고했더니
#print(loss.history) loss도 hist처럼 history볼수 있을줄 알았는데 'float' object has no attribute 'history' 라고 나온다.  <---------------------------------------------------------------------------- 질문할거


y_predict = model.predict(x_test) # 그냥 저장할이름이 y_predict인것. predict함수로 x_test를 예측했을때 나올 값들을 저장. evaluate는 y_test와 비교해서 loss를 뽑았다면 여기는 예측값을 저장.

r2 = r2_score(y_test,y_predict) #원본값을 앞에 두고 r2_score측정
print('r2스코어는', r2)


# 여기서부터는 시각화, 그림그려주는 작업
plt.figure(figsize=(9,6)) # 판 깔고 사이즈가 9,6이다.
plt.plot(hist.history['loss'], marker=".", c='red', label='loss') #plot 선을 보여준다 scatter 점찍다
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')  # 점찍을 데이터 valloss 점 형태 marker= " . " label은 이 점들이 뭐냐는 설명 -> 'val_loss'라고 입력해줌
plt.grid() # 그림그렸을때 격자를 보여주게 하기 위해 , 모눈종이 역할?
plt.title('loss')   # 제목
plt.ylabel('loss')  # y축 설명
plt.xlabel('epoch') # x축 설명
plt.legend(loc='upper right') # 그림그렸을때 나오는 설명? 정보들 표시되는 위치
plt.show()

