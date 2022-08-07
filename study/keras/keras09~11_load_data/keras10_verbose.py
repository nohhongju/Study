from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score    # r2_score 구하는 공식? 및 작업을 다 해놓은걸  import해서 가져다쓴다 
import time

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
model.compile(loss='mse', optimizer='adam')      

start = time.time()
model.fit(x,y,epochs=1000, batch_size=1, verbose=3)
end = time.time() - start
print("걸린시간 : ", end)  # 걸린시간 :  12.735913753509521
# verbose = 훈련 보여줄지 안보여 줄지 체크하는 역할
'''
0 안보여줌  2.3366405963897705
1 다보여줌  3.520465850830078
2 loss까지  2.8723676204681396
3~ epoch만 보임 훈련횟수 ㅇㅇ
사람이 진행정도를 확인할 수 있게 정보표기를 컨트롤 할수 있게해준다. 편의 기능 ㅇㅇ 
또한 불필요한 정보를 안보이게 해줌으로써 시간 단축의 역할도 한다.
'''
''' 1 - Epoch 1000/1000
        5/5 [==============================] - 0s 3ms/step - loss: 0.3924
    2 - Epoch 1000/1000
        5/5 - 0s - loss: 0.3966 - 13ms/epoch - 3ms/step
    3 - Epoch 1000/1000 '''


#4. 평가, 예측 
loss = model.evaluate(x,y) # 평가해보는 단계. 이미 다 나와있는  w,b에 test데이터를 넣어보고 평가해본다.
print('loss : ', loss)  # loss :  0.3800542652606964

y_predict = model.predict(x) #y의 예측값은 x의 테스트값에 wx + b 

r2 = r2_score(y,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)  # r2스코어 :  0.8099728739706379

# r2스코어 :  0.8096227565995434
# r2스코어 :  0.8099396914282224
# r2스코어 :  0.809914410101338
# r2스코어 :  0.8099800601232573
# r2스코어 :  0.8099735798526126
