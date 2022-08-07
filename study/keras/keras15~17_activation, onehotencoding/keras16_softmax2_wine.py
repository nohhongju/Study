from tensorflow.keras.models import Sequential          
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. 데이터 
datasets = load_wine()
#print(datasets.DESCR)           # Instances 행이 150개이다 Attributes 속성, 컬럼이 13개이다. class는 3개 여기선 class0,1,2 3종류로 그냥 구분해놓음
#print(datasets.feature_names)   # 컬럼 이름 확인                                                        class_0 = 0 , class _1 = 1, class_2 =2

x = datasets.data   # 싸이킷런에서 이런식으로 제공해서 이렇게 분리하는거다. 붙여서 제공해줬기 때문에
y = datasets.target

#print(x.shape, y.shape)     # (178,13) (178,)
#print(np.unique(y))         # [0 1 2] 3개인거확인

# one hot encoding 3가지 방법이 있다.
y = to_categorical(y)
#print(y.shape)  # 178에 3으로 바뀌어져 있다.
#print(np.unique(y))        # [1,0,0],[0,1,0],[0,0,1]로 바뀌어져있다.

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)



#2. 모델링 모델구성
model = Sequential()
model.add(Dense(120, activation='linear', input_dim=13))    
model.add(Dense(100, activation='linear'))   
model.add(Dense(80, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(40, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax'))   

#회귀모델 activation = linear (default값) 이진분류 sigmoid 다중분류 softmax 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)

hist = model.fit(x_train,y_train,epochs=10000, batch_size=1, verbose=1,validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)   
print('loss : ', loss[0])          # loss :  0.1730395406484604
print('accuracy : ', loss[1])       # accuracy :  0.9166666865348816

results = model.predict(x_test[:7])
print(x_test[:7])
print(y_test[:7])
print(results)