from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# R2 0.62이상, train_test set 0.7
#1. 데이터 정제
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

#2. 모델링 
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=30, batch_size=10, verbose=2)
# loss: 3239.1802

#4. 평가 , 예측
y_predict = model.predict(x_test) 

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)  # r2스코어 :  0.5085119329758198

# r2스코어 :  0.6205114113675493
# r2스코어 :  0.6255249959536848
