from tensorflow.keras.models import Sequential          
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.utils import to_categorical 
 
# 1. 데이터 정제
# 싸이킷런은 데이터를 x,y 붙여서 제공한다?
datasets = load_breast_cancer()
#print(datasets)

x = datasets.data
y = datasets.target
#y = to_categorical(y)
#print(datasets.DESCR)   # describe 약자 이게 뭐해주더라..? 내용확인 
#print(datasets.feature_names)   # 이름 확인
#print(x.shape, y.shape)         # 모양분석 (569, 30) (569, )
#print(y)    # 결과값이 0,1 밖에 없는걸 보는 순간 2진분류인거 판단 + loss 값 binary cross Entropy랑 sigmoid() 함수인거 까지 자동으로 생각
#print(np.unique(y))     # [0,1] 중복값 빼고 결과값이 뭐인지 보여주는거. 분류값에서 unique한 것이 몇개있는지 뭐 있는지 보여줌.

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66) 

#2. 모델링 
model = Sequential()
model.add(Dense(30, activation='linear', input_dim=30))    #activation='linear' -> 원래값을 그대로 넘겨준다. 생략해도 되는데 그냥 공부하라고 표기함.
model.add(Dense(25, activation='linear'))   #값이 너무 큰거같으면 중간에 sigmoid한번써서 줄여줄 수도 있다.
model.add(Dense(20, activation='linear'))
model.add(Dense(15, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(5, activation='linear'))
model.add(Dense(1, activation='sigmoid'))   # sigmoid함수는 레이어를 거치며 커져버린 y=wx+b의 값들을 0.0~1.0사이로 잡아주는 역할을 한다. 
#model.add(Dense(2, activation='softmax'))  이게 크게보면 이진분류도 결국 다중분류에 속해있어서 혹시나 onehotencoding하고 softmax = 2로 잡아주고 하면 될거 같아서 해봤더니 잔짜 된다
#회귀모델 activation = linear (default값) 이진분류 sigmoid 다중분류 softmax 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    # 다중분류 softmax할거면 loss도 categorical_crossentropy로 바꿔줘야한다.
# 회귀모델을 하기위해서 mse loss가 필요하고 이진분류하기위해서 binary_crossentropy가 필요하고 다중분류를 하기위해서 categorical_crossentropy가 필요하다 그런개념.
# 각모델에 맞는 mse값들이 있다.   
# matrics=['accuracy']는 그냥 r2 스코어처럼 지표를 보여주는거지 fit에 영향을 끼치지않는다. 다른것도넣을수 있다 matrics에.
# loss가 제일 중요하다  accuracy는 그냥평가지표이다. 몇개중에 몇개맞췄는지 보여주는 지표. 설령 잘 맞췄다 해도 loss값이 크면 운으로 때려맞춘거지. 좋다고 보장할순없다.
# loss와 val_loss를 따지면 val_loss가 더 믿을만하다.

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)

hist = model.fit(x_train,y_train,epochs=10000, batch_size=1, verbose=1,validation_split=0.2, callbacks=[es])

#4. 평가, 예측

loss = model.evaluate(x_test,y_test)    # evaluate의 평가값은 1개로 귀결된다. loss를 출력해보면 값이 1개만 나온다.
print('loss : ' , loss)
# loss,accuracy :  [0.16714714467525482, 0.9298245906829834]
# loss,accuracy : [0.24145513772964478, 0.9210526347160339]

# y_predict = model.predict(x_test)
# print(x_test)
# print(y_predict)
# print(y_test)
# model.predict의 값은 모델링에 따라서 크게 영향을 받는다.  모델링 못하면 값이 전부 똑같이 나올수도 있다.

#evaluate의 값은 원래 loss 1개 나왔었다. 근데 matrics=['accuracy'] 써서 정확도를 출력하니까 갑자기 값이
#  [0.30348193645477295, 0.8859649300575256] 2개 나온다. 첫번재 값은 loss고 두번째 값은 matrics=['accuracy'] 값이 출력된다.
# [ ] 는 리스트. 안에 값을 여러개담을때 사용한다.  evaluate 값 출력했을때 첫번째 값은 무조건 loss나오고 그 후의 값은 사전에 설정해준 값들이 순차적으로 나온다.