from sklearn.datasets import fetch_covtype              # 데이터 입력받음. 
from tensorflow.keras.models import Sequential          # 모델링 모델 -> Sequential 
from tensorflow.keras.layers import Dense               # 레이어 -> Dense
import numpy as np                                      # 데이터 numpy이용해서 처리
from sklearn.model_selection import train_test_split    # 데이터 정제작업 도와주는 함수
from tensorflow.keras.callbacks import EarlyStopping    # 데이터 훈련 자동으로 멈추게해주는 함수.

# 여기서부터 이제 3가지의 onehotencoding 도와주는 함수가 있다 
#1. from tensorflow.keras.utils import to_categorical       y라벨 값을 0부터순차적으로 끝까지 변환해준다. 0 1 2 3 4 5...
from sklearn.preprocessing import OneHotEncoder            #y라벨 값을 유니크값만큼만 변환해준다          1 2 4 6 8...
#3. from pandas import get_dummies                          y라벨 값을 유니크값만큼만 변환해주는데 print y해보면 라벨값이랑 인덱스정보가 들어가 있다.

#1. 데이터 
datasets = fetch_covtype()
#print(datasets.DESCR)           # 데이터셋 및 컬럼에 대한 설명 
#print(datasets.feature_names)   # 컬럼,열의 이름들                                                      

x = datasets.data   # 싸이킷런에서 이런식으로 제공해서 이렇게 분리하는거다. 붙여서 제공해줬기 때문에
y = datasets.target

#print(type(y))       엄밀히 말하면 y는 지금 <class 'numpy.ndarray'> 타입이다. 타입 확인.

#print(x.shape, y.shape)     # (581012,54) (581012,) 행과 열의 개수를 확인할수 있다
#print(np.unique(y))         # [1 2 3 4 5 6 7] 7개인거확인 이게 뭘 의미하냐 이 모델은 다중분류형태로 모델링 해줘야 한다~

# 여기서 이제 3가지 방법중 1가지를 이용해 onehotencoding해준다.
# catogorical은 (581012, 8)로 해주고 -> 0 1 2 3 4 5 6 7 8 이렇게 판단하고 바꿔주는거 같다.
# 싸이킷런의 OneHotEncoder은 (581012, 7)로 해준다. 
# 판다스의 get_dummies는 변환과 더불에 y값을 출력해보면 행의 개수와 y칼럼은 유니크별로 깔끔하게 다 정리까지 해준다.

# print(y.shape)   # (581012,) -> (581012, 7) or (581012, 8)로 softmax 넣으면 되겠지만 상식적으로 8 선택해야 할 이유가없다 

#y = to_categorical(y)   # onehotencoding해줘서 배열형태로 변환후 몇개의 값만 확인해보자.    해주는 이유는? 모든 값들을 동일값 1로 통일시켜서 모델이 공정하게(?) fit 학습시키려고
#print(y[:10]) #[0. 0. 0. 0. 0. 1. 0. 0.],[0. 0. 0. 0. 0. 1. 0. 0.],[0. 0. 1. 0. 0. 0. 0. 0.] 각각의 y값이 배열(?) 형태로 변환된것을 확인할 수 있다.
# 자리로는 [1,0,0,0,0,0,0,0] 8자리를 차지하는데 유니크값은 7개이다 
# ------ 완전 착각한 거였다. to_categorical의 오류? 문제점? 였고 categorical 방법말고 다른 방법 써서 onehotencoding해줘야한다. ----------
# from sklearn.preprocessing import OneHotEncoder 함수 가져오고 earlystopping처럼 enco = ~~로 정의해주고 쓰면 된다 이제.
# 무조건 유니크값 기준으로 개수 생각하고 변환도 그렇게 해줘야한다. 
# 이 문제는 판다스나 싸이킷런 써서 (581012, 7)로 해서 푸는게 더 좋다.

enco = OneHotEncoder(sparse=False)         # sparse=True가 디폴트이며 이는 Matrix를 반환한다. 원핫인코딩에서 필요한 것은 array이므로 sparse 옵션에 False를 넣어준다.
y = enco.fit_transform(y.reshape(-1,1))    # 2차원변환 해주기 위해 행의 자리에 -1넣고 열이1개라서 1넣은거다 그러면 세로베열된다. 가로배열은(1,-1)이다.

#print(y.shape)  # 바뀐거확인.              # (581012, 7)로 잘 바뀐걸 확인 할 수 있다    

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)    # 데이터 정제

# 숙제하기 위해 print(len(x_train))해서 길이를 구해보고 가자
#z = [1,2,3,4,5]  print(len(z)) 혹시 모르니까 5개는 5개로 출력 되는거 확인.
#print(len(x_train))    464809 개 인걸 확인할 수 있다.


#2. 모델링 모델구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=54))    
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))   
model.add(Dense(10))
model.add(Dense(7, activation='softmax'))  
 
#회귀모델 activation = linear (default값) 이진분류 sigmoid 다중분류 softmax 
# 행이 540000개이길래 500000만개부터 5만개씩줄여서 50만 40만 30만 20만 10만 이렇게 하려고했는데  cpu가 절대못한다고 뭐라고 함.;;


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=1000000,batch_size=100, verbose=1,validation_split=0.2, callbacks=[es])    

#batch_size 통째로 빼 보고 해보기 디폴트 사이즈 몇인지 알아보기  로딩되는 과정에서 1epcoh에 값이 몇인지 확인해보기. 데이터 수 / batch사이즈 값만큼 반복한다. 나머지도 1번으로 계산한다.
# 464809 * 0.8이 train 데이터고 나머지 0.2가 validation 데이터인데 train데이터만 fit에 들어가기때문에 train데이터의 개수를 구해야한다.
# 계산기로 464809 * 0.8 해보면 371,847.2 -> 371847 or 371848 둘중 하나 
# 실행시켜보면 37~~~~ 값만큼 반복하지는 않고 11621만큼 반복한다 이 값을 나눠보면 bacth_size의 default값을 구할 수 있다 
# 아마 train데이터 개수가 딱 맞아떨어지지 않아서 11621 반복값은 나머지 연산 + 1이 들어간 값일 것이다. 따라서 11620으로 나눠줘야한다.
# case1 371847 / 11620 = 32.00060240963855
# case2 371848 / 11620 = 32.00068846815835
# train데이터의 개수가 1개정도 차이난다고 쳐도 batch사이즈 기본값은 32이고 거기에 나머지가 좀 남은거 1번 연산 더해서 11621이 나온거 같다.
# batch_size: 정수 혹은 None. 경사 업데이트 별 샘플의 수. 따로 정하지 않으면 batch_size는 디폴트 값인 32가 됩니다. 
# 구글링 해봤는데 내 계산이 맞았다. 확인 끝났으면 batch_size=10000정도로 줘서 일단 해본다.

#print(len(x_train)) #model.fit에서 다시 train과 validation으로 나눠주니까 여기서 측정하면 나눠진후의 x_train값이 나올줄 알았는데 464809가 나왔다.
# model.fit안에서 자체적으로 나눠서 계산해주고 그 밖까지 값이 저장되지는 않는거같다.


#4. 평가
loss = model.evaluate(x_test,y_test)   
print('loss : ', loss[0])          #batch_size : 100  loss     :  0.6371333599090576
print('accuracy : ', loss[1])      #batch_size : 100  accuracy :  0.7232945561408997

#5. 예측
results = model.predict(x_test[:15])
print(y_test[:15])
print(results)


