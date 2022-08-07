from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#훈련과 평가를 7:3으로 나누는데 임의로 나누어지게 완성지켜라
#1. 데이터 정제작업 
x = np.array(range(100))            
y = np.array(range(1,101))          

'''
x_train = random.sample(list(x), 30) x데이터 값들중 30개를 중복없이 뽑는다. sample기능
x_test = [x for x in x if x not in x_train] 구글링해온 명령어 x에서 x_train인것들을 뺀다.
y_train = list(x_train+int(1))  x_train의 모든 값들에 1씩 더해주려던 시도.
#y_test = 내가 작업하던곳... 근데 난 랜덤난수가지 생각못해서 할때마다 값이 바뀌었을거 같다.
랜덤난수 --> 하나의 Train-test set에서 여러번 훈련 돌려가면서 weight측정할때 오차 없게하기 위해 
랜덤난수 없이 반복훈련하면 다른 Train-test set 작업하는거랑 다를게없다 쉽게 말해서
x_train = [1 3 5 7 9] x_train = [2 3 4  5 6 ] compile할때마다 train값이 바뀌어서 그전의 측정값들과
아무 연관이 없어서 실험하는 의미가 없다.
x,y를 train과 test로 원하는 비율로 나누고 값들을 랜덤하게 뽑아주는 작업까지 모두 한번에
from sklearn.model_selection import train_test_split 이 기능을 가져와서 쓸수있다.
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, shuffle=True, random_state=66)  # 0.9 - train 9, test 1 의 비율 
'''train_size => test와 train의 비율
    shuffle => default=True, split을 해주기 전에 섞을건지 여부. 보통은 default 값
    random_state => set를 shuffle할 때 int 값만큼 섞으며, 하이퍼파라미터를 튜닝시 이 값을 고정해두면 매번 데이터 set이 바뀌는 것을 방지'''
# print(x_train, x_test, y_train, y_test)
'''x_train : [25 18 26 29 66 50 80 45 38 58 49 85 94 87 15  3 14 33 23 24 99 48 70 43
            72 83 82 84 92 54 10 86 11 37 61  1 28 36 71 46 69 22 17 55 53 30  7 67
            63 89 19 95 62 78 65 42 98 64 35 34 76 57 44  9 40 56 12 59 81  6  2 32
            91 31 47 75 27 13 39 16 96 74 97 21 90 79 77 51 60 20] 
    x_test : [ 8 93  4  5 52 41  0 73 88 68] 
    y_train : [ 26  19  27  30  67  51  81  46  39  59  50  86  95 88  16   4  15  34
            24  25 100  49  71  44  73  84  83  85  93  55  11  87  12  38  62   2
            29  37  72  47  70  23  18  56  54  31   8  68  64  90  20  96  63  79
            66  43  99  65  36  35  77  58  45  10  41  57  13  60  82   7   3  33
            92  32  48  76  28  14  40  17  97  75  98  22  91  80  78  52  61  21] 
    y_test : [ 9 94  5  6 53 42  1 74 89 69]'''
#랜덤 난수 넣어준다 -> 훈련을 반복해도 동일한 값이 나와야 제대로 된 훈련이 가능하기때문. 
#이게 없으면 한번 다시돌릴때마다 x_train~~y_test 값이 계속 바뀐다. 

#2. 모델링
model =  Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) #evaluate에서 나온값들을 loss에 담는다. loss가 저거라는뜻이 아니다.
print('loss: ', loss) # 결과값에서 나온 loss?     loss:  7.505196208512643e-06 
result = model.predict([150])
print('[100]의 예측값 : ', result)  # [100]의 예측값 :  [[150.99577]]
