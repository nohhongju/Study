from sklearn.datasets import fetch_covtype             
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense               
import numpy as np                                      
from sklearn.model_selection import train_test_split    
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical       
from sklearn.preprocessing import OneHotEncoder            
from pandas import get_dummies
from sklearn.datasets import load_wine

datasets = load_wine()

x = datasets.data   
y = datasets.target

#print(x.shape,y.shape)  (178, 13) (178,)
#print(np.unique(y)) unique값 0,1,2

#1. pandas의 get_dummies
#y = get_dummies(y)
#print(y)           #y값보면 행 0~177 열 유니크값대로 0 1 2  마지막에 [178 rows x 3 columns] 까지 표시해준다.
#print(y.shape)     #(178, 3)

#2. tensorflow의 to_categorical
#y = to_categorical(y)
#print(y)           #y값보면 그냥 그안에 담긴 값만 딱 나온다.
#print(y.shape)     #(178, 3)

#3. sklearn의 OneHotEncoder 
#enco = OneHotEncoder(sparse=False)        
#y = enco.fit_transform(y.reshape(-1,1))
#print(y)            #y값보면 그냥 그안에 담긴 값만 딱 나온다.   
#print(y.shape)      #(178, 3)

# 유니크값이 좀 특이한 데이터를 분석해보고 싶다. ex) 0,1,3,5,6,7,9 ...