import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import pickle
import os
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#load data       
df=pd.read_csv("nutrition_dataset.csv")


#Basic cleaning, removes rows with any missing data
df=df.dropna()

#____feature_engineering____

df["total_macros"]=df['fat_g']+df['carbs_g']+df['protein_g']

#net carbs(excluding fibers)
df['net_carbs']=df['carbs_g']-df['fiber_g']

#sugar ratio
df['sugar_ratio']=df['sugar_g']/(df['carbs_g']+1e-6)


#Protein to Fat ratio
df['protein_fat_ratio']=df['protein_g']/(df['fat_g']+1e-6)




#Split data correctly 
X = df[
    [
        'protein_g',
        'carbs_g',
        'fat_g',
        'fiber_g',
        'sugar_g',
        'total_macros',
        'net_carbs',
        'sugar_ratio',
        'protein_fat_ratio'
    ]
]
y=df['calories']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

#Scaling 
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Models 
lr=LinearRegression()
rf=RandomForestRegressor(n_estimators=200,random_state=42)
#training the models
lr.fit(X_train_scaled,y_train) 
rf.fit(X_train,y_train) 





#creating models_nv folder 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "..", "models_nv") 

os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "linear_model.pkl"), "wb") as f:
    pickle.dump(lr, f)

with open(os.path.join(model_dir, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf, f)

with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Saved!")

print(os.listdir(model_dir))
















