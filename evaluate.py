import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor

#load data       
df=pd.read_csv("nutrition_dataset.csv")


#Basic cleaning
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

#Load the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lr_path=os.path.join(BASE_DIR,"..","models_nv","linear_model.pkl")
rf_path=os.path.join(BASE_DIR,"..","models_nv","rf_model.pkl")
scaler_path=os.path.join(BASE_DIR,"..","models_nv","scaler.pkl")
def load_model(path):
    return pickle.load(open(path,"rb"))
    
lr: LinearRegression=load_model(lr_path)
rf: RandomForestRegressor=load_model(rf_path)
scaler=load_model(scaler_path)
#Scale for lr
X_test_scaled=scaler.transform(X_test)

#Predictions
rf_pred=rf.predict(X_test)
lr_pred=lr.predict(X_test_scaled) 

#evaluate function
def evaluate_model(name,y_true,y_pred):
    mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred) 
    r2=r2_score(y_true,y_pred) 
    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.2f}")

#Evaluate both models
evaluate_model("Linear Regression",y_test,lr_pred)
evaluate_model("Random Forest",y_test,rf_pred)

#feature_importance_visualization 

importance=rf.feature_importances_
features=X.columns

#Sort features by importance 
indices=np.argsort(importance)


#Create the Figure
plt.figure(figsize=(12,7))

#Color customization 
colors=plt.colormaps['viridis'](np.linspace(0.2,0.8,len(features)))


bars=plt.barh(features[indices],importance[indices],color=colors)
plt.xlabel("Importance Score",fontsize=12)
plt.ylabel("Features",fontsize=12)
plt.title("Random Forest Feature Importance",fontsize=16,fontweight='bold')

#Adding Grid
plt.grid(axis='x',linestyle='--',alpha=0.5)

#Removing top/right borders
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)

#Add value Labels 
for bar in bars:
    width=bar.get_width()
    plt.text(width+0.001,bar.get_y()+bar.get_height()/2,f"{width:.3f}",va="center")

plt.tight_layout()
plt.show()
