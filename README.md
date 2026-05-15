# Nutrition Calorie Predictor ML Project 

A Machine Learning Project that predicts food calories based on their Nutritional Information such as protein, carbohydrates, fat, fiber, and sugar content.

In this project I have compared the performance of Linear Regression and Random Forest Regression models and I've also added feature engineering(creating new meaningful features from existing data to help the models learn patterns better), model evaluation (sklearn.metrics), feature importance visualization , and model serialization using pickle.

# The Project includes:
- Data Cleaning 
- Feature Engineering
- Feature Scaling 
- Model training 
- Model evaluation
- Feature Importance Visualization
- Saving/loading trained Models

# Technologies and libraries used:
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pickle

# Dataset Features:

Original Features:

- protein_g
- carbs_g
- fat_g
- fiber_g
- sugar_g

Engineered Features:

- total_macros
- net_carbs
- sugar_ratio
- protein_fat_ratio 

Note:
An engineered feature called `estimated_calories` was intentionally removed from the final model because it directly approximated the target variable (`calories`). Removing it prevented data leakage and allowed the models to learn more meaningful nutritional relationships.

# Target Variable:
calories 

# Feature Engineering 

Created additional features to help the models learn the patterns better and capture deeper nutritional relationships.

- total_macros:
Combines protein, carbohydrates, and fat into a single feature.
- net_carbs:
Represents digestible carbohydrates by subtracting fiber(non-digestible carbohydrate) from total carbs.
- sugar_ratio:
Measures how much of the carbohydrate content comes from sugar.
- protein_fat_ratio:
Compares protein content relative to fat content. 

# Models Used
- Linear Regression:
A statistical model that assumes a linear relationship between features and calorie values.

- Random Forest Regressor:
An ensemble learning model using multiple decision trees for prediction.

# Conclusion:
Linear Regression (R2 score = 0.93) slightly outperformed Random Forest (R2 score = 0.92), suggesting that the relationship between nutritional features and the target variable (`calories`) in the dataset is largely linear.

This aligns with nutritional science because calorie values are strongly influenced by macronutrient composition such as proteins, carbohydrates, and fats.

# Feature Importance Visualization:
Included a customized Matplotlib horizontal bar chart showing Random Forest feature importance scores.

# Visualization features include:

- Viridis color palette
- Grid styling
- Value labels
- Clean axis formatting

# Model Saving with Pickle:
Trained models and the scaler are saved using Pickle serialization.

Saved files:
- linear_model.pkl
- rf_model.pkl
- scaler.pkl
*This allows the trained models to be reused later without retraining

Project Structure:

LINEAR_REGR_FROM_SCRATCH/
│
├── train.py
├── evaluate.py
├── nutrition_dataset.csv
│
├── models_nv/
│   ├── linear_model.pkl
│   ├── rf_model.pkl
│   └── scaler.pkl
│
└── README.md

# How to Run:
1. Install dependencies
pip install pandas numpy matplotlib scikit-learn
2. Train the models
```bash
python train.py
```
3. Evaluate the models
```bash
python evaluate.py
```

# Key Learning Outcomes:

Throughout this Project I learned about:

- Machine learning workflows
- Feature Engineering techniques
- Model Evaluation metrics 
- Feature Importance Interpretation
- Data Preprocessing and scaling 
- Model serialization using Pickle
- Data visualization with Matplotlib

Future Improvements:
- Integrating a nutrition API for real-time food data
- Building a web application for user interaction
- Expanding the dataset with more food categories
- Testing additional machine learning models

# Author 

Asal Rajabzadeh Shateri
Second-year Computer Science Student at TMU