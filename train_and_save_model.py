import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

# Load dataset
cancer_df = pd.read_csv('risk_factors_cervical_cancer.csv')

# Replace missing values
cancer_df.replace('?', np.nan, inplace=True)
cancer_df = cancer_df.apply(pd.to_numeric)

# Drop unnecessary columns
cancer_df.drop(columns=['Dx','Smokes','Smokes (packs/year)','Hormonal Contraceptives','IUD','IUD (years)','STDs','STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV','STDs: Time since first diagnosis', 'STDs: Time since last diagnosis','Dx:CIN','STDs: Number of diagnosis','Hinselmann','Schiller','Citology'], inplace=True)

# Fill missing values with mean
cancer_df.fillna(cancer_df.mean(), inplace=True)

# Separate features and target
X = cancer_df.drop(columns=['Biopsy'])
y = cancer_df['Biopsy']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

# Initialize and train the XGBoost model
xgb_model = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100, random_state=42)
xgb_model.fit(x_train, y_train)

# Save the model to a file
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

print("Model trained and saved as 'xgb_model.pkl'")
