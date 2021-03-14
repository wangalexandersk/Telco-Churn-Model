# Churn Model - Prediction - Telco Subscribers

import pandas as pd
import numpy as np
import csv
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df = pd.read_csv('week5-Churn_Dataset.csv')
clean_df = df.drop('Cust_id', axis = 1)
clean_df.dropna(subset = ["Age"], inplace=True)

for column in clean_df.columns:
    if clean_df[column].dtype == np.number:
        continue
    clean_df[column] = LabelEncoder().fit_transform(clean_df[column])

clean_df = clean_dataset(clean_df)

cols = df.columns[df.dtypes.eq('int64')]
clean_df[cols] = clean_df[cols].astype('int')

X = clean_df.drop('churnid', axis = 1) 
y = clean_df['churnid']

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 256)

rf = RandomForestClassifier(n_estimators=50, max_depth=25,
                              random_state=256)
rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)

y_pred = rf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

