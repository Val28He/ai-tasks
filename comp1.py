import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge 

names = ['X1 transaction date',
'X2 house age',
'X3 distance to the nearest MRT station',
'X4 number of convenience stores',
'X5 latitude',
'X6 longitude',
'Y house price of unit area']

train = pd.read_csv('/kaggle/input/competitions/linear-regression-competition-2026/prices_train.csv', header=None, skiprows=1).iloc[:, 1:]
test  = pd.read_csv('/kaggle/input/competitions/linear-regression-competition-2026/prices_test.csv',  header=None, skiprows=1).iloc[:, 1:]

train.columns = names
test.columns = names[:-1]

X_train = train.drop(columns=['Y house price of unit area'])
y_train = np.log1p(train['Y house price of unit area'])

X_test = test.copy()

num_cols = X_train.columns
for df in (X_train, X_test):
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

def geo_feats(df):
    center = np.array([25.0330, 121.5654])
    df = df.copy()
    df['dist_to_center'] = np.hypot(df['X5 latitude']  - center[0], df['X6 longitude'] - center[1])
    df['log_dist_MRT']   = np.log1p(df['X3 distance to the nearest MRT station'])
    df['stores_per_age'] = df['X4 number of convenience stores'] / (1+df['X2 house age'])
    df['lat_lon'] = df['X5 latitude'] * df['X6 longitude']
    df['date_sq'] = df['X1 transaction date']**2
    return df

X_train = geo_feats(X_train)
X_test  = geo_feats(X_test)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', Ridge(alpha=1))  
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
pred = np.expm1(pred)

submit = pd.DataFrame({'index': np.arange(len(pred)), 'Y house price of unit area': pred})
submit.to_csv('submission.csv', index=False)
