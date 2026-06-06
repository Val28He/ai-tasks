import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('/kaggle/input/competitions/mental-health-prediction-2026/train.csv')
test  = pd.read_csv('/kaggle/input/competitions/mental-health-prediction-2026/test.csv')

target = 'Depression'
y = train[target]
train_x = train.drop(columns=[target])

num_cols = ['Age','Academic Pressure','Work Pressure','CGPA',
            'Study Satisfaction','Job Satisfaction',
            'Work/Study Hours','Financial Stress']
cat_cols = [c for c in train_x.columns if c not in num_cols + ['id']]

numeric_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipe, num_cols),
        ('cat', categorical_pipe, cat_cols)
    ])

models = {
    'lr'  : LogisticRegression(max_iter=1000, random_state=42),
    'knn' : KNeighborsClassifier(n_neighbors=5),
    'svm' : SVC(kernel='rbf', probability=True, random_state=42),
    'dt'  : DecisionTreeClassifier(random_state=42),
    'rf'  : RandomForestClassifier(n_estimators=300, random_state=42)
}

best_score = -np.inf
best_model = None

for name, clf in models.items():
    pipe = Pipeline(steps=[('prep', preprocessor),
                          ('clf', clf)])
    cv = cross_val_score(pipe, train_x, y,
                         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                         scoring='roc_auc')
    score = cv.mean()
    if score > best_score:
        best_score = score
        best_model = pipe

print('Best model:', best_model.named_steps['clf'].__class__.__name__)

best_model.fit(train_x, y)

test_ids = test['id']
X_test = test.drop(columns=['id'])
preds = best_model.predict(X_test)

sub = pd.DataFrame({'id': test_ids, 'Depression': preds})
sub.to_csv('submission.csv', index=False)
print("done")
