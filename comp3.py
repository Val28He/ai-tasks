import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import re
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/competitions/vacancies-2026/train.csv')
test = pd.read_csv('/kaggle/input/competitions/vacancies-2026/test_x.csv')

possible_id_cols = [col for col in test.columns if col.lower() in ['id', 'user_id', 'идентификатор', 'номер', 'identifier']]
id_col = possible_id_cols[0] if len(possible_id_cols) > 0 else test.columns[0]

test[id_col] = pd.to_numeric(test[id_col], errors='coerce')
original_len = len(test)
test = test.dropna(subset=[id_col]).reset_index(drop=True)

def extract_salary_mentions(text):
    if pd.isna(text):
        return np.nan
    patterns = [
        r'от\s*(\d{1,4}[^\s]*\s*т\.?р\.?)',
        r'до\s*(\d{1,4}[^\s]*\s*т\.?р\.?)',
        r'(\d{1,4}[^\s]*\s*т\.?р\.?)',
        r'(\d{1,4}\s*тыс\.?\s*руб\.?)',
        r'(\d{1,6}\s*руб\.?)',
        r'зарплата\s*от\s*(\d{1,4}[^\s]*\s*т\.?р\.?)',
        r'оклад\s*от\s*(\d{1,4}[^\s]*\s*т\.?р\.?)',
        r'зарабатывать\s*от\s*(\d{1,4}[^\s]*\s*т\.?р\.?)',
        r'(\d{1,4})\s*т\.?р\.?',
        r'(\d{1,6})\s*руб\.?'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, str(text), re.IGNORECASE)
        if matches:
            match_str = matches[0]
            if any(k in match_str.lower() for k in ['т.р.', 'тыс.', 'т']):
                num_str = re.sub(r'[^\d.,]', '', match_str).replace(',', '.')
                try:
                    return float(num_str) * 1000
                except:
                    continue
            else:
                num_str = re.sub(r'[^\d.,]', '', match_str).replace(',', '.')
                try:
                    return float(num_str)
                except:
                    continue
    return np.nan

text_cols = ['raw_description', 'raw_branded_description', 'lemmaized_wo_stopwords_raw_description', 'lemmaized_wo_stopwords_raw_branded_description']
existing_text_cols = [col for col in text_cols if col in train.columns]

for col in existing_text_cols:
    train[col] = train[col].fillna('').astype(str).replace('Не указано', '')
    test[col] = test[col].fillna('').astype(str).replace('Не указано', '')

for col in existing_text_cols:
    train[f'salary_extract_{col}'] = train[col].apply(extract_salary_mentions)
    test[f'salary_extract_{col}'] = test[col].apply(extract_salary_mentions)

extracted_cols = [col for col in train.columns if 'salary_extract_' in col]
if not extracted_cols:
    train['salary_extract_max'] = np.nan
    test['salary_extract_max'] = np.nan
else:
    train['salary_extract_max'] = train[extracted_cols].max(axis=1)
    test['salary_extract_max'] = test[extracted_cols].max(axis=1)

def has_salary_keywords(text):
    if pd.isna(text):
        return 0
    keywords = ['зарплата', 'оклад', 'вознаграждение', 'зарабатывать', 'доход', 'до 150', 'от 150', 'т.р.', 'тыс. руб.', 'руб.', 'высокооплачиваемая']
    return int(any(kw in str(text).lower() for kw in keywords))

base_desc_col = 'raw_description' if 'raw_description' in train.columns else train.columns[1]
train['has_salary_keywords'] = train[base_desc_col].apply(has_salary_keywords)
test['has_salary_keywords'] = test[base_desc_col].apply(has_salary_keywords)

categorical_cols = [
    'experience_name', 'schedule_name', 'unified_address_city', 'unified_address_state',
    'unified_address_region', 'unified_address_country', 'specializations_profarea_name',
    'professional_roles_name', 'employment_name', 'employer_industries', 'name_clean'
]
categorical_cols = [col for col in categorical_cols if col in train.columns and col in test.columns]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)]).fillna('unknown')
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str).fillna('unknown'))
    test[col] = le.transform(test[col].astype(str).fillna('unknown'))
    label_encoders[col] = le

bool_cols = ['accept_handicapped', 'accept_kids', 'if_foreign_language', 'is_branded_description']
bool_cols = [col for col in bool_cols if col in train.columns and col in test.columns]

for col in bool_cols:
    train[col] = train[col].astype(str).replace('Не указано', '0').replace('nan', '0')
    test[col] = test[col].astype(str).replace('Не указано', '0').replace('nan', '0')
    train[col] = pd.to_numeric(train[col], errors='coerce').fillna(0).astype(int)
    test[col] = pd.to_numeric(test[col], errors='coerce').fillna(0).astype(int)

numerical_features = ['salary_extract_max', 'has_salary_keywords'] + bool_cols + categorical_cols
if 'employer_id' in train.columns:
    numerical_features.append('employer_id')

X_train = train[numerical_features].copy()
y_train = train['salary_mean_net'].copy()

train_clean_mask = y_train.notna()
X_train = X_train[train_clean_mask]
y_train = y_train[train_clean_mask]

train_medians = X_train.median().fillna(0)
X_train = X_train.fillna(train_medians)

test_features = test[numerical_features].copy()
test_features = test_features.fillna(train_medians)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(test_features)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5, n_jobs=-1)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

lr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

pred_test_lr = lr.predict(test_scaled)
pred_test_rf = rf.predict(test_features)
pred_test_gb = gb.predict(test_features)

pred_test_ensemble = 0.2 * pred_test_lr + 0.4 * pred_test_rf + 0.4 * pred_test_gb

pred_test_ensemble = np.maximum(pred_test_ensemble, 0)
pred_test_ensemble = np.round(pred_test_ensemble).astype(int)

submission = pd.DataFrame({
    'id': test[id_col].astype(int),
    'salary_mean_net': pred_test_ensemble
})

submission.to_csv('submission.csv', index=False)
print("Сохранена как 'submission.csv'")
