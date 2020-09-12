import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)

enterprise_info = pd.read_csv('../data/1_info.csv')

enterprise_reputation = {}
for i, row in enterprise_info.iterrows():
    enterprise_reputation[row['企业代号']] = row['信誉评级']

reputation_map = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3
}

status_to_int = {
    '有效发票': 0,
    '作废发票': 1,
}


def preprocess(df: pd.DataFrame, interval=7):
    df['date'] = pd.to_datetime(df['开票日期'])
    df['week_stamp'] = pd.to_datetime(df['date']).astype(np.int64) // (10E9 * 3600 * 24 * interval)
    df['status'] = df['发票状态'].map(status_to_int).astype(np.int)

    invalid_percent = df.groupby(['企业代号', 'week_stamp']).apply(
        lambda x: pd.Series({'invalid_percent': np.count_nonzero(x['status'] == 1) / x.shape[0]}))

    df = df.merge(invalid_percent, on=['企业代号', 'week_stamp'], how='left')

    df['tax_ratio'] = df['税额'] / df['金额']
    df.drop(['开票日期', '发票状态'], axis=1, inplace=True)
    df['reputation'] = df['企业代号'].map(enterprise_reputation).map(reputation_map)
    return df


df = preprocess(pd.read_csv('../data/1_in.csv'))

# print(invoice_in)
# print(np.sum(invoice_in['status'].loc[invoice_in['企业代号'] == 'E1']))

feature_space = ['金额', '税额', '价税合计', 'status']

data_X = pd.concat([df[feature_space], df[feature_space]])
data_Y = pd.concat([df['reputation'], df['reputation']])

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

model = CatBoostClassifier()
# train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('A:', np.count_nonzero(y_pred[y_test == 0] == 0) / np.count_nonzero(y_test == 0))
print('B:', np.count_nonzero(y_pred[y_test == 1] == 1) / np.count_nonzero(y_test == 1))
print('C:', np.count_nonzero(y_pred[y_test == 2] == 2) / np.count_nonzero(y_test == 2))
print('D:', np.count_nonzero(y_pred[y_test == 3] == 3) / np.count_nonzero(y_test == 3))
print(accuracy_score(y_pred, y_test))
