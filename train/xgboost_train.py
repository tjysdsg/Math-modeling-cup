import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import date

pd.set_option('display.max_columns', None)

enterprise_info = pd.read_csv('../data/1_info.csv')

enterprise_reputation = {}
for i, row in enterprise_info.iterrows():
    enterprise_reputation[row['企业代号']] = row['信誉评级']

reputation_map = {
    'A': 0,
    'B': 0,
    'C': 0,
    'D': 1
}

status_to_int = {
    '有效发票': 0,
    '作废发票': 1,
}


def preprocess(df: pd.DataFrame):
    df['date'] = pd.to_datetime(df['开票日期'])
    df['status'] = df['发票状态'].map(status_to_int).astype(np.int)
    df.drop(['开票日期', '发票状态'], axis=1, inplace=True)


    df['reputation'] = df['企业代号'].map(enterprise_reputation).map(reputation_map)
    return df


invoice_in = preprocess(pd.read_csv('../data/1_in.csv'))
print(invoice_in)

print(np.sum(invoice_in['status'].loc[invoice_in['企业代号']=='E1']))

data_good_reputation = invoice_in.loc[invoice_in['reputation'] == 0]
data_bad_reputation = invoice_in.loc[invoice_in['reputation'] == 1]

data_good_train = pd.DataFrame.sample(data_good_reputation, n=data_bad_reputation.shape[0])
data_bad_train = data_bad_reputation

feature_space = ['金额', '税额', '价税合计', 'status']

data_X = pd.concat([data_good_train[feature_space], data_bad_train[feature_space]])
print("data_X = ", data_X)
data_Y = pd.concat([data_good_train['reputation'], data_bad_train['reputation']])
print("data_Y = ", data_Y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y,
                                                    test_size=0.2, random_state=42)

# dtrain = xgb.DMatrix(X_train, label=y_train.to_numpy())
dtrain = xgb.DMatrix(data_X, label=data_Y.to_numpy())

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 90,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'slient': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

xgb_model = xgb.train(params, dtrain, 4)

dtest = xgb.DMatrix(X_test)
y_pred = xgb_model.predict(dtest)
print(max(y_pred))
print(accuracy_score(y_pred, y_test.to_numpy()))

print(np.count_nonzero(y_pred[y_test == 1] == 1) / np.count_nonzero(y_test == 1))


