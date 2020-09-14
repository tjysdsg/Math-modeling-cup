import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)

enterprise_info = pd.read_csv('../data/meta_1.csv')

categories = [
    '环保',
    '食品',
    '咨询',
    '材料',
    '广告',
    '律师',
    '房地产',
    '建筑',
    '图书',
    '物流',
    '设施',
    '工程',
    '机械',
    '贸易',
    '信息',
    '技术',
    '科技',
    '服务',
    '个体经营',
    '店',
    '药房',
    '经营部',
    '研究院',
    '有限责任',
    '有限公司',
    '国际',
]

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

feature_space = [
    'revenue',
    'income', 'cost',
    'income_std', 'cost_std',
    'income_exclude_refund', 'cost_exclude_refund',
    'income_gross_valid', 'cost_gross_valid',
    'income_tax_ratio', 'cost_tax_ratio',
    'top5_buyer_ratio',
    'income_partners', 'cost_partners',
    'category',
]


def invoice_status(df: pd.DataFrame):
    df['status'] = df['发票状态'].map(status_to_int).astype(np.int)
    return df


def safe_divde(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def preprocess(df_cost: pd.DataFrame, df_rev: pd.DataFrame, enterprise_category: dict):
    df_cost = invoice_status(df_cost)
    df_rev = invoice_status(df_rev)
    df_cost['partner'] = df_cost['销方单位代号']
    df_rev['partner'] = df_rev['购方单位代号']
    df_cost.drop('销方单位代号', axis=1, inplace=True)
    df_rev.drop('购方单位代号', axis=1, inplace=True)

    df_cost['type'] = 0
    df_rev['type'] = 1
    df = pd.concat([df_cost, df_rev])
    df['date'] = pd.to_datetime(df['开票日期'])
    groups = df.groupby(['企业代号', pd.Grouper(key='date', freq='3MS')])

    def get_feats(x: pd.DataFrame):
        firm_code = x['企业代号'].tolist()[0]

        income = x[(x['type'] == 1) & (x['status'] == 0)]
        income_refund = x[(x['type'] == 1) & (x['status'] == 0) & (x['金额'] < 0)]
        income_invalid = x[(x['type'] == 1) & (x['status'] == 1)]

        cost = x[(x['type'] == 0) & (x['status'] == 0)]
        cost_refund = x[(x['type'] == 0) & (x['status'] == 0) & (x['金额'] < 0)]
        cost_invalid = x[(x['type'] == 0) & (x['status'] == 1)]

        top5_buyer_money = np.sum(income.groupby('partner')['金额'].sum().sort_values(ascending=False).to_numpy()[:5])
        top5_buyer_ratio = safe_divde(top5_buyer_money, income['金额'].sum())

        return pd.Series({
            'firm_code': firm_code,

            'revenue': income['金额'].sum() - cost['金额'].sum(),
            'income': income['金额'].mean(),
            'cost': cost['金额'].mean(),

            'income_std': income['金额'].std(),
            'cost_std': cost['金额'].std(),

            'income_exclude_refund': income['金额'].sum() - income_refund['金额'].sum(),
            'cost_exclude_refund': cost['金额'].sum() - cost_refund['金额'].sum(),

            'income_gross_valid': income['金额'].sum() - income_invalid['金额'].sum(),
            'cost_gross_valid': cost['金额'].sum() - cost_invalid['金额'].sum(),

            'income_tax_ratio': safe_divde(income['税额'].sum(), income['金额'].sum()),
            'cost_tax_ratio': safe_divde(cost['税额'].sum(), cost['金额'].sum()),

            'income_partners': income['partner'].unique().size,
            'cost_partners': cost['partner'].unique().size,

            'top5_buyer_ratio': top5_buyer_ratio,

            'category': enterprise_category[firm_code],
        })

    feats = groups.apply(get_feats).reset_index()
    feats.fillna(0, inplace=True)
    feats.to_csv('tmp.csv')
    return feats


def main():
    enterprise_reputation = {}
    enterprise_category = {}
    for i, row in enterprise_info.iterrows():
        enterprise_reputation[row['企业代号']] = row['信誉评级']

        enterprise_category[row['企业代号']] = -1
        for j, c in enumerate(categories):
            if c in row['企业名称']:
                enterprise_category[row['企业代号']] = j
                break

    df = preprocess(pd.read_csv('../data/cost_1.csv'), pd.read_csv('../data/income_1.csv'), enterprise_category)
    df['reputation'] = df['firm_code'].map(enterprise_reputation).map(reputation_map)

    data_X = df[feature_space]
    data_Y = df['reputation']
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

    # train
    model = CatBoostClassifier()
    model.fit(X_train, y_train)

    # eval
    y_pred = model.predict(X_test)
    print('A:', np.count_nonzero(y_pred[y_test == 0] == 0) / np.count_nonzero(y_test == 0))
    print('B:', np.count_nonzero(y_pred[y_test == 1] == 1) / np.count_nonzero(y_test == 1))
    print('C:', np.count_nonzero(y_pred[y_test == 2] == 2) / np.count_nonzero(y_test == 2))
    print('D:', np.count_nonzero(y_pred[y_test == 3] == 3) / np.count_nonzero(y_test == 3))
    print(accuracy_score(y_pred, y_test))

    # save
    model.save_model('model.cat')


if __name__ == '__main__':
    main()
