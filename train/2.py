import pandas as pd
import json
import numpy as np
from train.catboost_train import invoice_status, preprocess, categories, feature_space
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression

enterprise_info = pd.read_csv('../data/2_info.csv')
enterprise_category = {}
for i, row in enterprise_info.iterrows():
    enterprise_category[row['企业代号']] = -1
    for j, c in enumerate(categories):
        if c in row['企业名称']:
            enterprise_category[row['企业代号']] = j
            break


def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b


def get_feats(df_cost: pd.DataFrame, df_rev: pd.DataFrame, reputation):
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

    groups = df.groupby('企业代号')

    def get_feats(x: pd.DataFrame):
        firm_code = x['企业代号'].tolist()[0]

        income = x[(x['type'] == 1) & (x['status'] == 0)]
        income_invalid = x[(x['type'] == 1) & (x['status'] == 1)]

        cost = x[(x['type'] == 0) & (x['status'] == 0)]
        cost_invalid = x[(x['type'] == 0) & (x['status'] == 1)]

        income_dt = (income['date'].max() - income['date'].min()).days
        cost_dt = (cost['date'].max() - cost['date'].min()).days

        return pd.Series({
            'firm_code': firm_code,

            'income': income['金额'].sum() / income_dt,
            'cost': cost['金额'].sum() / cost_dt,
            'income_invalid': income_invalid['金额'].sum() / income_dt,
            'cost_invalid': cost_invalid['金额'].sum() / cost_dt,

            'income_std': income['金额'].std(),
            'cost_std': cost['金额'].std(),

            'reputation': reputation[firm_code][0],
        })

    feats = groups.apply(get_feats).reset_index()
    feats.fillna(0, inplace=True)
    return feats


def L(dM, dN, w_m, m_std, n_std, reputation, coef: np.ndarray):
    x = np.asarray([dM, m_std, dN, 5 - reputation]).T
    ret = x * coef[np.newaxis, :]
    return np.sum(ret, axis=1)


def Lp(L, dM, dN, coef: np.ndarray):
    x = np.vstack([L, dM - dN]).T
    ret = x * coef[np.newaxis, :]
    return np.sum(ret, axis=1)


def calc_profit(dM, dN, w_m, m_std, n_std, reputation, coef):
    l = L(dM, dN, w_m, m_std, n_std, reputation, coef[:4])
    lp = Lp(l, dM, dN, np.asarray(coef[5:]))
    prof = l * lp
    return prof, l, lp


def cons_cost(x, a, b):
    ret = 0
    if x < a:
        ret += a - x
    if x > b:
        ret += x - b
    return ret / (b - a)


def train():
    """
    df1 = pd.read_csv('../data/2_in.csv')
    df2 = pd.read_csv('../data/2_out.csv')

    print('predicting reputation...')
    model = CatBoostClassifier()
    model.load_model('model.cat')
    df = preprocess(df1, df2, enterprise_category)
    pred = model.predict(df[feature_space])
    pred = pred.astype(np.int)
    print('predicted reputation...')

    reputations = dict(zip(df['firm_code'].tolist(), pred.tolist()))
    json.dump(reputations, open('pred.json', 'w'))
    """

    reputations = json.load(open('pred.json', 'r'))
    df1 = pd.read_csv('../data/2_in.csv')
    df2 = pd.read_csv('../data/2_out.csv')
    df = get_feats(df1, df2, reputations)

    firm_code = df['firm_code']
    dM = df['income'].to_numpy()
    dN = df['cost'].to_numpy()
    w_m = df['income_invalid'].to_numpy()
    m_std = df['income_std'].to_numpy()
    n_std = df['cost_std'].to_numpy()
    reputation = df['reputation'].to_numpy()
    print(df)

    def obj(coef):
        prof, l, lp = calc_profit(dM, dN, w_m, m_std, n_std, reputation, coef)
        return (
                1 / np.sum(prof) +
                np.sum(np.vectorize(lambda x: cons_cost(x, 100000, 1000000))(l)) +
                np.sum(np.vectorize(lambda x: cons_cost(x, 0.04, 0.15))(lp)) +
                cons_cost(np.sum(l), 0, 100000000)
        )

    def constraint(coef):
        prof, l, lp = calc_profit(dM, dN, w_m, m_std, n_std, reputation, coef)
        cons = np.asarray([
            np.all(l >= 100000),
            np.all(l <= 1000000),
            np.all(lp >= 0.04),
            np.all(lp <= 0.15),
            np.sum(l) <= 100000000,
        ])
        return cons.astype(np.int)

    from scipy.optimize import minimize

    print('start minimizing')
    n_params = 6
    res = minimize(
        obj,
        np.full(n_params, 0.2),
        method='SLSQP',
        options={'disp': True, 'maxiter': 100000},
        # constraints={
        #     'type': 'ineq',
        #     'fun': lambda x: constraint(x),
        # }
    )
    coef = res.x
    print(coef)
    profit_pred, l, lp = calc_profit(dM, dN, w_m, m_std, n_std, reputation, coef)
    profit_pred = pd.DataFrame({'firm_code': firm_code, 'profit_pred': profit_pred, 'L': l, 'Lp': lp})
    profit_pred['reputation'] = profit_pred['firm_code'].map(lambda x: reputations[x][0])
    profit_pred = profit_pred[profit_pred['reputation'] != 3]
    print(profit_pred['L'].sum())
    profit_pred.to_csv('profit_pred.csv', index=False)


if __name__ == '__main__':
    train()
