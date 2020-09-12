import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import seaborn
from Invoice import Invoice
from Enterprise import Enterprise
from datetime import date
import pandas as pd

enterprise_info = pd.read_csv('../data/1_info.csv')
n_enterprise = enterprise_info.shape[0]

enterprise_dic = {}
for i in range(n_enterprise):
    number = enterprise_info['企业代号'][i]
    name = enterprise_info['企业名称'][i]
    credit_rating = enterprise_info['信誉评级'][i]
    break_contract_str = enterprise_info['是否违约'][i]
    break_contract = True if break_contract_str == '是' else False
    enterprise_object = Enterprise(number, name, credit_rating, break_contract)
    enterprise_dic[number] = enterprise_object

invoice_in = pd.read_csv('../data/1_in.csv')
n_in = invoice_in.shape[0]


def get_date(date_str):
    y, m, d = date_str.split('/')
    if len(m) != 2:
        m = '0' + m
    if len(d) != 2:
        d = '0' + d
    return date.fromisoformat(y + '-' + m + '-' + d)


for i in range(n_in):
    enterprise_name = invoice_in['企业代号'][i]
    number = invoice_in['发票号码'][i]
    date_str = invoice_in['开票日期'][i]
    date = get_date(date_str)
    self_enterprise = enterprise_dic[enterprise_name]
    partner = invoice_in['销方单位代号'][i]
    amount = invoice_in['金额'][i]
    tax = invoice_in['税额'][i]
    sum_money = invoice_in['价税合计'][i]
    state_available_str = invoice_in['发票状态'][i]
    state_available = True if state_available_str == '有效发票' else False
    invoice_object = Invoice(number, date, self_enterprise, partner, amount, tax, sum_money, state_available, True)
    self_enterprise.add_invoice(invoice_object)

invoice_out = pd.read_csv('../data/1_out.csv')
n_out = invoice_out.shape[0]


for i in range(n_out):
    enterprise_name = invoice_out['企业代号'][i]
    number = invoice_out['发票号码'][i]
    date_str = invoice_out['开票日期'][i]
    date = get_date(date_str)
    self_enterprise = enterprise_dic[enterprise_name]
    partner = invoice_out['购方单位代号'][i]
    amount = invoice_out['金额'][i]
    tax = invoice_out['税额'][i]
    sum_money = invoice_out['价税合计'][i]
    state_available_str = invoice_out['发票状态'][i]
    state_available = True if state_available_str == '有效发票' else False
    invoice_object = Invoice(number, date, self_enterprise, partner, amount, tax, sum_money, state_available, False)
    self_enterprise.add_invoice(invoice_object)


for enterprise in enterprise_dic.values():
    enterprise.invoice_list.sort(key=lambda x: x.date)


frame = pd.DataFrame(columns=['日期', '资金', '企业代号', '信誉评级'])
for enterprise in enterprise_dic.values():
    sum = 0
    current_date = enterprise.invoice_list[0].date
    temp = enterprise.invoice_list[0]
    for invoice in enterprise.invoice_list:
        temp = invoice
        if invoice.date != current_date:
            frame = frame.append(
                [{'日期': current_date, '资金': sum, '企业代号': enterprise.number, '信誉评级': enterprise.credit_rating}],
                ignore_index=True)
            current_date = invoice.date
        if invoice.buy_in:
            sum = sum - invoice.sum_money
        else:
            sum = sum + invoice.amount
    frame = frame.append([{'日期': temp.date, '资金': sum, '企业代号': enterprise.number, '信誉评级': enterprise.credit_rating}],
                         ignore_index=True)


print(frame)
frame.to_csv("./frame.csv")
