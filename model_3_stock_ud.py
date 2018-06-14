import pandas as pd
import csv
import numpy as np

def load_stock_ud(fname, workday):
    ud_df = pd.read_csv('../stock_label/' + fname + '_label_ud.csv')
    ud_df = list(ud_df[fname + '_收盤價(元)'])
    workday_df = pd.read_csv('../stock_workday/' + fname + '_workday.csv')
    workday_df = list(workday_df['date'])
    ud_data = []
    for day in workday:
        row_index = workday_df.index(int(day))
        ud_data.append(ud_df[row_index])

    return ud_data

def main():
    stk_name = ['50', '51', '53', '54', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']
    workday = ['20180110', '20180111', '20180112', '20180115', '20180116', '20180117', '20180118', '20180119', '20180122', '20180123', '20180124', '20180125', '20180126', '20180129', '20180130', '20180131', '20180201', '20180202', '20180205', '20180206', '20180207', '20180208', '20180209', '20180212', '20180221', '20180222', '20180223', '20180226', '20180227', '20180301', '20180302', '20180305', '20180306', '20180307', '20180308', '20180309', '20180312', '20180313', '20180314', '20180315', '20180316', '20180319', '20180320', '20180321', '20180322', '20180323', '20180326', '20180327', '20180328', '20180329', '20180330', '20180331', '20180402', '20180403', '20180409', '20180410', '20180411', '20180412', '20180413', '20180416', '20180417', '20180418', '20180419', '20180420', '20180423', '20180424', '20180425', '20180426', '20180427', '20180430', '20180502', '20180503', '20180504', '20180507', '20180508', '20180509', '20180510', '20180511', '20180514', '20180515', '20180516', '20180517', '20180518', '20180521', '20180522', '20180523', '20180524', '20180525', '20180528', '20180529', '20180530', '20180531', '20180601', '20180604', '20180605', '20180606', '20180607', '20180608']
    output_data = []
    stk_ud = {}
    for stk in stk_name:
        stk_ud.update({stk: load_stock_ud(stk, workday)})

    for day in range(len(workday)):
        up_cnt = 0
        down_cnt = 0
        for stk in stk_name:
            if stk_ud[stk][day] == 1:
                up_cnt += 1

            elif stk_ud[stk][day] == -1:
                down_cnt += 1

        if max(up_cnt, down_cnt) == up_cnt:
            output_data.append(1)

        else:
            output_data.append(-1)

    with open('../model_3/day_ud_label.csv', 'w', newline='') as fout:
        wr = csv.writer(fout)
        title = ['date', 'ud']
        wr.writerow(title)

        for row in range(len(output_data)):
            value = []
            value.append(int(workday[row]))
            value.append(output_data[row])
            wr.writerow(value)


if __name__ == '__main__':
    main()
