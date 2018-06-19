import pandas as pd
import csv
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn.model_selection import cross_val_predict


def load_stock_ud(day, ud_day):
    ud_df = pd.read_csv('../ensemble_pred_result_3/result_' + day + '.csv')
    ud_data_df = []
    ud_data = []
    for row in range(len(ud_df)):
        ud_data_df.append(ud_df[ud_day][row])

    for ud_item in ud_data_df:
        if ud_item == 1:
            ud_data.extend([1, 0, 0])

        elif ud_item == -1:
            ud_data.extend([0, 1, 0])

        else:
            ud_data.extend([0, 0, 1])
    return ud_data


def load_day_ud(day):
    df = pd.read_csv('day_ud_label.csv')
    df_date = list(df['date'])
    label = []
    day_index = df_date.index(int(day))

    for day_cnt in range(day_index, day_index+5, 1):
        label.append(df['ud'][day_cnt])

    return label

def main():
    stk_name = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']
    workday_pred = ['20180110', '20180111', '20180112', '20180115', '20180116', '20180117', '20180118', '20180119', '20180122', '20180123', '20180124', '20180125', '20180126', '20180129', '20180130', '20180131', '20180201', '20180202', '20180205', '20180206', '20180207', '20180208', '20180209', '20180212', '20180221', '20180222', '20180223', '20180226', '20180227', '20180301', '20180302', '20180305', '20180306', '20180307', '20180308', '20180309', '20180312', '20180313', '20180314', '20180315', '20180316', '20180319', '20180320', '20180321', '20180322', '20180323', '20180326', '20180327', '20180328', '20180329', '20180330', '20180331', '20180402', '20180403', '20180409', '20180410', '20180411', '20180412', '20180413', '20180416', '20180417', '20180418', '20180419', '20180420', '20180423', '20180424', '20180425', '20180426', '20180427', '20180430', '20180502', '20180503', '20180504', '20180507', '20180508', '20180509', '20180510', '20180511', '20180514', '20180515', '20180516', '20180517', '20180518', '20180521', '20180522', '20180523', '20180524', '20180525', '20180528', '20180529', '20180530', '20180531', '20180601', '20180604', '20180605', '20180606', '20180607', '20180608']
    workday = ['20180110', '20180111', '20180112', '20180115', '20180116', '20180117', '20180118', '20180119', '20180122', '20180123', '20180124', '20180125', '20180126', '20180129', '20180130', '20180131', '20180201', '20180202', '20180205', '20180206', '20180207', '20180208', '20180209', '20180212', '20180221', '20180222', '20180223', '20180226', '20180227', '20180301', '20180302', '20180305', '20180306', '20180307', '20180308', '20180309', '20180312', '20180313', '20180314', '20180315', '20180316', '20180319', '20180320', '20180321', '20180322', '20180323', '20180326', '20180327', '20180328', '20180329', '20180330', '20180331', '20180402', '20180403', '20180409', '20180410', '20180411', '20180412', '20180413', '20180416', '20180417', '20180418', '20180419', '20180420', '20180423', '20180424', '20180425', '20180426', '20180427', '20180430', '20180502', '20180503', '20180504', '20180507', '20180508', '20180509', '20180510', '20180511', '20180514', '20180515', '20180516', '20180517', '20180518', '20180521', '20180522', '20180523', '20180524', '20180525', '20180528', '20180529', '20180530', '20180531', '20180601']

    day_ud_list = ['Mon_ud', 'Tue_ud', 'Wed_ud', 'Thu_ud', 'Fri_ud']
    stk_ud = {}
    score = []
    feature = []
    label = []

    for day in workday:
        stk_ud.update({day:{}})
        for ud_day in day_ud_list:
            stk_ud[day].update({ud_day: load_stock_ud(day, ud_day)})
            feature.append(load_stock_ud(day, ud_day))
        label.extend(load_day_ud(day))

    feature = np.asarray(feature)
    label = np.asarray(label)

    clf = RandomForestClassifier()
    clf.fit(feature, label)
    score.append(cross_val_predict(clf, feature, label, cv=5))
    print(score)
    input()
    for date in workday_pred:
        print(date)
        for day_ud in day_ud_list:
            result_df = pd.read_csv('../ensemble_pred_result_3/result_' + date + '.csv')
            tr = []
            for stk_num in range(len(list(result_df[day_ud]))):
                if result_df[day_ud][stk_num] == 1:
                    tr.extend([1, 0, 0])
                elif result_df[day_ud][stk_num] == -1:
                    tr.extend([0, 1, 0])
                else:
                    tr.extend([0, 0, 1])

            tr = np.asarray(tr).reshape(1, -1)
            pred_result = clf.predict(tr)
            pred_result = pred_result[0]
            print(pred_result)

            with open('../ensemble_pred_result_3/result_' + date + '.csv', 'w', newline='') as fout:
                wr = csv.writer(fout)
                title = list(result_df.columns.values)
                wr.writerow(title)

                for row in range(len(result_df)):
                    value = []
                    for item in title:
                        if item == day_ud:
                            value.append(pred_result)

                        else:
                            value.append(result_df[item][row])
                    wr.writerow(value)

if __name__ == '__main__':
    main()
