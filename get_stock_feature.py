import pandas as pd
import datetime
import csv

def load_target_num(stock_num):
    target_num = []
    df = pd.read_csv('stock_relation/' + (stock_num + '_comp.csv'))
    for i in range(len(df)):
        target_num.append(str(int(df['number'][i])))

    target_num.append(stock_num)
    return target_num


def main():
    df = pd.read_csv('TBrain_Round2_DataSet_20180518/tsharep.csv', encoding = 'utf8', low_memory=False)
    df_label = pd.read_csv('TBrain_Round2_DataSet_20180518/tetfp.csv', encoding = 'utf8', low_memory=False)
    each_feature = ['收盤價(元)', '最高價(元)', '最低價(元)', '成交張數(張)']
    #fname = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']
    fname = ['701']

    for fn in fname:
        date_title = []
        workday = pd.read_csv('stock_workday/' + fn + '_workday.csv')
        data_v1 = {}
        data_v2 = {}

        target_num = load_target_num(fn)
        feature_title = []
        for num in target_num:
            for fe in each_feature:
                feature_title.append(num + '_' + fe)

        for date in workday['date']:
            data_v1.update({str(date):{}})
        print('running all stock data')
        for row in range(len(df)):
            print(str(row) + '/' + str(len(df)))
            if str(df['日期'][row]) in data_v1.keys():
                data_v1[str(df['日期'][row])].update({str(df['代碼'][row]):[]})
                data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(float(str(df['收盤價(元)'][row]).replace(',','')))
                data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(float(str(df['最高價(元)'][row]).replace(',','')))
                data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(float(str(df['最低價(元)'][row]).replace(',','')))
                data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(int(df['成交張數(張)'][row].replace(',','')))

        print('running' + fn + 'stock data')

        for row in range(len(df_label)):
            if int(df_label['代碼'][row]) == int(fn):
                if str(df_label['日期'][row]) in data_v1.keys():
                    data_v1[str(df_label['日期'][row])].update({str(df_label['代碼'][row]):[]})
                    data_v1[str(df_label['日期'][row])][str(df_label['代碼'][row])].append(float(str(df_label['收盤價(元)'][row]).replace(',','')))
                    data_v1[str(df_label['日期'][row])][str(df_label['代碼'][row])].append(float(str(df_label['最高價(元)'][row]).replace(',','')))
                    data_v1[str(df_label['日期'][row])][str(df_label['代碼'][row])].append(float(str(df_label['最低價(元)'][row]).replace(',','')))
                    data_v1[str(df_label['日期'][row])][str(df_label['代碼'][row])].append(int(df_label['成交張數(張)'][row].replace(',','')))

        for date in workday['date']:
            data_v2.update({str(date):[]})
            for num in target_num:
                data_v2[str(date)].extend(data_v1[str(date)][num])

        with open('stock_feature/' + fn + '_feature.csv', 'w', newline='', encoding='utf-8') as fout:
            wr = csv.writer(fout)
            wr.writerow(feature_title)

            for date in workday['date']:
                value = []
                for fe in data_v2[str(date)]:
                    value.append(fe)

                wr.writerow(value)

        print('')


if __name__ == '__main__':
    main()
