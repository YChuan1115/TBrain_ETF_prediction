import pandas as pd
import datetime
import csv

def load_target_num(stock_num):
    target_num = []
    df = pd.read_csv('stock_relation/' + (stock_num + '_comp.csv'))
    for i in range(len(df)):
        target_num.append(str(int(df['number'][i])))

    return target_num


def main():
    df = pd.read_csv('TBrain_Round2_DataSet_20180518/tsharep.csv', encoding = 'utf8')
    workday = pd.read_csv('2018_workday.csv', encoding = 'utf8')
    stock_num = pd.read_csv('stock_number.csv', encoding = 'utf8')
    each_feature = ['收盤價(元)', '成交張數(張)']
    date_title = []
    data_v1 = {}
    data_v2 = {}

    fname = '50'
    target_num = load_target_num(fname)
    feature_title = []
    for num in target_num:
        for fe in each_feature:
            feature_title.append(num + '_' + fe)

    for date in workday['date']:
        data_v1.update({str(date):{}})
    #data['日期']['代碼'][30.41, 30.53, 30.18, 30.45, 6374]
    input()
    for row in range(len(df)):
        print(str(row) + '/' + str(len(df)))
        if str(df['日期'][row]) in data_v1.keys():
            data_v1[str(df['日期'][row])].update({str(df['代碼'][row]):[]})
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(float(str(df['收盤價(元)'][row]).replace(',','')))
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(int(df['成交張數(張)'][row].replace(',','')))



    for date in workday['date']:
        data_v2.update({str(date):[]})
        for num in target_num:
            data_v2[str(date)].extend(data_v1[str(date)][num])

    with open(fname + '_feature.csv', 'w', newline='', encoding='utf-8') as fout:
        wr = csv.writer(fout)
        wr.writerow(feature_title)

        for date in workday['date']:
            value = []
            for fe in data_v2[str(date)]:
                value.append(fe)

            wr.writerow(value)


if __name__ == '__main__':
    main()
