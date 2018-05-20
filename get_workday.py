import pandas as pd
import datetime
import csv

def main():
    df = pd.read_csv('TBrain_Round2_DataSet_20180518/tetfp.csv', encoding = 'utf8')
    target_num = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '703']
    date_title = []
    data_v1 = {}


    #data['日期']['代碼'][30.41, 30.53, 30.18, 30.45, 6374]

    for row in range(len(df)):
        print(str(row) + '/' + str(len(df)))
        if df['日期'][row] in data_v1.keys():
            data_v1[str(df['日期'][row])].update({str(df['代碼'][row]):[]})
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['開盤價(元)'][row])
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['最高價(元)'][row])
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['最低價(元)'][row])
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['收盤價(元)'][row])
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['成交張數(張)'][row])

        else:
            data_v1.update({str(df['日期'][row]):{str(df['代碼'][row]):[]}})
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['開盤價(元)'][row])
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['最高價(元)'][row])
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['最低價(元)'][row])
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['收盤價(元)'][row])
            data_v1[str(df['日期'][row])][str(df['代碼'][row])].append(df['成交張數(張)'][row])

    for date in data_v1.keys():
        date_title.append(int(date))

    with open('2018_date.csv', 'w', newline='') as fout:
        wr = csv.writer(fout)
        title = ['date']
        wr.writerow(title)

        for row in date_title:
            value = []
            value.append(row)
            wr.writerow(value)

if __name__ == '__main__':
    main()
