import pandas as pd
import csv

def main():
    df = pd.read_csv('TBrain_Round2_DataSet_20180518/tetfp.csv', encoding = 'utf8', low_memory=False)
    df = df.dropna(axis=1, how='all')
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    fname = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']




    for fn in fname:
        data_v1 = {}
        feature_list = []
        feature_list.append(fn + '_收盤價(元)')
        workday = pd.read_csv('stock_workday/' + fn + '_workday.csv')

        for date in workday['date']:
            data_v1.update({str(date):[]})

        for row in range(len(df)):
            if int(df['代碼'][row]) == int(fn):
                data_v1[str(df['日期'][row])].append(float(str(df['收盤價(元)'][row]).replace(',','')))
        # data_v1['日期']['代碼'] return ['收盤價']


        with open('stock_label/' + fn + '_label_value.csv', 'w', newline='', encoding='utf-8') as fout:
            wr = csv.writer(fout)
            wr.writerow(feature_list)

            for date in workday['date']:
                value = []
                value.append(data_v1[str(date)][0])

                wr.writerow(value)

if __name__ == '__main__':
    main()
