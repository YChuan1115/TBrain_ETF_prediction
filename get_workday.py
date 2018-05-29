import pandas as pd
import datetime
import csv

def main():
    df = pd.read_csv('../TBrain_Round2_DataSet_20180518/tetfp.csv', encoding = 'utf8')
    fname = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']
    date_title = []
    data_v1 = {}


    #data['日期']['代碼'][30.41, 30.53, 30.18, 30.45, 6374]
    for fe in fname:
        date_title = []
        for row in range(len(df)):
            print(str(row) + '/' + str(len(df)))
            if int(df['代碼'][row]) == int(fe):
                if df['日期'][row] in date_title:
                    pass

                else:
                    date_title.append(df['日期'][row])

        with open('../stock_workday/' + fe + '_workday.csv', 'w', newline='') as fout:
            wr = csv.writer(fout)
            title = ['date']
            wr.writerow(title)

            for date in date_title:
                value = []
                value.append(date)
                wr.writerow(value)

if __name__ == '__main__':
    main()
