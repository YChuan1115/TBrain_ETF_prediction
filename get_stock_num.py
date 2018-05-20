import pandas as pd
import datetime
import csv

def main():
    df = pd.read_csv('TBrain_Round2_DataSet_20180518/tsharep.csv', encoding = 'utf8')
    workday = pd.read_csv('TBrain_Round2_DataSet_20180518/2018_workday.csv', encoding = 'utf8')
    target_num = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '703']
    date_title = []
    stock_num = {}
    data_v1 = {}
    data_v2 = {}


    #data['日期']['代碼'][30.41, 30.53, 30.18, 30.45, 6374]

    for row in range(len(df)):
        print(str(row) + '/' + str(len(df)))
        if str(df['代碼'][row]) in data_v1.keys():
            pass

        else:
            data_v1.update({str(df['代碼'][row]):df['中文簡稱'][row].replace(' ','')})

    with open('stock_number.csv', 'w', newline='') as fout:
        wr = csv.writer(fout)
        title = ['number', 'name']
        wr.writerow(title)

        for num in data_v1.keys():
            value = []
            value.append(num)
            value.append(data_v1[num])
            wr.writerow(value)

if __name__ == '__main__':
    main()
