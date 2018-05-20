from os import listdir
from os.path import isfile, isdir, join
import pandas as pd
import csv
import math

def load_stock_num():
    dict_name = {}
    dict_num = {}
    stock_num = pd.read_csv('stock_number.csv')
    for i in range(len(stock_num)):
        dict_name.update({stock_num['name'][i]:str(stock_num['number'][i])})
        dict_num.update({str(stock_num['number'][i]):stock_num['name'][i]})

    return dict_name, dict_num

def main():
    mypath = "/Users/105522013/Desktop/DART/TBrain_ETF/stock_relation"
    fname = listdir(mypath)
    stock_num = {}
    stock_name = {}
    stock_name, stock_num = load_stock_num()

    for name in fname:
        df = pd.read_csv('stock_relation/' + name)

        for i in range(len(df)):
            if math.isnan(df['number'][i]):
                try:
                    df['number'][i] = stock_name[df['name'][i]]
                except:
                    df['number'][i] = ''
            else:
                try:
                    df['name'][i] = stock_num[str(int(df['number'][i]))]
                except:
                    df['name'][i] = ''
        with open(name[:-4] + '_comp.csv', 'w', newline='', encoding='utf-8') as fout:
            wr = csv.writer(fout)
            title = ['number', 'name']
            wr.writerow(title)

            for i in range(len(df)):
                value = []
                value.append(df['number'][i])
                value.append(df['name'][i])

                wr.writerow(value)

if __name__ == '__main__':
    main()
