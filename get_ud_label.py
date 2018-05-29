import pandas as pd
import numpy as np
import csv

def main():
    fname = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']

    for fn in fname:
        df = pd.read_csv('../stock_label/' + fn + '_label_ratio.csv')
        sign_list = []
        
        for row in range(len(df)):
            sign = 0
            sign_list.append(int(np.sign([df[str(fn+'_收盤價(元)')][row]])))

        with open('../stock_label/' + fn + '_label_ud.csv', 'w', newline='') as fout:
            wr = csv.writer(fout)
            title = [str(fn+'_收盤價(元)')]
            wr.writerow(title)

            for row in range(len(sign_list)):
                value = []
                value.append(sign_list[row])
                wr.writerow(value)


if __name__ == '__main__':
    main()
