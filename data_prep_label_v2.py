import pandas as pd
import csv

def main():
    fname = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']

    for fn in fname:

        df = pd.read_csv('stock_label/' + fn + '_label_value.csv', encoding = 'utf8', low_memory=False)
        feature_list = list(df.columns.values)
        data = {}
        for feature in feature_list:
            data.update({feature:[]})

        for row in range(len(df)):
            for feature in feature_list:
                if row == 0:
                    data[feature].append(0)

                else:
                    data[feature].append((df[feature][row] - df[feature][row-1]) /df[feature][row-1])

        with open('stock_label/' + fn + '_label_ratio.csv', 'w', newline='', encoding='utf-8') as fout:
            wr = csv.writer(fout)
            wr.writerow(feature_list)

            for i in range(len(df)):
                value = []
                for feature in feature_list:
                    value.append(data[feature][i])

                wr.writerow(value)


if __name__ == '__main__':
    main()
