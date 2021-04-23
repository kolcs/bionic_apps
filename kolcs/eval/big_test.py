from pathlib import Path

import pandas as pd

path = Path(r'D:\Users\Csabi\Desktop\param invest').joinpath('big_test')
db_names = ['physionet', 'ttk']
setups = [
    'MicroVolt, PSD, MinMax0-1 featrure, l2norm',
    'MicroVolt, PSD, MinMax0-1 featrure, MinMax0-1+standardScale',
    'MicroVolt, PSD, MinMax0-1 featrure, standardScale',
    'MicroVolt, PSD, MinMax-1-1 featrure, standardScale',
]
feature_list = ['alpha', 'beta', 'range30', 'multi', 'range40']


def open_one_file(filename):
    df = pd.read_csv(str(filename), sep=';')
    acc = df[['Subject', 'Avg. Acc']]
    col_name = filename.stem.split('-')[0]
    acc = acc.rename(columns={'Avg. Acc': col_name})
    return acc


def merge_big_test_res(files):
    flist = files.copy()
    data = open_one_file(flist.pop(0))
    for file in flist:
        f = open_one_file(file)
        data = pd.merge(data, f, how="left", on="Subject")
    return data


def convert_data(tb, setup):
    avg_acc = {i: tb[col].mean() for i, col in enumerate(tb.columns)}
    avg_acc[0] = setup
    avg_acc = pd.DataFrame(avg_acc, index=[0])
    #     print(avg_acc)
    tb = tb.T.reset_index().T.reset_index(drop=True)
    tb = pd.concat([avg_acc, tb]).reset_index(drop=True)
    return tb, avg_acc


def concat_res(df_list):
    mrgd = df_list.pop(0)
    for df in df_list:
        mrgd = pd.concat([mrgd, df]).reset_index(drop=True)
    return mrgd


def main():
    for db in db_names:
        out_file = str(path.joinpath(db, 'out.xlsx'))
        with pd.ExcelWriter(out_file) as writer:
            avg_acc_list = []
            for i, setup in enumerate(setups):
                print(db, setup)
                res = list()
                for f in feature_list:
                    res.extend(path.joinpath(db, setup).glob(f + '*.csv'))
                tb = merge_big_test_res(res)
                tb, avg_acc = convert_data(tb, setup)
                avg_acc_list.append(avg_acc)

                tb.to_excel(writer, f'sheet{i + 1}', header=False, index=False)
                print('done...')
            merged = concat_res(avg_acc_list)
            merged = merged.rename(columns={i + 1: feature for i, feature in enumerate(feature_list)})
            merged = merged.rename(columns={0: 'type'})
            merged.to_excel(writer, 'results', index=False)


if __name__ == '__main__':
    main()
