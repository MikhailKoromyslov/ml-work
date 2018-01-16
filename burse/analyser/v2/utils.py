import os
import sys
import fileinput
import glob
import pandas as pd
import numpy as np
import glob
import ntpath
import shutil
import re
import math
import datetime as dt
import time

to_fit_by_date = ['comex.GC', 'EURRUB', 'ICE.BRN', 'USDRUB']


def replace_in_path(dir_path, from_val, to_val):
    file_paths = [file_path for file_path in glob.glob(dir_path + '*.csv')]
    for file_path in file_paths:
        replace_in_file(file_path, from_val, to_val)


def replace_in_file(file_path, from_val, to_val):
    temp_file = open(file_path, 'r+')
    for line in fileinput.input(file_path):
        new_line = line.replace(from_val, to_val)
        temp_file.write(new_line)
    temp_file.close()


def replace_in_file_using_df(file_path, to_replace, value):
    df = pd.read_csv(file_path)
    if df.shape[1] == 1:
        df = pd.read_csv(file_path, sep=';')
    df = df.replace(to_replace=to_replace, value=value)
    df.to_csv(file_path, index=False, encoding='utf-8')


def replace_in_path_old(from_path, repl_pairs):
    file_paths = [file_path for file_path in glob.glob(from_path + '*.csv')]
    for file_path in file_paths:
        tempFile = open(file_path, 'r+')

        for line in fileinput.input(file_path):
            new_line = line
            for from_val, to_val in repl_pairs:
                new_line = new_line.replace(from_val, to_val)
            tempFile.write(new_line)
        tempFile.close()


def set_standard_header_row(path, header_row):
    file_paths = [file_path for file_path in glob.glob(path + '*.csv')]
    for file_path in file_paths:
        with open(file_path) as f:
            lines = f.readlines()

        lines[0] = header_row + '\n'

        with open(file_path, "w") as f:
            f.writelines(lines)


def get_filename_to_path_dict_for(directory):
    file_paths_dict = {ntpath.basename(file_path).replace('.csv', ''): file_path
                       for file_path in glob.glob(directory + '*.csv')}
    return file_paths_dict


def get_ticker_to_path_dict_for(directory):
    file_paths_dict = {re.sub('([_\d])*.csv', '', ntpath.basename(file_path)): file_path
                       for file_path in glob.glob(directory + '*.csv')}
    return file_paths_dict


def tickers_unique(path):
    filename_to_path_dict_for = get_filename_to_path_dict_for(path)
    ticker_to_path_dict = get_ticker_to_path_dict_for(path)
    return len(filename_to_path_dict_for) == len(ticker_to_path_dict)


def union_by_tickers(from_path, to_path):
    """
    Объединим датафреймы с одним тикером за разные годы
    """
    df_dict = get_data_frames_dict(from_path)

    tickers = {}
    for key in df_dict.keys():
        ticker = key.split('_')[0]
        tickers[ticker] = tickers[ticker] if ticker in tickers else []
        tickers[ticker].append(key)

    for ticker, key_frases in tickers.items():
        union_dfs_of_same_ticker(key_frases, df_dict) \
            .drop_duplicates() \
            .to_csv(to_path + ticker + '.csv', index=False, encoding='utf-8')


def union_dfs_of_same_ticker(keys, df_dict):
    keys = sorted(keys)
    temp_df = df_dict[keys[0]]
    for key in keys[1:]:
        temp_df = pd.concat([temp_df, df_dict[key]])
    return temp_df


def check_paths(paths):
    for path in paths:
        if path:
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
    return paths


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    # if (type(weights) is pd.Series) | (type(weights) is np.ndarray):
    #     weights = weights.astype(np.float64).round().astype(np.int64)
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
    return average, math.sqrt(variance)


def get_data_frames_dict(from_path) -> dict:
    return {ntpath.basename(file_path): pd.read_csv(file_path)
            for file_path in glob.glob(from_path + '*.csv')}


def patch_files(patch_from: 'Путь к директории с новыми данными',
                patch_to: 'Путь к директории с уже имевшимися данными'):
    patch_from_paths = get_ticker_to_path_dict_for(patch_from)
    patch_to_paths = get_ticker_to_path_dict_for(patch_to)

    for key in patch_from_paths.keys():
        patch_from_csv, patch_to_csv = patch_from_paths[key], patch_to_paths[key]
        df_from, df_to = pd.read_csv(patch_from_csv), pd.read_csv(patch_to_csv)
        pd.concat([df_to, df_from]) \
            .drop_duplicates() \
            .to_csv(patch_to_csv, index=False, encoding='utf-8')


def clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def convert_intervals_to_day(df):
    new_df = pd.DataFrame(index=['000000', '003000', '010000', '013000', '020000', '023000', '030000', '033000',
                                   '040000', '043000', '050000', '053000', '060000', '063000', '070000', '073000',
                                   '080000', '083000', '090000', '093000', '100000', '103000', '110000', '113000',
                                   '120000', '123000', '130000', '133000', '140000', '143000', '150000', '153000',
                                   '160000', '163000', '170000', '173000', '180000', '183000', '190000', '193000',
                                   '200000', '203000', '210000', '213000', '220000', '223000', '230000', '233000'],
                          columns=['open', 'high', 'low', 'close', 'vol'])
    for i, item in df.iterrows():
        print(item, i)
        break
def subtract_one_month(t):
    """Return a `datetime.date` or `datetime.datetime` (as given) that is
    one month later.

    Note that the resultant day of the month might change if the following
    month has fewer days:

        subtract_one_month(datetime.date(2010, 3, 31))
        datetime.date(2010, 2, 28)
    """
    import datetime
    one_day = datetime.timedelta(days=1)
    one_month_earlier = t - one_day
    while one_month_earlier.month == t.month or one_month_earlier.day > t.day:
        one_month_earlier -= one_day
    return one_month_earlier


import pandas as dp
import numpy as np
df=pd.read_csv('D:/TEMP/data/united/ALRS.csv')
convert_intervals_to_day(df)


# replace_in_path_old(from_path='D:/TEMP/datasets/test/',
#                     repl_pairs=[(';', ','),
#                                 (' ', ''),
#                                 ('"', '')])
# # set_commas(from_path = 'D:/TEMP/data/temp/')
# # replace_diamonds()
# # patch_files()



# convert_intervals_to_day(df)