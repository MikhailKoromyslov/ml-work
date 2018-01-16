# import keras
# import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# def plot_fig(col_name1, data1, col_name2, data2):
#     plt.figure()
#     plt.plot(data1, data2, 'r', linewidth=2)
#     plt.legend([col_name1, col_name2])
#     plt.savefig('imgs/temp.png')
#
#
# df = pd.DataFrame(
#         {'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
#          'x2': [11, 22, 33, 44, 55, 66, 77, 88, 99]
#          })
#
# plot_fig('x1', df['x1'], 'x2', df['x2'])
# # result = pd.concat([temp_df, temp_df_1], axis=1, join='inner')


def test():
    y_col_name = "radiant_win"

    # df = pd.read_csv('final_df_1.csv')
    df = pd.read_csv('dota/data_frames/train-4.csv')
    # model = KMeans(n_clusters=3, init='k-means++', random_state=241)
    # df['cluster'] = model.fit_predict(df)

    # plot_fig_2d(df, y_col_name, 'r_sum_win_prob', 'r_sum_gold///divide///d_sum_gold')
    col0 = 'd_sum_win_prob'
    col1 = 'r_sum_win_prob'
    col2 = 'r_sum_gold'
    col3 = 'd_sum_gold'

    temp_df = df[[col0, col1, col2, col3, y_col_name]]
    temp_df_0 = temp_df.ix[temp_df[y_col_name] == 0][[col0, col1, col2, col3]]
    temp_df_1 = temp_df.ix[temp_df[y_col_name] == 1][[col0, col1, col2, col3]]
    plt.figure()
    plt.plot((temp_df_0[col1] - temp_df_0[col0]), (temp_df_0[col2] - temp_df_0[col3]), 'b.')
    plt.plot((temp_df_1[col1] - temp_df_1[col0]), (temp_df_1[col2] - temp_df_1[col3]), 'r.')
    plt.savefig('imgs/temp.png')
    plt.figure()
    plt.plot((temp_df_1[col1] - temp_df_1[col0]), (temp_df_1[col2] - temp_df_1[col3]), 'r.')
    plt.plot((temp_df_0[col1] - temp_df_0[col0]), (temp_df_0[col2] - temp_df_0[col3]), 'b.')
    plt.savefig('imgs/temp1.png')


d = {'a': 1, 'b': 2}
print(d)
d1 = d
d1['a'] = 11
print(d)

