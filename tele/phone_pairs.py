"""
Подразумевается, что формат данных, которые использовались в анализе, будет тем же и в дальнейшем.
-------------------------------------------------------------------------------------------------
Инструкция:
1) Файл 01_Facts.xlsx должен быть сохранён как MS-DOS CSV. Затем нужно решить вопрос с правами его чтения.
    После этого полный путь к нему должен быть указан в функции main_function() в facts_path.

2) В main_function() должны быть описаны заданные пути в соответствии с комментарием:
    # Путь на одноимённый файл. Содержимое этого файла ДОЛЖНО быть уже отсортировано по колонке tstamp!!!
    primary_path = 'D:/TEMP/datasets1/test/02_Data_test.csv'
    # Путь, в котором модифицированны исходный файл должен быть сохранён
    modified_path = 'D:/TEMP/datasets1/test/02_Data_test_modified.csv'
    # Путь на .csv файл, сохранённый из исходного 01_Facts.xlsx
    facts_path = 'D:/TEMP/datasets1/facts/pairs.csv'
    # Путь, в котором должны будут сохраниться подготовленные для обучения алгоритма данные
    train_path = 'D:/TEMP/datasets1/test/train.csv'
    # Путь для сохранения всех прогнозов
    results_path = 'D:/TEMP/datasets1/test/results.csv'
    # Путь для сохранения всех найденных пар
    only_pairs_path = 'D:/TEMP/datasets1/test/results_only_pairs.csv'

3) По путям primary_path и facts_path должны находиться соответствующие исходные файлы
4) По пути only_pairs_path по завершении работы алгоритма будет лежать файл с парами номеров,
    которые по мнению алгоритма должны принадлежать одному владельцу. Алгоритм предсказывает весь массив данных,
    может ошибаться, поэтому в этом файле наверняка будут не все пары, указанные в 01_Facts.xlsx
--------------------------------------------------------------------------------------------------
Коротко о механизме работы:
    Составляются все возможные пары номеров (потоково), на основании всех исходных данных формируются отношения
    между парами - Расстояния и временной лаг.
    За это отвечает функция prepare_data(x_df, y_df), обрабатывающая исходные данные по одной паре номеров.
    Эта функция - точка расширения, в которой можно задавать генерацию и других данных из исходных,
    чтобы повысить качество алгоритма.
    Для прогноза алгоритм использует логистическую регрессию, обученную по всем известным парам и оптимальному числу
    строк, представляющих НЕ пары. Это оптимальное значение находится итеративным подбором.
    Структура данных после предобработки и перед входом в анализ: одна строка - характеристика одной пары номеров.

"""

import pandas as pd
import numpy as np
import fileinput
import os
import math
import glob
import ntpath
import shutil

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import sin, cos, sqrt, atan2, radians
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score as acc, precision_score, recall_score, f1_score
from sklearn.preprocessing import scale
import time

new_df_columns = ['code', 'answer', 'mean_dist', 'mean_time_diff', 'num_dist_less_10_km', 'num_dist_less_5_km',
                  'num_dist_less_1_km', 'num_dist_less_05_km']

new_test_df_columns = ['code', 'mean_dist', 'mean_time_diff', 'num_dist_less_10_km', 'num_dist_less_5_km',
                       'num_dist_less_1_km', 'num_dist_less_05_km']


def replace_in_file_using_df(file_path, to_replace, value):
    df = pd.read_csv(file_path)
    if df.shape[1] == 1:
        df = pd.read_csv(file_path, sep=';')
    df = df.replace(to_replace=to_replace, value=value)
    df.to_csv(file_path, index=False, encoding='utf-8')


def modify(primary_path, modified_path):
    """
    Должно быть выполнено до того, как начинать работать с данными.
    """
    df = pd.read_csv(primary_path, sep=';')
    df = df.replace(to_replace='null', value='')
    df = convert_df_to_user_positions(df)
    df.to_csv(modified_path, index=False, encoding='utf-8')


# replace_in_file_using_df(file_path='D:/TEMP/datasets/test/02_Data_test.csv',
#                          to_replace='null', value='')
# replace_in_file_using_df(file_path='D:/TEMP/datasets/facts/pairs.csv',
#                          to_replace='null', value='')


def plot_pair_movements(x, x1, y, y1, z, z1):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='parametric curve')
    ax.plot(x1, y1, z1, label='parametric curve')
    ax.legend()
    plt.show()


def plot_pair_movements_2d(x, x1, y, y1):
    plt.figure()
    plt.plot(x, y, 'b')
    plt.plot(x1, y1, 'r')
    plt.show()


def owner_is_the_same(df_temp):
    """
    Мы считаем, что если для одного imei сначала целиком идёт один номер телефона, а затем целиком другой,
    причиной может быть смена владельца аппарата, а если сим-карты аппарата с заданным imei меняются поочерёдно,
    тогда можем считать, что это сим-карты одного владельца.
    :param df_temp:
    :return:
    """
    msisdn_ = df_temp['msisdn']
    found = []
    last_msi_2 = None
    for i in range(len(msisdn_) - 1):
        msi_1 = msisdn_.iloc[i]
        msi_2 = msisdn_.iloc[i + 1]
        if msi_1 not in found:
            found.append(msi_1)
        if msi_2 not in found:
            found.append(msi_2)
            last_msi_2 = msi_2
            continue
        elif (last_msi_2 is not None) & (last_msi_2 != msi_2):
            return True
    return False


def enrich_train(df_train, df_test):
    """
    Мы исходим из предположения, что если в одном аппарате меняются время от времени сим-карты, то эти сим-карты
    должны принадлежать одному человеку. С трудом видться ситуация, когда два разных владельца разных сим-карт
    по очереди вставляют их в один аппарат.
    Входные данные уже отсортированы по дате, но мы не уверены, что так будет и в дальнейшем.
    :param df_train:
    :param df_test:
    """
    imeis = df_test['imei'].value_counts().index
    for imei in imeis:
        if imei is np.nan:
            continue
        df_temp = df_test.loc[df_test['imei'] == imei]
        # Если менее двух сим-карт связано с одним imei, определять здесь зависимость смысла нет
        index = df_temp['msisdn'].value_counts().index.values
        if len(index) < 2:
            # print(imei, index)
            continue
        print('Найден imei с более чем одним связанным номером:\n', imei, index)
        if owner_is_the_same(df_temp):
            df_train = pd.concat([df_train, df_temp])
            print('df_train extended!')
        pass
    print('')


def convert_to_position(lat, lon, max_dist, start_angle, end_angle):
    """
    Преобразуем информацию в приближенное реальному положение пользователя номера.
    :param lat:
    :param lon:
    :param max_dist:
    :param start_angle:
    :param end_angle:
    :return:
    """
    if end_angle < start_angle:
        end_angle = end_angle + 360
    mid_angle = (end_angle - start_angle) % 360
    dist = max_dist * 0.82
    lat_step = 0.000008974
    lon_step = 0.0000161102
    lat = lat + math.cos(mid_angle * math.pi / 180) * dist * lat_step
    lon = lon + math.sin(mid_angle * math.pi / 180) * dist * lon_step
    return lat, lon


def convert_df_to_user_positions(df):
    new_df = df.drop(['max_dist', 'start_angle', 'end_angle'], axis=1)
    for i in range(df.shape[0]):
        print('row %d' % i)
        row = df.iloc[i, :]
        lat, lon = convert_to_position(
            row['lat'], row['long'], row['max_dist'], row['start_angle'], row['end_angle'])
        new_df.loc[i, 'lat'] = lat
        new_df.loc[i, 'long'] = lon
    return new_df


def count_distance_between_two_points(lat1, lon1, lat2, lon2):
    """
    Считаем расстояние между двумя точками, заданными координатами
    """
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def get_closest(to_value, in_array):
    """
    Поиск ближайшего значения в отсортированном (!!!) по возрастанию (!!!) массиве
    """
    last_val = 0
    for x in in_array:
        if x < to_value:
            last_val = x
        else:
            result = x if abs(x - to_value) < abs(last_val - to_value) else last_val
            return result
    return last_val


def prepare_data(x_df, y_df):
    """
    Функция ищет пары ближайших по времени событий для разных телефонных номеров.
    Для найденной пары ближайших событий вычисляются расстояние в пространстве и временной лаг

    !!! ВНИМАНИЕ !!!
    Данная функция является областью совершенствования качества прогноза методами Feature Engineering,
    поскольку именно здесь разные временные ряды двух разных номеров преобразуются в статическую строку показателей.

    Возвращаемые характеристики пары - это именно те данные, на которых будет обучаться и предсказывать алгоритм ML.

    :param x_df: подвыборка исходного обучающего датафрейма для номера телефона X
    :param y_df: подвыборка исходного обучающего датафрейма для номера телефона Y
    :return: характеристики пары:
                                mean_dist,
                                mean_time_diff,
                                num_dist_less_10_km,
                                num_dist_less_5_km,
                                num_dist_less_1_km,
                                num_dist_less_05_km
    """

    df_1, df_2 = (x_df, y_df) if x_df.shape[0] <= y_df.shape[0] else (y_df, x_df)
    long_2_values = df_2['long'].values
    lat_2_values = df_2['lat'].values
    tstamp_2_values = df_2['tstamp'].values
    distances_and_time_diffs = []

    for row_1 in df_1.itertuples():
        tstamp_1 = row_1[6]
        longitude_1 = row_1[7]
        latitude_1 = row_1[8]

        closest_tstamp = get_closest(tstamp_1, tstamp_2_values)

        index = None
        for i, item in enumerate(tstamp_2_values):
            if item == closest_tstamp:
                index = i

        longitude_2 = long_2_values[index]
        latitude_2 = lat_2_values[index]

        distance = count_distance_between_two_points(
            latitude_1, longitude_1, latitude_2, longitude_2)

        # Не замерено, должно быть быстро
        distances_and_time_diffs.append((distance, abs(tstamp_1 - closest_tstamp)))

    distances = [x[0] for x in distances_and_time_diffs]
    time_diffs = [x[1] for x in distances_and_time_diffs]
    mean_dist = sum(distances) / len(distances)
    mean_time_diff = sum(time_diffs) / len(time_diffs)
    num_dist_less_10_km = len([dist for dist in distances if dist < 10])
    num_dist_less_5_km = len([dist for dist in distances if dist < 5])
    num_dist_less_1_km = len([dist for dist in distances if dist < 1])
    num_dist_less_05_km = len([dist for dist in distances if dist < 0.5])
    return mean_dist, mean_time_diff, num_dist_less_10_km, num_dist_less_5_km, num_dist_less_1_km, num_dist_less_05_km


def get_answer(x_number, y_number, pair_df):
    if pair_df.loc[(pair_df[0].isin([x_number, y_number]))
            & (pair_df[1].isin([x_number, y_number]))].shape[0] > 0:
        return 1
    return 0


def prepare_train_df(modified_path, facts_path, train_path):
    """
    Функция, запускающая преобразование исходного массива обучающих данных в новый датафрейм,
    в котором строки - это характеристики одной конкретной пары телефонных номеров из исходных данных.
    Итоговое число строк есть число сочетаний из N по 2, где N - это число разных телефонных номеров в выборке.
    Это N*(N-1)/2
    Для тысяч номеров это миллионы пар. По этой причине предсказания будут организованы потоковыми.

    :param modified_path:
    :param facts_path:
    :param train_path:
    :return:
    """
    pair_df = pd.read_csv(facts_path, header=None)
    has_pair_list = list(pair_df[0].values)
    has_pair_list.extend(list(pair_df[1].values))
    df = pd.read_csv(modified_path)
    df_train = df.loc[df['msisdn'].isin(has_pair_list)]

    # df_test = df.loc[~df['msisdn'].isin(has_pair_list)]
    # TODO Пропускаем, пока отсутствует механизм добавления в pair_df найденных пар
    # enrich_train(df_train, df_test)

    # Раскомментируйте для вывода траекторий передвижений абонентов
    # demonstrate_movements(df_train, pair_df)

    new_df_train = None
    train_numbers = df_train['msisdn'].value_counts().index.values
    for i, x_number in enumerate(train_numbers):
        for j, y_number in enumerate(train_numbers):
            if j <= i:
                continue
            x_df = df_train.loc[df_train['msisdn'] == x_number]
            y_df = df_train.loc[df_train['msisdn'] == y_number]
            code = str(x_number) + '_' + str(y_number)
            answer = get_answer(x_number, y_number, pair_df)
            num_pair_values = [code, answer]
            num_pair_values.extend(list(prepare_data(x_df, y_df)))
            if new_df_train is None:
                new_df_train = pd.DataFrame(data=[num_pair_values], columns=new_df_columns)
            else:
                new_df_train = pd.concat([new_df_train,
                                          pd.DataFrame(data=[num_pair_values], columns=new_df_columns)])
            print('Pairs:', new_df_train.shape[0])

    print('Saving train Data Frame')
    new_df_train.to_csv(train_path, index=False, encoding='utf-8')


def demonstrate_movements(df_train, pair_df):
    for i in range(pair_df.shape[0]):
        x_number = pair_df.iloc[i, 0]
        y_number = pair_df.iloc[i, 1]
        x_df = df_train.loc[df_train['msisdn'] == x_number]
        y_df = df_train.loc[df_train['msisdn'] == y_number]
        show_figs(x_df, y_df)


def show_figs(x_df, y_df):
    x, y, z = x_df['long'].values, x_df['lat'].values, x_df['tstamp'].values
    x1, y1, z1 = y_df['long'].values, y_df['lat'].values, y_df['tstamp'].values
    # plot_pair_movements(x, x1, y, y1, z, z1)
    plot_pair_movements_2d(x, x1, y, y1)


def prepare_model(train_path,
                  from_val: 'bottom border of zero-class rows sample',
                  to_val: 'upper border of zero-class rows sample'):
    """
    BEST Precision, actual Recall = 0.80639589169, 0.11 в среднем у юзера прогнозируется 1-4 номера, реже больше
    Для смещения в сторону точности увеличте границы диапазона from_val : to_val
    Для смещения в сторону полноты уменьшите границы диапазона from_val : to_val
    ----------------
    При to_val = 90 Precision, actual Recall = 0.89, 0.066
    При to_val = 15 Precision, actual Recall = 0.45, 0.45
    Однако в последнем случае на выходе имеем слишком много False Positive - прогнозов, и одному
    юзеру начинает приписываться до 20-ти разных номеров.
    """
    df = pd.read_csv(train_path)

    x = df.drop(['code', 'answer'], axis=1).as_matrix()
    full_scaled_x = scale(x)
    full_y = df['answer']

    df_pos = df.loc[df['answer'] == 1]
    df_neg = df.loc[df['answer'] == 0]

    precisions = {}
    for i in range(from_val, to_val):
        number_of_negatives = i * 1000
        train_data_frames = \
            split_to_5_train_sets(df_neg, df_pos, number_of_negatives)
        precision, recall = calculate_f1_score_for_zero_sample_of(train_data_frames, full_scaled_x, full_y)
        precisions[precision] = (number_of_negatives, recall)

    # Максимизируем качество по гармоническому среднему точности и полноты
    max_score = max(list(precisions.keys()))
    best_number_of_negatives = precisions[max_score][0]
    recall = precisions[max_score][1]
    print('---=== BEST Precision, actual Recall and number of zero-answered rows ===---')
    print(max_score, recall, best_number_of_negatives)
    print('---=== Creating model ===---')
    df_neg_1 = df_neg.sample(best_number_of_negatives, replace=True, random_state=111)
    train_df = pd.concat([df_pos, df_neg_1])
    return fit_model_for(train_df)


def calculate_f1_score_for_zero_sample_of(train_data_frames, full_scaled_x, full_y):
    all_metrics = [using_log_reg(train_data_frames[0], full_scaled_x, full_y),
                   using_log_reg(train_data_frames[1], full_scaled_x, full_y),
                   using_log_reg(train_data_frames[2], full_scaled_x, full_y),
                   using_log_reg(train_data_frames[3], full_scaled_x, full_y),
                   using_log_reg(train_data_frames[4], full_scaled_x, full_y)]

    mean_precision = np.mean(np.asarray([x[1] for x in all_metrics]))
    mean_recall = np.mean(np.asarray([x[2] for x in all_metrics]))
    return mean_precision, mean_recall


def split_to_5_train_sets(df_neg, df_pos, zero_num):
    df_neg_1 = df_neg.sample(zero_num, replace=True, random_state=111)
    df_neg_2 = df_neg.sample(zero_num, replace=True, random_state=112)
    df_neg_3 = df_neg.sample(zero_num, replace=True, random_state=113)
    df_neg_4 = df_neg.sample(zero_num, replace=True, random_state=114)
    df_neg_5 = df_neg.sample(zero_num, replace=True, random_state=115)
    df_train_1 = pd.concat([df_pos, df_neg_1])
    df_train_2 = pd.concat([df_pos, df_neg_2])
    df_train_3 = pd.concat([df_pos, df_neg_3])
    df_train_4 = pd.concat([df_pos, df_neg_4])
    df_train_5 = pd.concat([df_pos, df_neg_5])
    return df_train_1, df_train_2, df_train_3, df_train_4, df_train_5


def fit_model_for(train_df):
    x = train_df.drop(['code', 'answer'], axis=1).as_matrix()
    scaled_x = scale(x)
    y = train_df['answer']
    clf = LogisticRegression(C=100, random_state=111)
    clf.fit(scaled_x, y)
    return clf


def using_log_reg(train_df, full_scaled_x, full_y):
    clf = fit_model_for(train_df)
    predict = clf.predict(full_scaled_x)
    # Metrics
    precision = precision_score(full_y, predict)
    recall = recall_score(full_y, predict)
    f1 = f1_score(full_y, predict)
    return f1, precision, recall


def predict_test_data(model, facts_path, modified_path, results_path):
    pair_df = pd.read_csv(facts_path, header=None)
    has_pair_list = list(pair_df[0].values)
    has_pair_list.extend(list(pair_df[1].values))
    df = pd.read_csv(modified_path)

    new_df_test = None
    result = None
    all_numbers = df['msisdn'].value_counts().index.values
    last_element_index = len(all_numbers) - 1
    for i, x_number in enumerate(all_numbers):
        for j, y_number in enumerate(all_numbers):
            if j <= i:
                continue
            x_df = df.loc[df['msisdn'] == x_number]
            y_df = df.loc[df['msisdn'] == y_number]
            code = str(x_number) + '_' + str(y_number)
            num_pair_values = [code]
            num_pair_values.extend(list(prepare_data(x_df, y_df)))
            if new_df_test is None:
                new_df_test = pd.DataFrame(data=[num_pair_values], columns=new_test_df_columns)
            else:
                new_df_test = pd.concat([new_df_test,
                                         pd.DataFrame(data=[num_pair_values], columns=new_test_df_columns)])
            shape_0 = new_df_test.shape[0]
            if (shape_0 > 10000) \
                    | ((i == last_element_index - 1) & (j == last_element_index)):
                code = new_df_test['code']
                x = scale(new_df_test.drop(['code'], axis=1).as_matrix())
                pred = model.predict(x)
                local_result = pd.DataFrame(data={'code': code, 'pred': pred})
                if result is None:
                    result = local_result
                else:
                    result = pd.concat([result, local_result])
                print('---=== Batch counted, creating new batch to predict... ===---')
                new_df_test = None

    # Все данные обработаны, сохраняем результат
    result.to_csv(results_path, index=False, encoding='utf-8')


def save_only_found_pairs(results_path, only_pairs_path):
    df = pd.read_csv(results_path)
    codes = df.loc[df['pred'] == 1]['code']
    num_1_column = []
    num_2_column = []
    for code in codes:
        num_1_column.append(code.split('_')[0])
        num_2_column.append(code.split('_')[1])
    df_new = pd.DataFrame(data={'num_1': num_1_column, 'num_2': num_2_column})
    df_new.to_csv(only_pairs_path, index=False, encoding='utf-8')


def save_in_required_format(results_path, only_pairs_path, target_path):
    df = pd.read_csv(results_path)
    df_pairs = pd.read_csv(only_pairs_path)
    df_pairs_num_1 = df_pairs['num_1'].values
    df_pairs_num_2 = df_pairs['num_2'].values
    codes = df['code']
    num_1_column = []
    for code in codes:
        num_1_column.append(code.split('_')[0])
    df['num_1'] = num_1_column
    all_numbers = df['num_1'].value_counts().index.values

    wired_nums_list = []
    for i, num in enumerate(all_numbers):
        print(i)
        result = [num]
        to_break_if_not_equals = False
        for j, prim_num in enumerate(df_pairs_num_1):
            if str(prim_num) == str(num):
                to_break_if_not_equals = True
                result.extend([df_pairs_num_2[j]])
            elif to_break_if_not_equals:
                break
        wired_nums_list.append(result)

    new_df = pd.DataFrame(data={'wired_numbers': wired_nums_list})
    new_df.to_csv(target_path, index=True, encoding='utf-8')


def main_function():
    """
    Точка входа. Вызов функций:
        modify(primary_path, modified_path)
        prepare_train_df(modified_path, facts_path, train_path)
    можно произвести единожды, затем их промежуточные результаты будут сохранены на диск.
    Эти промежуточные результаты затем могут быть вычитаны сразу функцией:
        analise(modified_path, train_path, results_path)
    -> Рекомендуется после первого запуска их закомментировать.
    ----------------------------------------------------------------------------

    """
    # Путь на одноимённый файл. Содержимое этого файла ДОЛЖНО быть уже отсортировано по колонке tstamp!!!
    primary_path = 'D:/TEMP/datasets1/test/02_Data_test.csv'
    # Путь, в котором модифицированны исходный файл должен быть сохранён
    modified_path = 'D:/TEMP/datasets1/test/02_Data_test_modified.csv'
    # Путь на .csv файл, сохранённый из исходного 01_Facts.xlsx
    facts_path = 'D:/TEMP/datasets1/facts/pairs.csv'
    # Путь, в котором должны будут сохраниться подготовленные для обучения алгоритма данные
    train_path = 'D:/TEMP/datasets1/test/train.csv'
    # Путь для сохранения всех прогнозов
    results_path = 'D:/TEMP/datasets1/test/results.csv'
    # Путь для сохранения всех найденных пар
    only_pairs_path = 'D:/TEMP/datasets1/test/results_only_pairs.csv'
    # Путь для сохранения в формате: User_id, Все номера этого юзера
    required_format_output_path = 'D:/TEMP/datasets1/test/required_format_output.csv'

    # Можно закомментировать две следующие строки после первого запуска,
    # если следующие запуски будут на тех же данных
    modify(primary_path, modified_path)
    prepare_train_df(modified_path, facts_path, train_path)
    model = prepare_model(train_path, from_val=80, to_val=90)
    predict_test_data(model, facts_path, modified_path, results_path)
    save_only_found_pairs(results_path, only_pairs_path)
    save_in_required_format(results_path, only_pairs_path, required_format_output_path)

main_function()
