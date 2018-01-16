"""
Принцип подготовки трейна и теста из итоговых датафреймов активов
-----------------------------------------------------------------
Сначала все датафреймы, предназначенные для предсказания, конкатенируем вместе (pd.concat([], axis=1)):
а,b + a,b + a,b = t1_a,t1_b,t2_a,t2_b,t3_a,t3_b
-----------------------------------------------
1,1   1,1   1,1     1    1    1    1    1    1
2,2   2,2   2,2     2    2    2    2    2    2
3,3   3,3   3,3     3    3    3    3    3    3
-----------------------------------------------
Далее для каждого из дф активов, по которым нужно строить предсказания, формируем колонку ответов Y,
после чего справа "приклеиваем" ранее сформированную таблицу общих признаков.
Затем все получившиеся "склеенные" дф конкатенируем сверху вниз (pd.concat([], axis=0))

Так мы получили общую таблицу, в которой одной строке соответствует один день одного актива,
а также значение одного из его показателей, сдвинутое на энное время вперёд (Y).

Поскольку изначально все таблицы уже отсортированы по возрастанию времени (сверху вниз),
отрезаем одну треть с конца - это будет тестовая выборка. Начало будет трейном.

Отдельно будет предусмотрена стратегия прогнозирования по всем предшествующим дням для предсказываемого,
и так для всего периода. Такой прогноз возможно будет точнее.
"""

import pandas as pd
import numpy as np
import os
import math
import glob
import ntpath
import shutil
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
import xgboost as xgb
from v2.utils import check_paths, weighted_avg_and_std
from v2.strategies.extractions import mock, extract_daily_data

to_predict = ['BANE', 'CHMF', 'MTSS', 'NLMK', 'NVTK', 'RTKM', 'SNGS', 'TATN', 'TRNFP', 'URKA', 'VSMO']
to_predict_dict = {to_predict[i]: i for i in range(len(to_predict))}
to_be_features = ['ALRS', 'GAZP', 'GMKN', 'LKOH', 'ROSN', 'SBER', 'VTBR']
to_fit_by_date = ['comex.GC', 'EURRUB', 'ICE.BRN', 'USDRUB']

# importancies_df = pd.read_csv('feature_importance.csv', header=None)
#
#
# def reduce_by_importance(border):
#     temp_df = importancies_df.loc[importancies_df[1] > border]
#     return temp_df[0].values


# to_predict = ['BANE', 'CHMF', 'MTSS', 'NLMK', 'NVTK', 'RTKM', 'SNGS', 'TATN', 'TRNFP', 'URKA', 'VSMO', 'VTBR']
# to_be_features = ['ALRS', 'GAZP', 'GMKN', 'LKOH', 'MGNT', 'ROSN', 'SBER']


# to_be_features = ['ALRS', 'comex.GC', 'EURRUB', 'GAZP', 'GMKN', 'ICE.BRN', 'LKOH', 'MGNT', 'ROSN', 'SBER', 'USDRUB']


def show_stats(real, pred):
    print('MSE test:', mse(real, pred))
    print('MAE test:', mae(real, pred))
    diff = np.asarray(real) - np.asarray(pred)
    print('Mistakes std:', np.std(diff))
    pred_real_dict = {k: v for (k, v) in zip(pred, real)}
    high_2_bord = 0.006
    high_1_bord = 0.0045
    low_2_bord = -0.003
    low_1_bord = -0.0025
    highest = {k: v for (k, v) in pred_real_dict.items() if k > high_2_bord}
    high = {k: v for (k, v) in pred_real_dict.items() if (k > high_1_bord) & (k <= high_2_bord)}
    lowest = {k: v for (k, v) in pred_real_dict.items() if k < low_2_bord}
    low = {k: v for (k, v) in pred_real_dict.items() if (k < low_1_bord) & (k >= low_2_bord)}
    medium = {k: v for (k, v) in pred_real_dict.items() if (k <= high_1_bord) & (k >= low_1_bord)}
    print('MAE:')
    print('highest', len(highest), mae([x for x in highest.keys()], [x for x in highest.values()]))
    print('high', len(high), mae([x for x in high.keys()], [x for x in high.values()]))
    print('lowest', len(lowest), mae([x for x in lowest.keys()], [x for x in lowest.values()]))
    print('low', len(low),
          mae([x for x in low.keys()], [x for x in low.values()]))
    print('medium', len(medium), mae([x for x in medium.keys()], [x for x in medium.values()]))

    print('MEANs:')
    print('highest', np.mean(np.asarray([x for x in highest.keys()])), np.mean([x for x in highest.values()]))
    print('high', np.mean(np.asarray([x for x in high.keys()])), np.mean([x for x in high.values()]))
    print('lowest', np.mean(np.asarray([x for x in lowest.keys()])), np.mean([x for x in lowest.values()]))
    print('low', np.mean(np.asarray([x for x in low.keys()])), np.mean([x for x in low.values()]))
    print('medium', np.mean(np.asarray([x for x in medium.keys()])), np.mean([x for x in medium.values()]))

    print('STD:')
    print('highest', np.std(np.asarray([x for x in highest.keys()])), np.std([x for x in highest.values()]))
    print('high', np.std(np.asarray([x for x in high.keys()])), np.std([x for x in high.values()]))
    print('lowest', np.std(np.asarray([x for x in lowest.keys()])), np.std([x for x in lowest.values()]))
    print('low', np.std(np.asarray([x for x in low.keys()])), np.std([x for x in low.values()]))
    print('medium', np.std(np.asarray([x for x in medium.keys()])), np.std([x for x in medium.values()]))

    print('MISTAKES STD:')
    print('highest', np.std(np.asarray([x for x in highest.keys()]) - np.asarray([x for x in highest.values()])))
    print('high', np.std(np.asarray([x for x in high.keys()]) - np.asarray([x for x in high.values()])))
    print('lowest', np.std(np.asarray([x for x in lowest.keys()]) - np.asarray([x for x in lowest.values()])))
    print('low', np.std(np.asarray([x for x in low.keys()]) - np.asarray([x for x in low.values()])))
    print('medium', np.std(np.asarray([x for x in medium.keys()]) - np.asarray([x for x in medium.values()])))


class Pipeline:
    """
    Подготовка данных для анализа с промежуточными сохранениями.
    Если выбрать параметр need_rewrite = True для функции, для которой нет данных в целевой папке,
    просто по всей последующей цепочке ничего не будет обработано.
    """

    def __init__(self,
                 y_col_name='last_part_mean',
                 in_memory: 'режим быстрого прогноза с последующим обновлением имеющихся данных на диске' = False,
                 initial_path: 'путь до директории с новыми данными' = None):
        self.y_col_name = y_col_name
        self.need_rewrite = False
        self.headers = ['TICKER', 'PER', 'DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']
        self.df_dict = None
        self.in_memory = in_memory
        if in_memory:
            if initial_path is None:
                raise Exception('Если in_memory=True, initial_path должен быть определён!')
            self.df_dict = self.init_df_dict_from_disc(initial_path)

    def should_be_rewritten(self, func_name, need_rewrite):
        self.need_rewrite = need_rewrite if need_rewrite else self.need_rewrite
        if self.need_rewrite:
            print('-----------------\n'
                  'EXECUTING: %s\n'
                  '-----------------' % func_name)
        else:
            print('-----------------\n'
                  'SKIPPED: %s\n'
                  '-----------------' % func_name)
        return self.need_rewrite

    @staticmethod
    def should_be_skipped(col_name, skip_list):
        for string in skip_list:
            if string in col_name:
                return True
        return False

    @staticmethod
    def init_df_dict_from_disc(from_path):
        return {ntpath.basename(file_path): pd.read_csv(file_path)
                for file_path in glob.glob(from_path + '*.csv')}

    def transformation_in_memory(self, exec_func: {'help': 'Функция преобразованный датафрейма',
                                                   'type': callable((pd.DataFrame, tuple))} = mock,
                                 skip_columns=()):
        for filename, df in self.df_dict.items():
            self.df_dict[filename] = exec_func(df, skip_columns)
        return self

    def transformation(self, from_path, to_path,
                       exec_func: {
                           'help': 'Функция, принимающая df и скип_лист колонок и возвращающая преобразованный df',
                           'type': callable((pd.DataFrame, tuple))} = mock,
                       skip_columns=(), need_rewrite=False):
        if self.in_memory:
            return self.transformation_in_memory(
                exec_func=exec_func, skip_columns=skip_columns)

        # Трансформация с сохранением промежуточного результата
        check_paths((from_path, to_path))
        if not self.should_be_rewritten(exec_func.__name__, need_rewrite):
            return self
        df_dict = self.get_data_frames_dict(from_path)
        self.apply_function_and_save(df_dict, exec_func, to_path, skip_columns=skip_columns)
        return self

    @staticmethod
    def apply_function_and_save(df_dict, func, to_path, skip_columns=()):
        for filename, df in df_dict.items():
            # Will be saved only if to_path is not None and not empty
            if to_path:
                func(df, skip_columns).to_csv(to_path + filename, index=False, encoding='utf-8')
            else:
                func(df, skip_columns)

    def get_data_frames_dict(self, from_path) -> dict:
        if self.in_memory:
            return self.df_dict
        else:
            return self.init_df_dict_from_disc(from_path)

    def fit_by_dates(self, from_path, to_path, need_rewrite=False):
        """
        Операция очень дорогая, поэтому не проводится на стадии минутных таблиц.
        Должна быть проведена сразу после преобразования из минутных таблиц в целевой формат.
        """
        # Проверка только для режима с пошаговым сохранением промежуточных результатов
        if not self.in_memory:
            check_paths((from_path, to_path))
            if not self.should_be_rewritten('fit_by_dates', need_rewrite):
                return self

        df_dict = self.get_data_frames_dict(from_path)
        gazp_df = df_dict['GAZP.csv'].copy()
        gazp_df.columns = range(gazp_df.shape[1])
        valid_dates = [date for date in gazp_df[0]]

        for filename, df in df_dict.items():
            if filename.replace('.csv', '') not in to_fit_by_date:
                continue
            df_temp = df.loc[df['DATE'].isin(valid_dates)]
            old_len = df_temp.shape[1]
            df_temp = df_temp.merge(gazp_df, left_on='DATE', right_on=0, how='outer').iloc[:, :old_len] \
                .sort_values(by='DATE').fillna(method='ffill')
            if self.in_memory:
                self.df_dict[filename] = df_temp
            else:
                df_temp.to_csv(to_path + filename, index=False, encoding='utf-8')

        return self

    # Обогащаем значениями статистик окон прошедших периодов callable
    def generate_new_columns(self, df, skip_columns=()):
        new_df = df.drop(['DATE'], axis=1).copy()
        for key in new_df.keys():
            new_df['%s-1' % key] = new_df[key].shift(1)
            new_df['%s-2' % key] = new_df[key].shift(2)
            new_df['%s-3' % key] = new_df[key].shift(3)
            if self.should_be_skipped(key, skip_columns):
                continue
            window10 = new_df['%s-1' % key].rolling(window=10)
            new_df['%s-1_w10_min' % key] = window10.min()
            new_df['%s-1_w10_mean' % key] = window10.mean()
            new_df['%s-1_w10_max' % key] = window10.max()

        new_df['date_diff'] = df['DATE'] - df['DATE'].shift(1)
        new_df['date_diff'] = new_df['date_diff'].mask(new_df['date_diff'] == 70, 1)
        new_df['date_diff'] = new_df['date_diff'].mask(new_df['date_diff'] > 5, 5)

        new_df['yesterday_last_part_mean'] = new_df['end_parts_mean_1'].shift(1)
        # После добавления этих колонок MAE немного уменьшается и чуть растёт MSE (на средней за день),
        # но эти колонки мы хотим предсказывать именно в таком виде.
        new_df['TEMP_ylpm'] = new_df['yesterday_last_part_mean'] * new_df['day_mean-1'] + new_df['day_mean-1']
        new_df['qtr_mean_0'] = (new_df['qtr_mean_0'] - new_df['TEMP_ylpm']) / new_df['TEMP_ylpm']
        new_df['qtr_mean_1'] = (new_df['qtr_mean_1'] - new_df['TEMP_ylpm']) / new_df['TEMP_ylpm']
        new_df = new_df.drop(['TEMP_ylpm'], axis=1)

        return pd.concat([df[['DATE']], new_df], axis=1).dropna(how='any')

    def convert_to_diffs(self, df,
                         skip_columns: 'parts of column names to decide if column should be skipped' = ()):
        date, temp = 'DATE', 'TEMP'
        df_temp = df.drop(date, axis=1).copy()
        for key in df_temp.keys():
            if self.should_be_skipped(key, skip_columns):
                continue
            df_temp[temp] = df_temp[key].shift(1)
            df_temp[key] = (df_temp[key] - df_temp[temp]) / df_temp[temp]
        return pd.concat([df[[date]], df_temp.drop(temp, axis=1)], axis=1).iloc[1:, :].round(5)

    def prepare_separate_train_tables(self, from_path, to_path, need_rewrite=False):
        # Проверка только для режима с пошаговым сохранением промежуточных результатов
        if not self.in_memory:
            check_paths((from_path, to_path))
            if not self.should_be_rewritten('prepare_separate_train_tables', need_rewrite):
                return self

        df_dict = self.get_data_frames_dict(from_path)

        if self.in_memory:
            self.glue_to_historical_data(df_dict, from_path)

        # Сначала сформируем общие для всех признаки
        feature_dfs = {key: value for (key, value) in df_dict.items() if key.replace('.csv', '') in to_be_features}
        new_columns = []
        for key, df_temp in feature_dfs.items():
            # Колонки дат дропаем, так как ранее уже должны были проверить, что все строки
            # соответствуют одинаковым датам для разных датафреймов
            new_columns = new_columns + [key.replace('.csv', '_') + s for s in df_temp.keys() if s != 'DATE']

        features_df = pd.DataFrame(data=pd.concat([df.drop(['DATE'], axis=1)
                                                   for df in feature_dfs.values()], axis=1).as_matrix(),
                                   columns=new_columns)

        # Составим словарь датафреймов, для которых будем предсказывать значения
        predict_dfs = {key: value for (key, value) in df_dict.items() if key.replace('.csv', '') in to_predict}
        # Склеим каждый из них со сформированным выше датафреймом общих фич и сохраним
        for filename, df_temp in predict_dfs.items():
            # При работе в памяти индексы могут не совпадать, приведём их в соответствие
            df_temp.index = features_df.index
            pd.concat([df_temp, features_df], axis=1) \
                .round(5) \
                .to_csv(to_path + filename, index=False, encoding='utf-8')

        return self

    def glue_to_historical_data(self, df_dict, from_path):
        historical_df_dict = self.init_df_dict_from_disc(from_path)
        for key in df_dict.keys():
            df_dict[key] = pd.concat([historical_df_dict[key], df_dict[key]]).drop_duplicates()

    def predict_tomorrow(self, from_path, to_path, y_col_names, n_estimators=90, need_rewrite=True):
        # TODO refactor!
        if not need_rewrite:
            return self
        temp_path = to_path + 'train_test/'
        file_paths_list = [ntpath.basename(file_path).replace('.csv', '')
                           for file_path in glob.glob(from_path + '*.csv')]
        df = pd.DataFrame(data={'TICKER': file_paths_list})
        for y_col_name in y_col_names:
            self.prepare_train_and_test(from_path=from_path, to_path=temp_path, need_rewrite=True,
                                        y_col_name=y_col_name, n_test=1, real_prediction=True)
            if not self.should_be_rewritten('predict_tomorrow %s' % y_col_name, True):
                return self
            df_train = pd.read_csv(temp_path + 'train.csv')
            df_test = pd.read_csv(temp_path + 'test.csv')

            y_train = df_train['y']
            x_train = df_train.drop(['y', 'DATE'], axis=1).as_matrix()
            x_test = df_test.drop(['y', 'DATE'], axis=1).as_matrix()
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, seed=1, n_jobs=8).fit(X=x_train, y=y_train)
            pred = xgb_model.predict(x_test)
            df[y_col_name] = pd.Series(data=pred)

        # for key in df.keys():
        #     df[key] = df[key].round(5) if df[key].dtype in [np.float32, np.float64] else df[key]
        df.to_csv(to_path + 'tomorrow_predictions.csv', index=False, encoding='utf-8')
        return self

    def prepare_train_and_test(self, from_path, to_path, n_test=30,
                               need_rewrite=False, real_prediction=False):
        # Проверка только для режима с пошаговым сохранением промежуточных результатов
        if not self.in_memory:
            check_paths((from_path, to_path))
            if not self.should_be_rewritten('prepare_train_and_test', need_rewrite):
                return self

        df_dict = self.get_data_frames_dict(from_path)

        # Добавляем колонку ответов (Удалим последнюю строку с NaN вместо ответа,
        # если не ставится цель реальный прогноз по последнему дню седующего)
        for key, df_temp in df_dict.items():
            df_temp['y'] = df_temp[self.y_col_name].shift(-1)
            df_dict[key] = df_temp if real_prediction else df_temp.iloc[:-1, :]

        # Разбиваем на train и test
        train_dfs, test_dfs = [], []
        for key, df_temp in df_dict.items():
            train_dfs.append(df_temp.iloc[:-n_test, :])
            test_dfs.append(df_temp.iloc[-n_test:, :])

        # Склеем по вертикали и сохраним
        pd.concat(train_dfs) \
            .round(5) \
            .to_csv(to_path + 'train.csv', index=False, encoding='utf-8')
        pd.concat(test_dfs) \
            .round(5) \
            .to_csv(to_path + 'test.csv', index=False, encoding='utf-8')

        return self

    def analise_using_xgboost(self, from_path, to_path, n_estimators=100, need_rewrite=False):
        check_paths((from_path, to_path))
        if not self.should_be_rewritten('analise', need_rewrite):
            return self
        df_train = pd.read_csv(from_path + 'train.csv')
        df_test = pd.read_csv(from_path + 'test.csv')
        print(df_test.describe())
        y_train = df_train['y']
        x_train = df_train.drop(['y', 'DATE'], axis=1).as_matrix()
        y_test = df_test['y']
        x_test = df_test.drop(['y', 'DATE'], axis=1).as_matrix()
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, seed=1, n_jobs=8).fit(X=x_train, y=y_train)
        pred = xgb_model.predict(x_test)
        show_stats(y_test, pred)

        return self

    def analise_using_xgboost_iteratively(self, from_path, to_path, n_estimators=100):
        check_paths((from_path, to_path))
        if not self.should_be_rewritten('analise', True):
            return self
        df_train = pd.read_csv(from_path + 'train.csv')
        df_test = pd.read_csv(from_path + 'test.csv')
        df_test = df_test.sort_values(by='DATE')
        all_dates = sorted(df_test['DATE'].value_counts().index)
        # print(df_test.describe())
        print('===--- ITERATIVELY ---===')

        pred = []
        names = None
        feature_importancies = []
        zip_to_train = None
        for date in all_dates:
            print('Processing date:', date)
            test_day = df_test[df_test['DATE'] == date]
            if zip_to_train is None:
                zip_to_train = test_day
            else:
                df_train = pd.concat([df_train, zip_to_train])
                zip_to_train = test_day

            y_train = df_train['y']
            x_train = df_train.drop(['y', 'DATE'], axis=1).as_matrix()
            x_test = test_day.drop(['y', 'DATE'], axis=1).as_matrix()
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, seed=1, n_jobs=8).fit(X=x_train, y=y_train)
            pred_local = xgb_model.predict(x_test)
            pred.extend(pred_local)
            feature_importancies.append(xgb_model.feature_importances_)
            if names is None:
                names = list(df_train.drop(['y', 'DATE'], axis=1).keys())
        print('All dates complete! Saving result...')
        res_df = pd.DataFrame(data={self.y_col_name: pred,
                                    'DATE': df_test['DATE'].values,
                                    'last_end_parts_mean_1': df_test['end_parts_mean_1'].values})
        res_df.to_csv(to_path + self.y_col_name + '.csv', index=False, encoding='utf-8')
        self.show_feature_importancies(names, feature_importancies)
        show_stats(df_test['y'].values, pred)
        return self

    @staticmethod
    def show_feature_importancies(names, feature_importancies):
        sum_fi = None
        for fi in feature_importancies:
            if sum_fi is None:
                sum_fi = np.asarray(fi)
            else:
                sum_fi = sum_fi + np.asarray(fi)
        importancies = sum_fi / len(feature_importancies)
        res_dict = dict(zip(names, importancies))
        print('Feature importancies:\n', res_dict)

    @staticmethod
    def finish():
        print('Finished successfully!')
