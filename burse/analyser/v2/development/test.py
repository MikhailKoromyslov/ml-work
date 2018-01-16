from v2.prepare import Pipeline
from v2.strategies.extractions import extract_daily_data
import pandas as pd
import numpy as np
import glob
import ntpath


incorrect_dates = [20160732, 20160832, 20160931, 20161032, 20161131, 20161232, 20170132, 20170229, 20170332, 20170431,
                   20170532, 20170631, 20170732, 20170832, 20170931, 20171032, 20171131]
correct_dates = {20160732: 20160801,
                 20160832: 20160901,
                 20160931: 20161001,
                 20161032: 20161101,
                 20161131: 20161201,
                 20161232: 20170101,
                 20170132: 20170201,
                 20170229: 20170301,
                 20170332: 20170401,
                 20170431: 20170501,
                 20170532: 20170601,
                 20170631: 20170701,
                 20170732: 20170801,
                 20170832: 20170901,
                 20170931: 20171001,
                 20171032: 20171101,
                 20171131: 20171201}

# TODO доработать проект по следующим пунктам (вкл проверку гипотез):

"""
---=== Совершенствование модели ===---
1) Попробовать предсказывать сдвинутые значения, предварительно умноженные на 300 и возведённые в 3-ю степень.
   В этом случае 0.(3) будет границей: 0.(3)*3 = 1, 1**3 = 1. Что больше - сильно больше, что меньше - близко к нулю.
   Вероятно, так алгоритм сможет лучше предсказать случаи больших приростов (как в плюс, так и в минус)
   Этот результат после обратного преобразования предикта также можно попробовать постэкать с обычным.
2) ...

---=== Алгоритм принятия решений ===---
1) Ждать целевого для первичной сделки показателя не до 14:00, а до 12:30 - 13:00 (Сделано!)
2) ...

---=== Инфраструктура ===---
1) Патчи файлов с данными (Сделано!)
2) Когда-нибудь потом - сервис, автоматически выполняющий патч, запускающий обработку и возвращающий advice
3) ...
"""
# TODO после трансформации провести тест, проверяющий, что содержимое колонки DATE у всех датафреймов одинаковое!
pipe = Pipeline()
pipe.transformation(from_path='D:/TEMP/data_v2/',
                    to_path='D:/TEMP/data_v2/transformed/',
                    exec_func=extract_daily_data,
                    need_rewrite=False) \
    .fit_by_dates(from_path='D:/TEMP/data_v2/transformed/',
                  to_path='D:/TEMP/data_v2/transformed/',
                  need_rewrite=False) \
    .transformation(from_path='D:/TEMP/data_v2/transformed/',
                    to_path='D:/TEMP/data_v2/enriched/',
                    exec_func=pipe.generate_new_columns,
                    skip_columns=('day_std', 'qtr_std', 'end_parts', '_low'),
                    need_rewrite=False) \
    .transformation(from_path='D:/TEMP/data_v2/enriched/',
                    to_path='D:/TEMP/data_v2/diff/',
                    exec_func=pipe.convert_to_diffs,
                    skip_columns=('day_std', 'qtr_std', 'qtr_mean', 'date_diff', 'end_parts', 'diff_to_prev_last',
                                  '_growth', '_low', 'day_max_min_growth'),
                    need_rewrite=False) \
    .prepare_separate_train_tables(from_path='D:/TEMP/data_v2/diff/',
                                   to_path='D:/TEMP/data_v2/final/',
                                   need_rewrite=False) \
    .prepare_train_and_test(from_path='D:/TEMP/data_v2/final/',
                            to_path='D:/TEMP/data_v2/train_test/',
                            n_test=175,
                            need_rewrite=False) \
    .analise_using_xgboost_iteratively(from_path='D:/TEMP/data_v2/train_test/',
                                       to_path='D:/TEMP/data_v2/results/',
                                       n_estimators=90) \
    .finish()


# .predict_tomorrow(from_path='D:/TEMP/data_v2/final/',
#                   to_path='D:/TEMP/data_v2/results/',
#                   y_col_names=['day_mean', 'day_std', 'qtr_mean_0', 'qtr_mean_1'],
#                   need_rewrite=False) \
# .emulate_trades(from_path='D:/TEMP/data_v2/results/')


class Analyzer:
    """
    Подготовка данных для анализа с промежуточными сохранениями.
    Если выбрать параметр need_rewrite = True для функции, для которой нет данных в целевой папке,
    просто по всей последующей цепочке ничего не будет обработано.
    """

    def __init__(self):
        self.need_rewrite = False

    def emulate_trades(self, from_path):
        """
        Для упрощения вычисление прогнозов произведены заранее, только считываем прогнозы и эмулируем принятие решений.
        """
        day_mean_pred_df = pd.read_csv(from_path + 'day_mean.csv')
        day_std_pred_df = pd.read_csv(from_path + 'day_std.csv')
        qtr_mean_0_pred_df = pd.read_csv(from_path + 'qtr_mean_0.csv')
        qtr_mean_1_pred_df = pd.read_csv(from_path + 'qtr_mean_1.csv')

        df_pred = pd.concat([day_mean_pred_df,
                             day_std_pred_df[['day_std']],
                             qtr_mean_0_pred_df[['qtr_mean_0']],
                             qtr_mean_1_pred_df[['qtr_mean_1']]], axis=1)

        all_dates = sorted(df_pred['DATE'].value_counts().index)
        file_paths_list = [ntpath.basename(file_path).replace('.csv', '')
                           for file_path in glob.glob('D:/TEMP/data_v2/final/*.csv')]
        tickers_column = np.asarray([file_paths_list for x in range(len(all_dates))]).reshape(-1)
        df_pred['TICKER'] = tickers_column

        file_paths_dict = {ntpath.basename(file_path).replace('.csv', ''): file_path
                           for file_path in glob.glob('D:/TEMP/data_v2/final/*.csv')}
        for ticker, path in file_paths_dict.items():
            file_paths_dict[ticker] = path.replace('final', 'enriched')

        advices_by_date = {}
        for date in all_dates:
            advices_by_date[date] = df_pred[df_pred['DATE'] == date]
        for date in all_dates:
            advices_by_date[date] = self.give_trading_advise(advices_by_date[date])
        for date in all_dates:
            advices_by_date[date] = self.add_financial_results(advices_by_date[date], file_paths_dict)
        res = []
        res_df_no_zeroes = pd.concat([df for df in list(advices_by_date.values())])
        res_df_no_zeroes = res_df_no_zeroes.loc[res_df_no_zeroes['RESULT'] != 0].dropna()
        for df in advices_by_date.values():
            if 'RESULT' in list(df.keys()):
                res.extend(df.dropna()['RESULT'].values)
            else:
                res.extend([0])
        res_df = pd.DataFrame(data={'res': res}).fillna(0)
        res = [x if float(x) > -0.009 else -0.009 for x in res_df['res'].values]
        print('DONE! Total:', sum(res))

    def add_financial_results(self, advice: 'df of single date trade advice', file_paths_dict):
        if len(advice._values) == 0:
            return advice
        single_advice_list = []
        print('Processing advice for', advice['DATE'])
        for ticker in advice['TICKER'].values:
            single_advice = advice[advice['TICKER'] == ticker]
            path = file_paths_dict[ticker]
            df = pd.read_csv(path)
            last_day_mean = df[df['DATE'] == single_advice['DATE'].values[0]]['day_mean'].values[0]
            growth = single_advice['last_end_parts_mean_1'].values[0]
            single_advice['last_end_parts_mean_1'] = [last_day_mean + last_day_mean * growth]
            last_end_mean_abs = single_advice['last_end_parts_mean_1'].values[0]
            single_advice['qtr_mean_0'] = [
                last_end_mean_abs + last_end_mean_abs * single_advice['qtr_mean_0'].values[0]]
            single_advice['qtr_mean_1'] = [
                last_end_mean_abs + last_end_mean_abs * single_advice['qtr_mean_1'].values[0]]
            single_advice['buy_at'] = [last_end_mean_abs + last_end_mean_abs * single_advice['buy_at'].values[0]]
            single_advice['than_sell_at'] = [
                last_end_mean_abs + last_end_mean_abs * single_advice['than_sell_at'].values[0]]

            primary_path = path.replace('.csv', '_161128_171129.csv').replace('/enriched', '')
            df = pd.read_csv(primary_path)
            df = df[df['DATE'] == self.get_next_date(single_advice['DATE'].values[0])]
            df_temp = df[df['TIME'].isin([i for i in range(0, 123100)])]
            was_bought = single_advice['buy_at'].values[0] > df_temp['LOW'].min()
            if not was_bought:
                single_advice['RESULT'] = [0]
                single_advice_list.append(single_advice)
                continue
            df_temp = df[df['TIME'].isin([i for i in range(123100, 241000)])]
            sold_properly = single_advice['than_sell_at'].values[0] < df_temp['HIGH'].max()
            if sold_properly:
                single_advice['RESULT'] = [(single_advice['than_sell_at'].values[0] - single_advice['buy_at'].values[0])
                                           / single_advice['buy_at'].values[0]]
                single_advice_list.append(single_advice)
                continue
            df_temp = df[df['TIME'].isin([i for i in range(184000, 191000)])]
            single_advice['RESULT'] = [(df_temp['HIGH'].mean() - single_advice['buy_at'].values[0])
                                       / single_advice['buy_at'].values[0]]
            single_advice_list.append(single_advice)

        advice = pd.concat(single_advice_list)
        return advice

    def get_next_date(self, date: 'date to be raised by 1'):
        new_date = int(date) + 1
        if new_date in incorrect_dates:
            new_date = correct_dates[new_date]
        return new_date

    def give_trading_advise(self, predict_data: 'df or path to df'):
        """
        Совет по цене покупки и цене последующей продажи на основании прогноза следующего дня.
        :param predict_data: Дата фрейм с прогнозом следующего дня.
        :return:
        """
        if type(predict_data) is pd.DataFrame:
            df = predict_data
        elif type(predict_data) is str:
            df = pd.read_csv(predict_data + 'tomorrow_predictions.csv')
        else:
            raise Exception('Allowed only str path to csv or pandas.DataFrame. Given %s' % type(predict_data))

        df['buy_at'] = df['qtr_mean_0'].copy()
        df['growth'] = df['qtr_mean_1'] - df['qtr_mean_0']
        df['growth'][df['growth'] < -0.001] = np.nan
        df['buy_at'] = df['buy_at'] - df['day_std'] * 1.4
        df['buy_at'][df['buy_at'] > 0] = 0
        df['than_sell_at'] = df['buy_at']
        df['than_sell_at'][df['growth'] > 0] = df['buy_at'] + df['day_std'] * 1.5
        df['than_sell_at'][df['growth'] < 0] = df['buy_at'] + df['day_std']

        df['recommended'] = (df['than_sell_at'] - df['buy_at'])
        # df['recommended'][df['recommended'] != df['recommended'].max()] = np.nan
        df = df.dropna(how='any').drop(['recommended'], axis=1)
        return df

# Analyzer().emulate_trades(from_path='D:/TEMP/data_v2/results/')