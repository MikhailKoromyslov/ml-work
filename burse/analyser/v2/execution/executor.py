import pandas as pd
from v2.execution.updater import Updater
from v2.loader.loader import Loader
from v2.predictions.predictors import fast_predict
import datetime as dt
import time
from time import mktime
from v2.utils import tickers_unique, union_by_tickers, get_ticker_to_path_dict_for, \
    subtract_one_month, clean_dir, set_standard_header_row


# Запустить, если нужно первично объединить файлы за разные года в общие по тикерам
# union_by_tickers('D:/TEMP/data_v2/primary/', 'D:/TEMP/data_v2/')


def execute_strategy():
    """
    ВНИМАНИЕ!!!
    На момент начала работы папка, в которую должны сохраняться новые данные, не должна быть открыта в ОС!!!
    """
    all_data_path, new_data_path = 'D:/TEMP/data_v2/', 'D:/TEMP/data_v2/patch_from/'
    clean_dir(new_data_path)
    if not tickers_unique(all_data_path):
        raise Exception('На один актив разрешён только один csv-файл с данными! '
                        'Первичная предобработка должна быть проведена единожды и до вызова текущей функции!')

    # Узнаем, по каким активам у нас уже есть данные, чтобы запросить обновление именно по этим активам.
    actual_tickers = list(get_ticker_to_path_dict_for(all_data_path).keys())

    # union_by_tickers('D:/TEMP/', 'D:/TEMP/d2/')

    # Скачаем новые данные за последние 30 календарных дней.
    current_date = dt.datetime.now().date()
    start_date = dt.datetime(year=current_date.year-6, month=current_date.month, day=current_date.day-1)
    end_date = dt.datetime(year=current_date.year-6, month=current_date.month, day=current_date.day)
    loader = Loader()
    for ticker in actual_tickers:
        loader.download_quotations_for_name_and_period(name=ticker,
                                                       new_data_path=new_data_path,
                                                       start_date=start_date,
                                                       end_date=end_date,
                                                       period=7)



    # Приведём первую строку с именами к требуемому формату
    # set_standard_header_row(new_data_path, 'TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL')
    #
    # print('STARTED fast prediction!')
    # fast_predict()
    # print('FINISHED fast prediction!')
    #
    # # Начало этапа обновления наших данных теми, которые были скачаны
    # print('STARTED full update!')
    # Updater(all_data_path=all_data_path,
    #         new_data_path=new_data_path).update()
    # print('FINISHED full update!')

execute_strategy()