"""
В файле представлены функции, имеющее стандартное апи: (df, skip_columns=())
Они все - разные стратегии преобразования исходных минутных временных рядов к некоторому формату,
который можно пускать в обработку - это либо показатели за день, либо по часам, либо ещё как - не столь важно.
Главное - это то, что входной формат данных здесь полностью преобразуется к новому, удобному для алгоритмов МL.
"""

import time
import pandas as pd
from v2.utils import weighted_avg_and_std

day_parts_borders_dict = {
    2: (0, 1401, 2410),
    3: (0, 1301, 1601, 2410),
    4: (0, 1221, 1441, 1701, 2410),
    5: (0, 1201, 1401, 1601, 1801, 2410),
    6: (0, 1131, 1301, 1431, 1601, 1731, 2410),
    9: (0, 1101, 1201, 1301, 1401, 1501, 1601, 1701, 1801, 2410)
}

end_parts_borders_dict = {
    1: (1841, 1901),
    2: (1601, 1731, 2410),
    3: (1431, 1601, 1731, 2410)
}


def mock(df, skip_columns=()):
    return df


def extract_daily_data(df, skip_columns=()):
    """
    Преобразование минутного ряда к ряду дней с обогащением признаками о дневной динамике.
    -----------------
    Для оптимизации отталкиваемся от факта, что все данные уже отсортированы по времени!
    """
    start_time = time.time()
    day_parts_amount, end_parts_amount = 2, 2
    df['CLOSE'] = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].mean(axis=1)

    all_dates = sorted(df['DATE'].value_counts().index)
    new_df = pd.DataFrame(data=all_dates, columns=['DATE'])

    # Здесь перечислим создаваемые колонки нового датафрейма и затем в цикле их наполним
    day_mean_w, day_std_w, day_low_w, day_high_w, day_min, day_max = [], [], [], [], [], []

    day_parts_borders = day_parts_borders_dict[day_parts_amount]
    end_parts_borders = end_parts_borders_dict[end_parts_amount]
    last_part_borders = end_parts_borders_dict[1]

    day_parts_mean, day_parts_low, day_parts_high, day_parts_std = \
        ([[] for i in range(day_parts_amount)] for j in range(4))

    end_parts_mean, end_parts_low, end_parts_high, end_parts_std = \
        ([[] for i in range(end_parts_amount)] for j in range(4))

    last_part_mean = []

    # Средняя взвешенная по объёмам, взвешенное отклонение (сигма)
    for date in all_dates:
        temp_df = df[df['DATE'] == date]

        mean, std = \
            weighted_avg_and_std(values=temp_df['CLOSE'], weights=temp_df['VOL'])
        # Absolute values
        mean_low, std_low = \
            weighted_avg_and_std(values=temp_df['LOW'], weights=temp_df['VOL'])

        # Преобразуем среднеквадратичное отклонение в относительную величину,
        # чтобы для всех активов эта величина была измерена в одной шкале.
        std = std / mean
        day_mean_w.append(mean)
        day_min.append(temp_df['LOW'].min())
        day_max.append(temp_df['HIGH'].max())
        day_std_w.append(std)
        day_low_w.append(mean_low)

        # Показатели по долям дня
        for i in range(len(day_parts_borders) - 1):
            from_time, to_time = day_parts_borders[i], day_parts_borders[i + 1]

            sub_temp_df = temp_df[~temp_df['TIME'].isin([i * 100 for i in range(from_time, to_time)])]
            prices_, low_, vol_ = \
                sub_temp_df['CLOSE'], sub_temp_df['LOW'], sub_temp_df['VOL']
            if len(prices_) == 0 | len(vol_) == 0:
                day_parts_mean[i].append(mean)
                day_parts_low[i].append(mean_low)
                day_parts_std[i].append(std)
                continue
            q_mean, q_std = \
                weighted_avg_and_std(values=prices_, weights=vol_)
            q_low_mean, q_low_std = \
                weighted_avg_and_std(values=low_, weights=vol_)
            # Преобразуем среднеквадратичное отклонение в относительную величину,
            # чтобы для всех активов эта величина была измерена в одной шкале.
            q_std = q_std / q_mean
            q_std = 0.0001 if q_std < 0.0001 else q_std
            day_parts_mean[i].append(q_mean)
            day_parts_low[i].append(q_low_mean)
            day_parts_std[i].append(q_std)

        # Показатели по долям конца дня
        for i in range(len(end_parts_borders) - 1):
            from_time, to_time = end_parts_borders[i], end_parts_borders[i + 1]
            sub_temp_df = temp_df[~temp_df['TIME'].isin([i * 100 for i in range(from_time, to_time)])]
            prices_, low_, vol_ = \
                sub_temp_df['CLOSE'], sub_temp_df['LOW'], sub_temp_df['VOL']
            if len(prices_) == 0 | len(vol_) == 0:
                end_parts_mean[i].append(mean)
                end_parts_low[i].append(mean_low)
                end_parts_std[i].append(std)
                continue
            end_mean, end_std = \
                weighted_avg_and_std(values=prices_, weights=vol_)
            end_low_mean, end_low_std = \
                weighted_avg_and_std(values=low_, weights=vol_)
            # Преобразуем среднеквадратичное отклонение в относительную величину,
            # чтобы для всех активов эти величины была измерена в одной шкале.
            end_std = end_std / end_mean
            end_mean = (end_mean - mean) / mean
            end_std = 0.0001 if end_std < 0.0001 else end_std
            end_parts_mean[i].append(end_mean)
            end_parts_low[i].append(end_low_mean)
            end_parts_std[i].append(end_std)

        # CLOSE in interval 10 minutes
        from_time, to_time = last_part_borders[0], last_part_borders[1]
        sub_temp_df = temp_df[~temp_df['TIME'].isin([i * 100 for i in range(from_time, to_time)])]
        prices_, vol_ = sub_temp_df['CLOSE'], sub_temp_df['VOL']
        if len(prices_) == 0 | len(vol_) == 0:
            last_part_mean.append(mean)
            continue
        last_mean, last_part_std = \
            weighted_avg_and_std(values=prices_, weights=vol_)
        last_part_mean.append(last_mean)

    new_df['day_mean'] = day_mean_w
    new_df['last_part_mean'] = last_part_mean
    new_df['day_min'] = day_min
    new_df['day_max'] = day_max
    new_df['day_max_min_growth'] = (new_df['day_max'] - new_df['day_min']) / new_df['day_min']
    new_df = new_df.drop(['day_min', 'day_max'], axis=1)
    new_df['day_low'] = day_low_w
    new_df['day_std'] = day_std_w

    for i in range(day_parts_amount):
        new_df['qtr_mean_%d' % i] = day_parts_mean[i]
        new_df['qtr_low_%d' % i] = day_parts_low[i]
        new_df['qtr_std_%d' % i] = day_parts_std[i]

    for i in range(end_parts_amount):
        new_df['end_parts_mean_%d' % i] = end_parts_mean[i]
        new_df['end_parts_low_%d' % i] = end_parts_low[i]
        new_df['end_parts_std_%d' % i] = end_parts_std[i]

    print('Single Data Frame transformation took %d seconds' % (time.time() - start_time))
    return new_df
