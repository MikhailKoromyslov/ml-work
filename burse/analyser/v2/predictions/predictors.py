from v2.prepare import Pipeline
from v2.strategies.extractions import extract_daily_data


def fast_predict(list_of_y=('last_part_mean')):
    pipe = Pipeline(in_memory=True, initial_path='D:/TEMP/data_v2/patch_from/')
    pipe.transformation(from_path=None,
                        to_path=None,
                        exec_func=extract_daily_data) \
        .fit_by_dates(from_path=None,
                      to_path=None) \
        .transformation(from_path=None,
                        to_path=None,
                        exec_func=pipe.generate_new_columns,
                        skip_columns=('day_std', 'qtr_std', 'end_parts', '_low')) \
        .transformation(from_path=None,
                        to_path=None,
                        exec_func=pipe.convert_to_diffs,
                        skip_columns=('day_std', 'qtr_std', 'qtr_mean', 'date_diff', 'end_parts', 'diff_to_prev_last',
                                      '_growth', '_low', 'day_max_min_growth'))\
        .prepare_separate_train_tables(from_path='D:/TEMP/data_v2/diff/',
                                       to_path='D:/TEMP/data_v2/final/',
                                       need_rewrite=False) \
        .prepare_train_and_test(from_path='D:/TEMP/data_v2/final/',
                                to_path='D:/TEMP/data_v2/train_test/',
                                n_test=175,
                                need_rewrite=False)
