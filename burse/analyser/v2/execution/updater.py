from v2.utils import patch_files
from v2.prepare import Pipeline
from v2.strategies.extractions import extract_daily_data

class Updater:
    """
    Предназначен для обновления имеющихся данных склеиванием к ним новых - СТРОГО ПОСЛЕ этапа прогноза!!!
    """

    def __init__(self, all_data_path, new_data_path):
        """
        Новые данные скачаны только для тех активов, по которым у нас есть история в all_data_path.
        По этой причине просто патчим наши старые данные новыми, после чего прогоняем всю цепочку трансформаций.
        """
        self.all_data_path = all_data_path
        self.new_data_path = new_data_path

    def update(self):
        # Сначала пропатчим файлы исходных данных свежескаченными
        patch_files(self.new_data_path, self.all_data_path)

        # Затем прогоним серию трансформаций, чтобы обновить все промежуточные сохранения
        pipe = Pipeline()
        pipe.transformation(from_path='D:/TEMP/data_v2/',
                            to_path='D:/TEMP/data_v2/transformed/',
                            exec_func=extract_daily_data,
                            need_rewrite=True) \
            .fit_by_dates(from_path='D:/TEMP/data_v2/transformed/',
                          to_path='D:/TEMP/data_v2/transformed/',
                          need_rewrite=True) \
            .transformation(from_path='D:/TEMP/data_v2/transformed/',
                            to_path='D:/TEMP/data_v2/enriched/',
                            exec_func=pipe.generate_new_columns,
                            skip_columns=('day_std', 'qtr_std', 'end_parts', '_low'),
                            need_rewrite=False) \
            .transformation(from_path='D:/TEMP/data_v2/enriched/',
                            to_path='D:/TEMP/data_v2/diff/',
                            exec_func=pipe.convert_to_diffs,
                            skip_columns=(
                                'day_std', 'qtr_std', 'qtr_mean', 'date_diff', 'end_parts', 'diff_to_prev_last',
                                '_growth', '_low', 'day_max_min_growth'),
                            need_rewrite=False)
