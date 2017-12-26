import urllib
import pandas as pd 
import numpy as np
import datetime as dt

#%%
"""
далее идут словари предназначенные для корректного формирования запроса. 
ну и так... для наглядности
с хабра:
market, em, code – об этих параметрах, упоминал ранее, при обращении к функции их значения будут приниматься из файла.
df, mf, yf, from, dt, mt, yt, to – это параметры времени.
p — период котировок (тики, 1 мин., 5 мин., 10 мин., 15 мин., 30 мин., 1 час, 1 день, 1 неделя, 1 месяц)
e – расширение получаемого файла; возможны варианты — .txt либо .csv
dtf — формат даты (1 — ггггммдд, 2 — ггммдд, 3 — ддммгг, 4 — дд/мм/гг, 5 — мм/дд/гг)
tmf — формат времени (1 — ччммсс, 2 — ччмм, 3 — чч: мм: сс, 4 — чч: мм)
MSOR — выдавать время (0 — начала свечи, 1 — окончания свечи)
mstimever — выдавать время (НЕ московское — mstimever=0; московское — mstime='on', mstimever='1')
sep — параметр разделитель полей (1 — запятая (,), 2 — точка (.), 3 — точка с запятой (;), 4 — табуляция (»), 5 — пробел ( ))
sep2 — параметр разделитель разрядов (1 — нет, 2 — точка (.), 3 — запятая (,), 4 — пробел ( ), 5 — кавычка ('))
datf — Перечень получаемых данных (#1 — TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL; #2 — TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE; #3 — TICKER, PER, DATE, TIME, CLOSE, VOL; #4 — TICKER, PER, DATE, TIME, CLOSE; #5 — DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL; #6 — DATE, TIME, LAST, VOL, ID, OPER).
at — добавлять заголовок в файл (0 — нет, 1 — да)
"""
#форматы даты
date_formats = {'yyyymmdd':1, 'yymmdd': 2, 'ddmmyy': 3, 'dd/mm/yy':4, 'mm/dd/yy':5} 
#форматы времени
time_formats = {'hhmmss':1, 'hhmm':2, 'hh: mm: ss':3, 'hh: mm':4}
#форматы данных
data_formats = {'TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL':1, 'TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE':2, 'TICKER, PER, DATE, TIME, CLOSE, VOL':3, 'TICKER, PER, DATE, TIME, CLOSE':4, 'DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL':5, 'DATE, TIME, LAST, VOL, ID, OPER':6}
#периодичность записей
time_periods = {'tick':1, 'min':2, '5 min':3, '10 min':4, '15 min':5, '30 min': 6, '1 hour': 7, 'day':8, 'week':9, 'month':10}
#устанавливает значения времени на начало или конец японской свечи (интервала измерения)
time_froms = {'start candlestick': 0, 'end canlestick':1}
#ну тут очевидно
time_zones = {'greenwich':0, 'moscow':1}
#добавлять ли в датасет заголовки
download_headers = {'no':0, 'yes':1}
#разделители записей между столбцами
separators = {',' :1, '.':2, ';':3, '   ':4, ' ':5}
#десятичный разделитель
decimal_marks = {'':1, '.':2, ',':3, ' ':4, '\'':5}


"""
выгружает данные с finam.ru
привязан к файлу loader_info.csv, в котором содержится информация о всех активах, 
данные по которым можно получить с finam'а, о рынках к на которых данными активами торгуют (там еще категории для выборки есть) 
а так же данные необходимые для формирования запроса по каждому из активов (tool_id и market_id)
"""
class Loader:

    def __init__(self):
        self.info_df=pd.read_csv('loader_info.csv', names=['tool_id', 'tool', 'market_id', 'market'])
    
    """
    имена активов и рынков можно посмотреть в https://www.finam.ru/profile/moex-akcii/sakhalinenergo-ao/export/?market=200
    искать можно по имени рынка или актива
    в итоге нужно получить id актива и рынка, использовать в download_quotations
    некоторые имена не соответствуют именам на сайте, в частности такие, в которых
    присутствует символ '&' и возможно еще какие то, в которых есть излишние знаки
    в датафрейме все лишние символы кроме '-' потёрты
    """
    def get_info(self):
        return self.info_df

    def get_info_by_market(self, market):
        return self.info_df[self.info_df['market']==market]

    def get_info_by_tool(self, tool):
        return self.info_df[self.info_df['tool']==tool]

"""
формирует запрос на выгрузку, а так же сохраняет данные в файл 
имя файла формируется из названия актива и периода выгрузки
параметры:
name - название актива (абривиатура или что угодно чем можно идентифицировать актив)
start_date - дата начала выгрузки 
end_date - дата конца выгрузки
market_id - брать из loader_info.csv
tool_id - брать из loader_info.csv
extension - расширение итогового файла, по умолчанию '.csv', есть еще варик выгружать в txt 
period - интервал одного измерения, по умолчанию 1 день 
date_format - формат даты (брать из date_formats)
time_format - формат времени (брать из time_formats)
data_format - формат данных
time_from - время относительно периода отсчета
time_zone - а не московское ли время
headers - нужны ли заголовки в файле
separator - разделитель столбцов
decimal_mark - десятичный разделитель
"""
    def download_quotations(self, 
                            name,
                            start_date, 
                            end_date,
                            market_id, 
                            tool_id, 
                            extension='.csv',
                            period=time_periods['day'], 
                            date_format=date_formats['yyyymmdd'], 
                            time_format=time_formats['hhmmss'], 
                            data_format=data_formats['TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL'], 
                            time_from=time_froms['start candlestick'],
                            time_zone=time_zones['moscow'], 
                            headers=download_headers['yes'],
                            separator=separators[';'],
                            decimal_mark=decimal_marks['.']):
        page = urllib.request.urlopen('http://export.finam.ru/'+str(name)+'_'+start_date.strftime("%Y%m%d")+'_'+end_date.strftime("%Y%m%d")+str(extension)+'?market='+str(market_id)+'&em='+str(tool_id)+'&code='+str(name)+'&apply=0&df='+str(start_date.day)+'&mf='+str(start_date.month-1)+'&yf='+str(start_date.year)+'&from='+str(start_date.strftime("%d.%m.%Y"))+'&dt='+str(end_date.day)+'&mt='+str(end_date.month-1)+'&yt='+str(end_date.year)+'&to='+str(end_date.strftime("%d.%m.%Y"))+'&p='+str(period)+'&f='+str(name)+'_'+str(start_date.strftime("%y%m%d"))+'_'+str(end_date.strftime("%y%m%d"))+'&e='+str(extension)+'&cn='+str(name)+'&dtf='+str(date_format)+'&tmf='+str(time_format)+'&MSOR='+str(time_from)+'&mstimever='+str(time_zone)+'&sep='+str(separator)+'&sep2='+str(decimal_mark)+'&datf='+str(data_format)+'&at='+str(headers))
        f = open(str(name)+'_'+start_date.strftime("%Y%m%d")+'_'+end_date.strftime("%Y%m%d")+str(e), "wb")
        content = page.read()
        print('http://export.finam.ru/'+str(name)+'_'+start_date.strftime("%Y%m%d")+'_'+end_date.strftime("%Y%m%d")+str(extension)+'?market='+str(market_id)+'&em='+str(tool_id)+'&code='+str(name)+'&apply=0&df='+str(start_date.day)+'&mf='+str(start_date.month-1)+'&yf='+str(start_date.year)+'&from='+str(start_date.strftime("%d.%m.%Y"))+'&dt='+str(end_date.day)+'&mt='+str(end_date.month-1)+'&yt='+str(end_date.year)+'&to='+str(end_date.strftime("%d.%m.%Y"))+'&p='+str(period)+'&f='+str(name)+'_'+str(start_date.strftime("%y%m%d"))+'_'+str(end_date.strftime("%y%m%d"))+'&e='+str(extension)+'&cn='+str(name)+'&dtf='+str(date_format)+'&tmf='+str(time_format)+'&MSOR='+str(time_from)+'&mstimever='+str(time_zone)+'&sep='+str(separator)+'&sep2='+str(decimal_mark)+'&datf='+str(data_format)+'&at='+str(headers))
        f.write(content)
        f.close()


#%%
"""
пример использования
соответствует https://www.finam.ru/profile/moex-akcii/seligdar/export/?market=1&em=473000&code=SELG&apply=0&df=10&mf=9&yf=2017&from=10.10.2017&dt=15&mt=10&yt=2017&to=15.11.2017&p=8&f=SELG_171010_171115&e=.csv&cn=SELG&dtf=1&tmf=1&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=2&datf=1&at=1
"""

loader=Loader()
loader.download_quotations('SLEN', dt.date(2017, 10, 10), dt.date(2017, 11, 15), 1, 473000)