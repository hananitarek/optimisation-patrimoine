import requests
import pandas as pd

from typing import List
from typing import Dict
from typing import Union

from datetime import datetime
from datetime import timedelta
from datetime import date

from dateutil.relativedelta import *

from fake_useragent import UserAgent

class PriceHistory():
    """ 
    Cette classe permet de récupérer les données historiques des actions
    du site NASDAQ. 
    """

    def __init__(self, symbols, user_agent):
        self._api_url = "https://api.nasdaq.com/api/quote/"
        self._symbols = symbols
        self._api_service = "historical"

        self.user_agent = user_agent
        self.dataframe_prix = self._build_dataframes()

    def _build_url(self, symbol):
        parts = [self._api_url, symbol, self._api_service]
        return "/".join(parts)

    def symbols(self):
        return self._symbols

    def _build_dataframes(self):
        data = []

        to_date = datetime.today().date()

        from_date = to_date - relativedelta(years=10)

        for symbol in self._symbols:
            data = self._grab_prices(
                symbol = symbol,
                from_date = from_date,
                to_date = to_date
            ) + data

        dataframe_prix = pd.DataFrame(data = data)
        dataframe_prix['date'] = pd.to_datetime(dataframe_prix['date'])

        return dataframe_prix

    def _grab_prices(self, symbol, from_date, to_date):
        price_url = self._build_url(symbol)
        limit = to_date - from_date

        params = {
            'fromdate': from_date.isoformat(),
            'todate': to_date.isoformat(),
            'assetclass': 'stocks',
            'limit': limit.days
        }

        headers = {
            'user-agent': self.user_agent
        }

        historical_data = requests.get(
            url = price_url,
            params = params,
            headers = headers,
            verify = True
        )

        if historical_data.ok:
            historical_data = historical_data.json()
            print(symbol)
            try :
                historical_data = historical_data['data']['tradesTable']['rows']
            except :
                historical_data = []
            

            for table_row in historical_data:
                table_row['symbol'] = symbol
                table_row['close'] = table_row['close'].replace(',', '')
                table_row['close'] = float(table_row['close'].replace('$', ''))

                if table_row['volume'] == 'N/A':
                    table_row['volume'] = 0
                else :
                    table_row['volume'] = int(table_row['volume'].replace(',', ''))

                table_row['open'] = table_row['open'].replace(',', '')
                table_row['open'] = float(table_row['open'].replace('$', ''))

                table_row['high'] = table_row['high'].replace(',', '')
                table_row['high'] = float(table_row['high'].replace('$', ''))

                table_row['low'] = table_row['low'].replace(',', '')
                table_row['low'] = float(table_row['low'].replace('$', ''))
            
            return historical_data