from datetime import datetime
import requests
import pandas as pd
import yesg

import numpy as np
import yfinance as yf
from pandas_datareader import data as wb

from dateutil.relativedelta import *

from tqdm import tqdm

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

        self.dataframe_esg = self._build_esg_dataframes()


    def symbols(self):
        return self._symbols

    def _build_dataframes(self):
        data = pd.DataFrame()

        to_date = datetime.today().date()

        from_date = to_date - relativedelta(years=10)

        for symbol in tqdm(self._symbols):
            yf.pdr_override()
            data_index = pd.DataFrame()
            data_index = wb.get_data_yahoo(symbol,start=from_date, end=to_date, interval='1d')
            data_index['symbol'] = symbol

            data_index['date'] = data_index.index
            
            # concaténer les dataframes
            data = pd.concat([data, data_index])

        data = data[data.columns[::-1]]
        return data


    def _build_esg_dataframes(self):
        data = pd.DataFrame()

        for symbol in tqdm(self._symbols):
            yf.pdr_override()
            # create a dataframe with the symbol
            esg_data = pd.DataFrame()
            esg_data['symbol'] = [symbol]
            
            try:
                esg_score = yesg.get_historic_esg(symbol).iloc[-1]['Total-Score']
                esg_data['esg_score'] = esg_score
            except:
                esg_data['esg_score'] = "NaN"

            data = pd.concat([data, esg_data])
        
        data = data[['symbol', 'esg_score']]
        return data
