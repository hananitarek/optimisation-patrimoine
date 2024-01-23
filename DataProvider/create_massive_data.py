''' 
this file is used to :
1) extract symbols from tickers.txt
2) extract data from yahoo finance using client()
3) save data in a csv file

'''

# import yfinance as yf
# import pandas as pd
# import pathlib
# from pandas_datareader import data as wb
import os
from client import PriceHistory
from fake_useragent import UserAgent
# from pprint import pprint
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import matplotlib.pyplot as plt

fichier = 'tickers.txt'
chemin_complet = os.path.join('DataProvider', fichier)
# 1) extract symbols from tickers.txt
symbols = []
with open(chemin_complet, 'r') as f:
    for line in f:
        line = line.strip()
        x = line.split()
        symbols.append(x)

print(symbols)



price_history_client = PriceHistory(symbols[0], user_agent=UserAgent().edge)

    # On sauvegarde les données dans un fichier csv
price_history_client.dataframe_prix.to_csv(
        'DataProvider/stock_data.csv',
        index=False
    )

