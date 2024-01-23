''' 
this file is used to :
1) extract symbols from tickers.txt
2) extract data from yahoo finance using client()
3) save data in a csv file
'''


import os
from client import PriceHistory
from fake_useragent import UserAgent

FICHIER = 'tickers.txt'
chemin_complet = os.path.join('DataProvider', FICHIER)
# 1) extract symbols from tickers.txt
symbols = []
with open(chemin_complet, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        x = line.split()
        symbols.append(x)

price_history_client = PriceHistory(symbols[0], user_agent=UserAgent().edge)

price_history_client.dataframe_prix.to_csv('DataProvider/stock_datat.csv', index=False)
