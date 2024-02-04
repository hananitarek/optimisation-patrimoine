''' 
this file is used to :
1) extract symbols from tickers.txt
2) extract data from yahoo finance using client()
3) save data in a csv file
'''

import csv
import os
from client_Fr import PriceHistory
from fake_useragent import UserAgent
from pandas_datareader import data as wb

FICHIER = 'tickers_french.txt'
chemin_complet = os.path.join('DataProvider', FICHIER)
# 1) extract symbols from tickers.txt
symbols = []
with open(chemin_complet, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # x = line.split()
        symbols.append(line)

price_history_client = PriceHistory(symbols, user_agent=UserAgent().edge)


price_history_client.dataframe_prix.to_csv('DataProvider/stock_data.csv', index=False)
# price_history_client.dataframe_esg.to_csv('DataProvider/stock_datat_french_esg.csv', index=False)


# FICHIER_IN = 'stock_datat_french.csv'
# FICHIER_OUT = 'stock_datat_french_out.csv'
# chemin_complet_IN = os.path.join('DataProvider', FICHIER_IN)
# chemin_complet_OUT = os.path.join('DataProvider', FICHIER_OUT)
# with open(chemin_complet_IN) as fr, open(chemin_complet_OUT, "w", newline="") as fw:
#    cr = csv.reader(fr, delimiter=",")
#    cw = csv.writer(fw, delimiter=",")
#    cw.writerow(next(cr))
#    cw.writerows(reversed(list(cr)))