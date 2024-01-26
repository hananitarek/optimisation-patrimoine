''' 
this file is used to :
1) extract symbols from tickers.txt
2) extract data from yahoo finance using client()
3) save data in a csv file
'''

import csv
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

price_history_client.dataframe_prix.to_csv('DataProvider/stock_datat2.csv', index=False)

FICHIER_IN = 'stock_data.csv'
FICHIER_OUT = 'stock_data_out.csv'
chemin_complet_IN = os.path.join('DataProvider', FICHIER_IN)
chemin_complet_OUT = os.path.join('DataProvider', FICHIER_OUT)
with open(chemin_complet_IN) as fr, open(chemin_complet_OUT, "w", newline="") as fw:
   cr = csv.reader(fr, delimiter=",")
   cw = csv.writer(fw, delimiter=",")
   cw.writerow(next(cr))
   cw.writerows(reversed(list(cr)))