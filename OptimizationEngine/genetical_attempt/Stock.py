import numpy as np
import pandas

class Stock():
    def __init__(self, epic, data):
        self.epic = epic 
        self.dates = dict(zip(np.arange(data.shape[0]), data['Date'].values))
        data.drop('Date', axis=1, inplace=True)
        self.prices = data.to_dict().values()[0]
    
