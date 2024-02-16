import numpy as np
import pandas

class Stock:
    def __init__(self, epic, data):
        self.epic = epic 
        self.dates = dict(zip(np.arange(data.shape[0]), data.index.values))
        self.prices = data.values
    
