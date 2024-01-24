"""
Defining an asset class more precise.
"""

from cmath import nan

import numpy as np


class BetterAsset:
    def __init__(self) -> None:
        self.symbol = "None"
        
        self.open = []
        self.low = []
        self.DailyPrices = []
        self.crisisYield = 0.0 
        self.ethicGrade = 0.0 # chosen by the user (randomly)
        self.numPrices = 0


    def __repr__(self):
        return f"Asset: {self.symbol} "