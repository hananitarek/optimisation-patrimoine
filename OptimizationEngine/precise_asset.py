"""
Defining an asset class more precise.
"""

from cmath import nan

import numpy as np


class BetterAsset:
    def __init__(self) -> None:
        self.symbol = "None"
        
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.volume = []
        self.crisisYield = 0.0 
        self.ethicGrade = 0.0 # chosen by the user (randomly)


    def __repr__(self):
        return f"Asset: {self.symbol} - Ethic Grade: {self.ethicGrade} - Crisis Yield: {self.crisisYield}- Open: {self.open} - High: {self.high} - Low: {self.low} - Close: {self.close} - Volume: {self.volume}"