"""
Defining the asset class.
"""

from cmath import nan

import numpy as np


class Asset:
    def __init__(self) -> None:
        self.name = "None"
        self.crisisYield = 0.0
        self.ethicGrade = 0.0
        self.dailyPrices = []
        self.numPrices = 0 # number of days of asset prices available

    def __repr__(self):
        return self.name + ": , Number of days of asset prices : " + str(self.numPrices)
