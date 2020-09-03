import numpy as np


class Regression:
    def __init__(self, n):
        self.n = n


    def ReadData(filename, p): #p = number of features in you polynomial
        #Lese
        #Produsere Design matrise

        for i in range p:
            x = 2
