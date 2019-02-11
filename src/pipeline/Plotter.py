import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

import matplotlib.pyplot as plt
import pandas as pd

class Plotter():
    def __init__(self):
        self.name = None

    def plot_temperature(self, temp_dataframe):
        temp_dataframe.plot(style=['r-', 'b--'])
        plt.savefig('Temp.png')

if __name__ == '__main__':
    pass


