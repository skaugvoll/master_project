import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from pipeline.DataHandler import DataHandler
import cwa_converter
import matplotlib.pyplot as plt
import pandas as pd

#
# os.system("rm -rf ../data/temp/4000181.7z/")
# Create a data handling object for importing and manipulating dataset ## PREPROCESSING
print('CREATING datahandlerS')
dh1 = DataHandler()
dh2 = DataHandler()

print('CREATED datahandlerS')


