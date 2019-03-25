import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src import models


# first unzip and synch .7z folder
pipeline = Pipeline()
datahandler = pipeline.unzipNsynch('../data/input/Thomas2.7z') # returns datahandler

# add labels
pipeline.addLables(intervals={}, column_name="label")

