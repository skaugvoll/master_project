import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src import models


input_dir_rel_path = "../data/input"
data_name = "Thomas.7z"
labels_file = ""

# if there allready is a temp folder with the same name
# TODO get this in the unzip N Synch method, path is unzip_path + filename.7z
if os.path.exists("../data/temp/{}".format(data_name)):
    print("REMVOING OLD TEMP FOLDER")
    os.system("rm -rf ../data/temp/{}".format(data_name))


# first unzip and synch .7z folder
pipeline = Pipeline()
datahandler = pipeline.unzipNsynch(os.path.join(input_dir_rel_path, data_name), save=True) # returns datahandler

# # add labels
# pipeline.addLables(intervals={}, column_name="label")
#
#
#
# # Get the model
# rfc = models.get("RFC", {})
