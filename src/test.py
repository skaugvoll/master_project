import os, sys
from pipeline.DataHandler import DataHandler


dh = DataHandler()
print('created datahandler')

dh.load_dataframe_from_7z(
    input_arhcive_path=os.path.join(os.getcwd(),'../data/input/testSubject08.7z'),
)



