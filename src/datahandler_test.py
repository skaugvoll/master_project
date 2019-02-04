import os, sys
from pipeline.DataHandler import DataHandler


dh = DataHandler()
print('created datahandler')

# dh.load_dataframe_from_7z(
#     input_arhcive_path=os.path.join(os.getcwd(),'../data/input/testSubject08.7z'),
# )

# print("Starting cleanup....")
# dh.cleanup_temp_folder()

dh.read_and_return_multiple_csv_iterators(
    dir_path='./models/',
    filenames=['rfc'],
    format='py'
)






