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


# header = 0 because the csv has the first row indicating column names, let pandas
# know that the first row is a header row
dh1.load_dataframe_from_csv(
    input_directory_path='/app/data/temp/testSNTAtle.7z/testSNTAtle/',
    filename='P1_atle_B_TEMP_SYNCHED_BT.csv',
    header=0,
    columns=['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'btemp', 'ttemp']
)

dh1.head_dataframe(5)

##############################

dh2.load_dataframe_from_csv(
    input_directory_path='/app/data/temp/testVegar.7z/testVegar/',
    filename='P1_vegar_B_TEMP_SYNCHED_BT.csv',
    header=0,
    columns=['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'btemp', 'ttemp']
)

dh2.tail_dataframe(5)

########### VERTICAL STACKING

union = dh1.vertical_stack_dataframes(dh1.get_dataframe_iterator(), dh2.get_dataframe_iterator(), set_as_current_df=False)
print(union.head(5))
print(union.tail(5))

print(union.shape)