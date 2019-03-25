import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src import models





input_dir_rel_path = "/app/data/input"
data_name = "Thomas.7z"
label_file = "Thomas intervals.json"

pipeline = Pipeline()

###########
#
# IF first time running script on data, else it is saved in ../data/temp/name
#
##########

#
# # if there allready is a temp folder with the same name
# # TODO get this in the unzip N Synch method, path is unzip_path + filename.7z
# if os.path.exists("../data/temp/{}".format(data_name)):
#     print("REMVOING OLD TEMP FOLDER")
#     os.system("rm -rf ../data/temp/{}".format(data_name))
#
#
# # first unzip and synch .7z folder
# datahandler = pipeline.unzipNsynch(os.path.join(input_dir_rel_path, data_name), save=True) # returns datahandler
# unzipped_path = datahandler.get_unzipped_path()
#
# pipeline.addLables(intervals="../data/temp/Thomas.7z/Thomas/Thomas intervals.json", column_name="label")
# dataframe = pipeline.dh.get_dataframe_iterator()
# print(dataframe.head(10))

###########
#
# IF data is csv file
#
##########

dh = DataHandler()
sub_name = data_name.split(".")[0]
input_dir = "../data/temp/Thomas.7z/Thomas/"
filename = "25228_B_25228_T_timesync_output_TEMP_BT.csv"

dh.load_dataframe_from_csv(
    input_directory_path=input_dir,
    filename=filename,
    header=0,
    columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz', 'btemp', 'ttemp']
)

dh.convert_column_from_str_to_datetime(column_name='time')
dh.set_column_as_index("time")

# # add labels
dh.add_new_column("label")
intervals = dh.read_labels_from_json(filepath="../data/temp/Thomas.7z/Thomas/Thomas intervals.json")
dh.add_labels_file_based_on_intervals(intervals=intervals)
df = dh.get_dataframe_iterator()
print(df.head(10))


########
#
# DATA INPUT FORMAT SPECIFIC DONE
#
#######








# Get the model
# rfc = models.get("RFC", {})
