import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src import models





input_dir_rel_path = "/app/data/input"
data_name = "Sigve.7z"
label_file = "Sigve intervals.json"

pipeline = Pipeline()

###########
#
# IF first time running script on data, else it is saved in ../data/temp/name
#
##########


# if there allready is a temp folder with the same name
# TODO get this in the unzip N Synch method, path is unzip_path + filename.7z
# if os.path.exists("../data/temp/{}".format(data_name)):
#     print("REMVOING OLD TEMP FOLDER")
#     os.system("rm -rf ../data/temp/{}".format(data_name))
#
#
# # first unzip and synch .7z folder
# datahandler = pipeline.unzipNsynch(os.path.join(input_dir_rel_path, data_name), save=True) # returns datahandler
# unzipped_path = datahandler.get_unzipped_path()
#
# pipeline.addLables(intervals="../data/temp/{}/{}/{}".format(
#     data_name, data_name.split(".")[0], label_file), column_name="label")
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
print()
print(df.dtypes)
#
# ########
# #
# # DATA INPUT FORMAT SPECIFIC DONE
# #
# #######
#
# # Do some magic numbering
sampling_frequency = 50
window_length = 120
tempearture_reading_rate = 120
samples_pr_second = 1/(tempearture_reading_rate/sampling_frequency)
samples_pr_window = int(window_length*samples_pr_second)
train_overlap = .8
number_of_trees_in_forest=100


# extract the features
back, thigh, labels = pipeline.get_features_and_labels_as_np_array(df)



# Get the model
RFC = models.get("RFC", {})

####
# Train the model
####
RFC.train(
    back_training_feat=back,
    thigh_training_feat=thigh,
    labels=labels,
    samples_pr_window=samples_pr_window,
    train_overlap=train_overlap,
    number_of_trees=number_of_trees_in_forest
)


#####
# TEST THE MODEL
####
dh = DataHandler()
sub_name = data_name.split(".")[0]
input_dir = "../data/temp/Sigve.7z/Sigve/"
filename = "25226_B_25226_T_timesync_output_TEMP_BT.csv"

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
intervals = dh.read_labels_from_json(filepath="../data/temp/Sigve.7z/Sigve/Sigve intervals.json")
dh.add_labels_file_based_on_intervals(intervals=intervals)
df = dh.get_dataframe_iterator()
print(df.head(10))
print()
print(df.dtypes)

back, thigh, labels = pipeline.get_features_and_labels_as_np_array(df)

RFC.test(back, thigh, labels, samples_pr_window, train_overlap)

acc = RFC.calculate_accuracy()
print("ACC: ", acc)