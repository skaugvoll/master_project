import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from pipeline.Pipeline import Pipeline
from pipeline.DataHandler import DataHandler
from src import models


################################ First we need to get the training data! #################################

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

dh1.convert_column_from_str_to_datetime_test(column_name='timestamp')
dh1.set_column_as_index("timestamp")

##############################

dh2.load_dataframe_from_csv(
    input_directory_path='/app/data/temp/testVegar.7z/testVegar/',
    filename='P1_vegar_B_TEMP_SYNCHED_BT.csv',
    header=0,
    columns=['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'btemp', 'ttemp']
)

dh2.convert_column_from_str_to_datetime_test(column_name='timestamp')
dh2.set_column_as_index("timestamp")

################################################ ADD LABELS  TO THE DATAFRAMES ##########################

dh1.add_new_column()
dh1.add_labels_file_based_on_intervals(
    intervals={
        "1": [
            [
               '2018-04-27',
               '10:03:37',
               '10:03:38'
            ],
            [
               '2018-04-27',
               '10:03:39',
               '11:09:00'
            ]
        ],
        '2': [
            [
               '2018-04-27',
               '11:09:01',
               '12:19:00'
            ]
        ],
        '3': [
            [
               '2018-04-27',
               '12:19:01',
               '14:28:00'
            ]
        ]
    }
)

dh2.add_new_column()
dh2.add_labels_file_based_on_intervals(
    intervals={
        "1": [
           [
               '2018-04-24',
               '12:09:00',
               '13:08:00'
           ]
       ],
       '2': [
           [
               '2018-04-24',
               '13:08:01',
               '14:08:00'
           ]
       ],
       '3': [
           [
               '2018-04-24',
               '14:08:01',
               '15:08:00'
           ]
       ]
    }
)




###################################### remove rows that does not have label ###########################

df1 = dh1.get_dataframe_iterator()
df2 = dh2.get_dataframe_iterator()


print(df1.shape, df2.shape)
df1.dropna(subset=['label'], inplace=True)
df2.dropna(subset=['label'], inplace=True)
print(df1.shape, df2.shape)
input("HOW MUCH WAS DROPPED ? ")

############################## THEN COMBINE INTO ONE BIG TRAINING SET  AKA VERTICAL STACKING #############

dataframe = dh1.vertical_stack_dataframes(df1, df2, set_as_current_df=False)
print("DATAFRAME\n", dataframe.head(5), dataframe.shape)
input("LOOKS GOOD ?")

############################## THEN WE MUST EXTRACT FEATURES N LABELS ######################################

pipeObj = Pipeline()
back_feat_train, thigh_feat_train, label_train = pipeObj.get_features_and_labels(dataframe)

############################## THEN WE MUST TRAIN THE CLASSIFIER ######################################

RFC = models.get("RFC", {})

# Do some magic numbering
sampling_frequency = 50
window_length = 120
tempearture_reading_rate = 120
samples_pr_second = 1/(tempearture_reading_rate/sampling_frequency)
samples_pr_window = int(window_length*samples_pr_second)

# pass to the model, for training
RFC.train(back_training_feat=back_feat_train,
        thigh_training_feat=thigh_feat_train,
        labels=label_train,
        samples_pr_window=samples_pr_window,
        train_overlap=0.8,
        number_of_trees=100
        )


#########################################################
##
# TESTING
##
#########################################################


