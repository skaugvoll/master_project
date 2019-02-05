import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from pipeline.DataHandler import DataHandler
from src import models
import cwa_converter



# Create a data handling object for importing and manipulating dataset ## PREPROCESSING
print('CREATING datahandler')
dh = DataHandler()
print('CREATED datahandler')

##########################
#
#
##########################


# unzip cwas from 7z arhcive
unzipped_path = dh.unzip_7z_archive(
    filepath=os.path.join(os.getcwd(), '../data/input/testSNTAtle.7z'),
    unzip_to_path='../data/temp',
    cleanup=False
)

print('UNZIPPED PATH RETURNED', unzipped_path)

##########################
#
#
##########################

# convert the cwas to independent csv containing timestamp xyz and temp
back_csv, thigh_csv = cwa_converter.convert_cwas_to_csv_with_temp(
    subject_dir=unzipped_path,
    out_dir=unzipped_path,
    paralell=True
)

##########################
#
#
##########################

# Timesynch and concate csvs
dh.merge_csvs_on_first_time_overlap(
    master_csv_path=back_csv,
    slave_csv_path=thigh_csv,
    rearrange_columns_to=[
        'time',
        'bx',
        'by',
        'bz',
        'tx',
        'ty',
        'tz',
        'btemp',
        'ttemp'
    ]
)

df = dh.get_dataframe_iterator()
print(df.head(5))
input("looks ok ? \n")


##########################
#
#
##########################

dh.convert_ADC_temp_to_C(
    dataframe=df,
    dataframe_path=None,
    normalize=False,
    save=True
)

df = dh.get_dataframe_iterator()
print(df.head(5))
input("looks ok ? \n")

##########################
#
#
##########################


print('SET INDEX TO TIMESTAMP')
#test that this works with a dataframe and not only path to csv
# thus pre-loaded and makes it run a little faster
dh.convert_column_from_str_to_datetime_test(
        dataframe='/app/data/temp/testSNTAtle.7z/testSNTAtle/P1_atle_B_TEMP_SYNCHED_BT.csv',
)

dh.set_column_as_index("time")
print('DONE')

##########################
#
#
##########################


print('MAKE NUMERIC')
dh.convert_column_from_str_to_numeric(column_name="btemp")

dh.convert_column_from_str_to_numeric(column_name="ttemp")
print('DONE')

##########################
#
#
##########################


print('ADDING LABELS')
dh.add_new_column()
print('DONE')

dh.add_labels_file_based_on_intervals(
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
        '2' : [
            [
                '2018-04-27',
                '11:09:01',
                '12:19:00'
            ]
        ],
        '3' : [
            [
                '2018-04-27',
                '12:19:01',
                '14:28:00'
            ]
        ]
    },
    label_mapping={"1":"Both", "2": "Thigh", "3": "Back"}
)


##########################
#
#
##########################

dh.show_dataframe()
df = dh.get_dataframe_iterator()

##########################
#
#
##########################

print("ADDING label 2 to rest data")
df['label'] = df.loc["2018-04-30 09:14:57", :] = 2
print("DONE...")


save = input("SAVE with added label 2 to NaN entitites? y | n\n")
if save == "y":

    print("SAVING TO ./test_dataframe.csv")
    try:
        dh.save_dataframe_to_path(path="./test_dataframe.csv", dataframe=df)
        print("DONE SAVING TO ./test_dataframe.csv")
    except Exception as e:
        print(e)
        print("FAILED SAVING TO ./test_dataframe.csv")

##########################
#
#
##########################


# Get the model
model = models.get( "RFC", {} )

# split features
print("1")
back_feat = dh.get_rows_and_columns(dataframe=df, columns=[0,1,2,6]).values
# print(back_feat.head(2))
print("2")
thigh_feat = dh.get_rows_and_columns(dataframe=df, columns=[3,4,5,7]).values
# print(thigh_feat.head(2))
print("3")
# print(df.head(2))
labels = dh.get_rows_and_columns(dataframe=df, columns=[8]).values
# print(labels.head(2))
print("4")

##########################
#
#
##########################

# Do some magic numbering
sampling_frequency = 50
window_length = 120
tempearture_reading_rate = 120
samples_pr_second = 1/(tempearture_reading_rate/sampling_frequency)
samples_pr_window = int(window_length*samples_pr_second)

##########################
#
#
##########################

# pass to the model
model.train(back_training_feat=back_feat,
            thigh_training_feat=thigh_feat,
            labels=labels,
            samples_pr_window=samples_pr_window,
            train_overlap=0.8)


##########################
#
#
##########################