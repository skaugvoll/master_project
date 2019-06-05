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
dh_003 = DataHandler()
dh_005 = DataHandler()
dh_003_2 = DataHandler()
dh_005_2 = DataHandler()
dh_combined = DataHandler()

print('CREATED datahandlerS')

##############################
# ATLE P1
##############################

# header = 0 because the csv has the first row indicating column names, let pandas
# know that the first row is a header row
dh_003.load_dataframe_from_csv(
    input_directory_path='/app/data/temp/testSNTAtle.7z/testSNTAtle/',
    filename='P1_atle_B_TEMP_SYNCHED_BT.csv',
    header=0,
    columns=['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'btemp', 'ttemp']
)

dh_003.set_column_as_index("timestamp")

dh_003.add_new_column()
dh_003.add_labels_file_based_on_intervals(
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
dh_003.get_dataframe_iterator().dropna(subset=['label'], inplace=True)
dh_003.head_dataframe(10)


##############################
# ATLE P2
##############################

dh_003_2.load_dataframe_from_csv(
    input_directory_path='/app/data/temp/xxx_x.7z/xxx_x/',
    filename='xxx_x_B_TEMP_SYNCHED_BT.csv',
    header=0,
    columns=['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'btemp', 'ttemp']
)

dh_003_2.set_column_as_index("timestamp")

dh_003_2.add_new_column()
dh_003_2.add_labels_file_based_on_intervals(
    intervals={
        "1": [
            [
                '2018-04-25',
                '09:36:00',
                '10:35:00'
            ],
            [
                '2018-04-25',
                '11:37:01',
                '12:38:00'
            ],
            [
                '2018-04-25',
                '13:42:01',
                '14:33:00'
            ]

        ],
        "2": [
            [
                '2018-04-25',
                '10:35:01',
                '11:37:00'
            ],
        ],
        "3": [
            [
                '2018-04-25',
                '12:38:01',
                '13:42:00'
            ]
        ]
    }
)

dh_003_2.get_dataframe_iterator().dropna(subset=['label'], inplace=True)
dh_003_2.head_dataframe(10)


##############################
# VEGAR P1
##############################

dh_005.load_dataframe_from_csv(
    input_directory_path='/app/data/temp/testVegar.7z/testVegar/',
    filename='P1_vegar_B_TEMP_SYNCHED_BT.csv',
    header=0,
    columns=['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'btemp', 'ttemp']
)

dh_005.set_column_as_index("timestamp")

dh_005.add_new_column()
dh_005.add_labels_file_based_on_intervals(
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

dh_005.get_dataframe_iterator().dropna(subset=['label'], inplace=True)
dh_005.head_dataframe(10)


##############################
# VEGAR P2
##############################

dh_005_2.load_dataframe_from_csv(
    input_directory_path='/app/data/temp/xxx_x.7z/xxx_x/',
    filename='xxx_x_B_TEMP_SYNCHED_BT.csv',
    header=0,
    columns=['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'btemp', 'ttemp']
)

dh_005_2.set_column_as_index("timestamp")

dh_005_2.add_new_column()
dh_005_2.add_labels_file_based_on_intervals(
    intervals={
        "1": [
            [
                '2018-04-26',
                '10:53:00',
                '11:53:00',
            ],
            [
                '2018-04-26',
                '12:53:01',
                '13:53:00',
            ],
            [
                '2018-04-26',
                '14:58:01',
                '15:34:00',
            ]
        ],
        "2": [
            [
                '2018-04-26',
                '11:53:00',
                '12:53:00',
            ],
        ],
        "3": [
            [
                '2018-04-26',
                '13:53:01',
                '14:58:00',
            ]
        ]
    }
)

dh_005_2.get_dataframe_iterator().dropna(subset=['label'], inplace=True)
dh_005_2.head_dataframe(10)


########### VERTICAL STACKING

union_p1_df = dh_combined.vertical_stack_dataframes(dh_003.get_dataframe_iterator(), dh_005.get_dataframe_iterator(), set_as_current_df=False)
print(union_p1_df.head(5))
print(union_p1_df.tail(5))
print(union_p1_df.shape)
input('next one?')

union_p2_df = dh_combined.vertical_stack_dataframes(dh_003_2.get_dataframe_iterator(), dh_005_2.get_dataframe_iterator(), set_as_current_df=False)
print(union_p2_df.head(5))
print(union_p2_df.tail(5))
print(union_p2_df.shape)
input('next one?')

union_p1_p2 = dh_combined.vertical_stack_dataframes(union_p1_df, union_p2_df , set_as_current_df=True)
print(union_p1_p2.head(5))
print(union_p1_p2.tail(5))
print(union_p1_p2.shape)
input('DONE one?')

# union_p1_p2.to_csv("../data/input/A_and_V_BothProtocols_stacked_and_labeled.csv", index=False)