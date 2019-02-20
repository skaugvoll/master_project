import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: pass


from pipeline.DataHandler import DataHandler

full_path = '/Users/sigveskaugvoll/Documents/Master/data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv'

datahandler = DataHandler()

datahandler.load_dataframe_from_csv('../data/temp/4000181.7z/4000181/',
                                '4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv',
                                header=0,
                                columns=['timestamp', 'bx', 'by', 'bz','tx','ty','tz','btemp','ttemp'],
                                whole_days=False,
                                chunk_size=20000,
                                max_days=6)

# print(datahandler.get_dataframe_iterator().describe)

# print(datahandler.head_dataframe())

datahandler.convert_column_from_str_to_datetime_test(column_name='timestamp')
datahandler.set_column_as_index('timestamp')
# print(datahandler.head_dataframe())

# ADD THE LABELS
datahandler.add_new_column()
datahandler.add_labels_file_based_on_intervals(
    intervals={
        '1': [
            [
                '2017-09-19',
                '18:31:09',
                '23:59:59'
            ],
            [
                '2017-09-20',
                '00:00:00',
                '08:23:08'
            ],
            [
                '2017-09-20',
                '08:35:13',
                '16:03:58'
            ],
            [
                '2017-09-20',
                '16:20:21',
                '23:59:59'
            ],
            [
                '2017-09-21',
                '00:00:00',
                '09:23:07'
            ],
            [
                '2017-09-21',
                '09:35:40',
                '23:59:59'
            ],
            [
                '2017-09-22',
                '00:00:00',
                '09:54:29'
            ]
	    ],
        '3': [
            [
                '2017-09-20',
                '08:23:09',
                '08:35:12'
            ],
            [
                '2017-09-20',
                '16:03:59',
                '16:20:20'
            ],
            [
                '2017-09-21',
                '09:23:08',
                '09:35:39'
            ]
        ]
    }
)

dataframe = datahandler.get_dataframe_iterator()
dataframe.dropna(subset=['label'], inplace=True)

from src import models

lstm = models.get('LSTMTEST', {})
lstm.train(dataframe, epochs=10, batch_size=512, sequence_lenght=250, split=0.8)