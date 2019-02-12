import os, sys
from pipeline.DataHandler import DataHandler
import cwa_converter

dh = DataHandler()
print('created datahandler')

# dh.load_dataframe_from_7z(
#     input_arhcive_path=os.path.join(os.getcwd(),'../data/input/testSubject08.7z'),
# )

# print("Starting cleanup....")
# dh.cleanup_temp_folder()

# dh.read_and_return_multiple_csv_iterators(
#     dir_path='./models/',
#     filenames=['rfc'],
#     format='py'
# )


# unzip cwas from 7z arhcive
unzipped_path = dh.unzip_7z_archive(
    filepath=os.path.join(os.getcwd(), "../data/input/4000181.7z"),
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
# input("looks ok ? \n")


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
# input("looks ok ? \n")

##########################
#
#
##########################


print('SET INDEX TO TIMESTAMP')
#test that this works with a dataframe and not only path to csv
# thus pre-loaded and makes it run a little faster
dh.convert_column_from_str_to_datetime_test(
        dataframe=df,
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




