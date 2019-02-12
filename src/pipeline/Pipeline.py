import sys, os
# try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
# except: print("SAdsadsadhsa;hkldasjkd")

from pipeline.DataHandler import DataHandler
from src import models
import cwa_converter


class Pipeline:
    def __init__(self):
        print("HELLO FROM PIPELINE")


    def unzip_extractNconvert_temp_merge_dataset(self, rel_filepath, label_interval, label_mapping):
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
            filepath=os.path.join(os.getcwd(), rel_filepath),
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

        ##########################
        #
        #
        ##########################


        print('ADDING LABELS')
        dh.add_new_column()
        print('DONE')

        dh.add_labels_file_based_on_intervals(
            intervals=label_interval,
            label_mapping=label_mapping
        )


        # ##########################
        # #
        # #
        # ##########################

        # dh.show_dataframe()
        df = dh.get_dataframe_iterator()

        return df, dh

    # TODO create method for unzip_extractNconvert_temp_stack_dataset() or adopt the above def..

    def get_features_and_labels(self, df, dh=None, columns_back=[0,1,2,6], columns_thigh=[3,4,5,7], column_label=[8]):
        if dh is None:
            dh = DataHandler()

        back_feat = dh.get_rows_and_columns(dataframe=df, columns=columns_back).values

        thigh_feat = dh.get_rows_and_columns(dataframe=df, columns=columns_thigh).values

        labels = dh.get_rows_and_columns(dataframe=df, columns=column_label).values

        return back_feat, thigh_feat, labels


if __name__ == '__main__':
    p = Pipeline()