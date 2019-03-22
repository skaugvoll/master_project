import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from pipeline.DataHandler import DataHandler
from pipeline.Plotter import Plotter
import cwa_converter



def sync_csv_shit(dh):
        # Create a data handling object for importing and manipulating dataset ## PREPROCESSING
        os.system("rm -rf ../data/temp/4000181.7z/")
        # print('CREATING datahandler')
        # dh = DataHandler()
        # print('CREATED datahandler')

        ##########################

        # unzip cwas from 7z arhcive
        unzipped_path = dh.unzip_7z_archive(
            filepath=os.path.join(os.getcwd(), '../data/input/4000181.7z'),
            unzip_to_path='../data/temp',
            cleanup=False
        )

        print('UNZIPPED PATH RETURNED', unzipped_path)

        ##########################

        # convert the cwas to independent csv containing timestamp xyz and temp
        back_csv, thigh_csv = cwa_converter.convert_cwas_to_csv_with_temp(
            subject_dir=unzipped_path,
            out_dir=unzipped_path,
            paralell=True
        )

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


        ##########################

        dh.convert_ADC_temp_to_C(
            dataframe=df,
            dataframe_path=None,
            normalize=False,
            save=True
        )

        df = dh.get_dataframe_iterator()
        print(df.head(5))

        ##########################

        make_dataframe(dh)


def make_dataframe(dh):
        pl = Plotter()

        df = '../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv'

        print('SET INDEX TO TIMESTAMP')
        # test that this works with a dataframe and not only path to csv
        # thus pre-loaded and makes it run a little faster
        dh.convert_column_from_str_to_datetime(
            dataframe=df,
        )
        dh.set_column_as_index("time")
        print('DONE')

        df = dh.get_dataframe_iterator()

        ##########################

        print('MAKE NUMERIC')
        dh.convert_column_from_str_to_numeric(column_name="btemp")

        dh.convert_column_from_str_to_numeric(column_name="ttemp")
        print('DONE')

        temp = dh.get_rows_and_columns(dataframe=df, columns=[6, 7])

        pl.plot_temperature(temp)


if __name__ == '__main__':
    print('CREATING datahandler')
    dh = DataHandler()
    print('CREATED datahandler')
    # sync_csv_shit(dh)
    # make_dataframe(dh)



