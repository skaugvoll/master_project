import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")


import axivity
import cwa_converter
import pandas as pd
from DataHandler import DataHandler


class Tempsynch:
    def __init__(self):
        print("HELLO FROM PIPELINE")
        # Create a data handling object for importing and manipulating dataset ## PREPROCESSING
        print('CREATING datahandler')
        self.dh = DataHandler()
        print('CREATED datahandler')


    def unzipNsynch(self, rel_filepath, unzip=False, temp=False, timesynch=False, unzip_path='../../data/temp', unzip_cleanup=False, cwa_paralell_convert=True):
        # unzip cwas from 7z arhcive

        if unzip:
            os.system("rm -rf ../../data/temp/4000181.7z/")

            unzipped_path = self.dh.unzip_7z_archive(
                filepath=os.path.join(os.getcwd(), rel_filepath),
                unzip_to_path=unzip_path,
                cleanup=unzip_cleanup
            )

        if timesynch:
            with axivity.timesynched_csv('../../data/temp/4000181.7z/4000181', clean_up=False) as synch_csv:
                df = pd.read_csv(synch_csv)
                print(df.head(5))

        if temp:
            # back_csv, thigh_csv = cwa_converter.convert_cwas_to_csv_with_temp(
            #     subject_dir='../../data/temp/4000181.7z/4000181',
            #     out_dir='../../data/temp/4000181.7z/4000181',
            #     paralell=True
            # )

            self.dh.merge_csvs_on_first_time_overlap(
                master_csv_path='../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B.csv',
                slave_csv_path='../../data/temp/4000181.7z/4000181/4000181-26584_2017-09-19_T.csv',
                rearrange_columns_to=[
                    'time',
                    'btemp',
                    'ttemp'
                ]
            )

            self.dh.convert_ADC_temp_to_C(
                dataframe=self.dh.get_dataframe_iterator(),
                dataframe_path=None,
                normalize=False,
                save=True
            )

            # self.dh.merge_multiple_csvs(
            #     master_csv_path='../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_4000181-26584_2017-09-19_T_timesync_output.csv',
            #     slave_csv_path='../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B.csv',
            #     slave2_csv_path='../../data/temp/4000181.7z/4000181/4000181-26584_2017-09-19_T.csv',
            #     merge_how='left',
            #     rearrange_columns_to=[
            #         'time',
            #         'bx',
            #         'by',
            #         'bz',
            #         'tx',
            #         'ty',
            #         'tz',
            #         'btemp',
            #         'ttemp'
            #     ]
            # )


        # self.dh.merge_csvs_on_first_time_overlap(
        #     master_csv_path='../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_4000181-26584_2017-09-19_T_timesync_output.csv',
        #     slave_csv_path='../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv',
        #     master_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz'],
        #     slave_columns=['time', 'btemp', 'ttemp'],
        #     merge_how='left',
        #     rearrange_columns_to=[
        #         'time',
        #         'bx',
        #         'by',
        #         'bz',
        #         'tx',
        #         'ty',
        #         'tz',
        #         'btemp',
        #         'ttemp'
        #     ]
        # )

        self.dh.write_temp_to_txt(
            dataframe_path='../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_4000181-26584_2017-09-19_T_timesync_output_TEMP_SYNCHED_BT.csv'
            # dataframe_path='../../data/temp/csv2/synched_TEMP_SYNCHED_BT.csv'
        )

        # print("READING MASTER CSV")
        # master_df = pd.read_csv('../../data/temp/csv2/synched.csv')
        #
        # print("READING SLAVE CSV")
        # slave_df = pd.read_csv('../../data/temp/csv2/temptext.txt')
        #
        # # Merge the csvs
        # print("MERGING MASTER AND SLAVE CSV")
        # merged_df = pd.concat([master_df, slave_df], axis=1,)
        #
        #
        # master_file_dir, master_filename_w_format = os.path.split('../../data/temp/csv2/synched.csv')
        # out_path = os.path.join(master_file_dir, master_filename_w_format.split('.')[0] + '_TEMP_BT.csv')
        #
        #
        # print("SAVING MERGED CSV")
        # print("DONE, here is a sneak peak:\n", merged_df.head(5))
        # print("Saving")
        # merged_df.to_csv(out_path, index=False, float_format='%.6f')
        # print("Saved synched and merged as csv to : ", os.path.abspath(out_path))

        # self.dh.merge_csvs_on_first_time_overlap(
        #     master_csv_path='../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_4000181-26584_2017-09-19_T_timesync_output.csv',
        #     slave_csv_path='../../data/temp/4000181.7z/4000181/temptext.txt',
        #     master_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz'],
        #     slave_columns=['time', 'btemp', 'ttemp'],
        # )



if __name__ == '__main__':
    t = Tempsynch()
    print("Start")
    t.unzipNsynch('../../data/input/4000181.7z')


    print('Stopp')