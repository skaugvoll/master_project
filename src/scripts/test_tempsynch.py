import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")


import axivity
import cwa_converter
import pandas as pd
from pipeline.DataHandler import DataHandler


class Tempsynch:
    def __init__(self):
        # Create a data handling object for importing and manipulating dataset ## PREPROCESSING
        print('CREATING datahandler')
        self.dh = DataHandler()
        print('CREATED datahandler')


    def unzipNsynch(self, rel_filepath, unzip=True, temp=True, txt=True, unzip_path='../../data/temp', unzip_cleanup=False, cwa_paralell_convert=True):
        # unzip cwas from 7z arhcive

        if unzip:
            os.system("rm -rf ../../data/temp/4000181.7z/")
            self.dh.unzip_synch_cwa(rel_filepath)
            #
            # unzipped_path = self.dh.unzip_7z_archive(
            #     filepath=os.path.join(os.getcwd(), rel_filepath),
            #     unzip_to_path=unzip_path,
            #     cleanup=unzip_cleanup
            # )

        if temp:
            back_csv, thigh_csv = cwa_converter.convert_cwas_to_csv_with_temp(
                subject_dir=self.dh.get_unzipped_path(),
                out_dir=self.dh.get_unzipped_path(),
                paralell=True
            )

            self.dh.merge_multiple_csvs(
                master_csv_path=self.dh.get_synched_csv_path(),
                slave_csv_path=back_csv,
                slave2_csv_path=thigh_csv,
                merge_how='left',
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

        if txt:
            self.dh.write_temp_to_txt(
                dataframe=self.dh.get_dataframe_iterator(),
                # dataframe_path='../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_4000181-26584_2017-09-19_T_timesync_output_TEMP_SYNCHED_BT.csv'
            )

            self.dh.concat_dataframes(
                master_path=self.dh.get_synched_csv_path(),
                slave_path=self.dh.get_unzipped_path() + '/btemp.txt',
                slave2_path=self.dh.get_unzipped_path() + '/ttemp.txt',
            )


if __name__ == '__main__':
    t = Tempsynch()
    t.unzipNsynch('../../data/input/4000181.7z')

    print('Stopp')