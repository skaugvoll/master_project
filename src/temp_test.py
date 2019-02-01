import os, sys
import cwa_converter
from pipeline.DataHandler import DataHandler





if __name__ == '__main__':
    dh = DataHandler()
    print('created datahandler')


    # # unzip cwas from 7z arhcive
    # unzipped_path = dh.unzip_7z_archive(
    #     filepath=os.path.join(os.getcwd(), '../data/input/testSNTAtle.7z'),
    #     unzip_to_path='../data/temp',
    #     cleanup=False
    # )
    #
    # print('UNZIPPED PATH RETURNED', unzipped_path)
    #
    # # convert the cwas to independent csv containing timestamp xyz and temp
    # back_csv, thigh_csv = cwa_converter.convert_cwas_to_csv_with_temp(
    #     subject_dir=unzipped_path,
    #     out_dir=unzipped_path,
    #     paralell=True
    # )
    #
    # # Timesynch and concate csvs
    # dh.merge_csvs_on_first_time_overlap(
    #     master_csv_path=back_csv,
    #     slave_csv_path=thigh_csv,
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

    # dh.merge_csvs_on_first_time_overlap(
    #     '../data/temp/testSNTAtle.7z/testSNTAtle/P1_atle_B.csv',
    #     '../data/temp/testSNTAtle.7z/testSNTAtle/P1_atle_T.csv',
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
    #     ],
    #     # out_path='../data/thomas/test.csv'
    # )

    dh.convert_ADC_temp_to_C(
        dataframe=None,
        dataframe_path='/app/data/temp/testSNTAtle.7z/testSNTAtle/P1_atle_B_TEMP_SYNCHED_BT.csv',
        normalize=False,
        save=True
    )



    # CLEAN THE TEMPORARY (temp) FOLDER
    # dh.cleanup_temp_folder()
