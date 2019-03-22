import os, sys
import cwa_converter
from pipeline.DataHandler import DataHandler





if __name__ == '__main__':
    print('CREATING datahandler')
    dh = DataHandler()
    print('CREATED datahandler')

    try:
        os.system("rm -rf ../data/temp/P2_vegar.7z")
    except:
        print("Could not remove folder, perhaps path is wrong")
        sys.exit(0)

    ################################# RUNNING ################

    # unzip cwas from 7z arhcive
    unzipped_path = dh.unzip_7z_archive(
        filepath=os.path.join(os.getcwd(), '../data/input/P2_vegar.7z'),
        unzip_to_path='../data/temp',
        cleanup=False
    )

    print('UNZIPPED PATH RETURNED', unzipped_path)

    # convert the cwas to independent csv containing timestamp xyz and temp
    back_csv, thigh_csv = cwa_converter.convert_cwas_to_csv_with_temp(
        subject_dir=unzipped_path,
        out_dir=unzipped_path,
        paralell=True
    )

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

    dh.convert_ADC_temp_to_C(
        dataframe=None,
        dataframe_path='/app/data/temp/P2_vegar.7z/P2_vegar/P2_vegar_B_TEMP_SYNCHED_BT.csv',
        normalize=False,
        save=True
    )

######################################## TESTING #############

    dh.merge_csvs_on_first_time_overlap(
        '../data/temp/P2_vegar.7z/P2_vegar/P2_vegar_B.csv',
        '../data/temp/P2_vegar.7z/P2_vegar/P2_vegar_T.csv',
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
        ],
        # out_path='../data/thomas/test.csv'
    )

    dh.convert_column_from_str_to_datetime(
        dataframe='/app/data/temp/P2_vegar.7z/P2_vegar/P2_vegar_B_TEMP_SYNCHED_BT.csv',
    )

    dh.set_column_as_index("time")


    dh.convert_ADC_temp_to_C(
        dataframe=None,
        dataframe_path='/app/data/temp/P2_vegar.7z/P2_vegar/P2_vegar_B_TEMP_SYNCHED_BT.csv',
        normalize=False,
        save=True
    )


    # dh.get_rows_based_on_timestamp(
    #     start="2018-04-27 10:03:37",
    #     end="2018-04-27 10:03:38"
    # )
    #
    # dh.add_new_column()
    #
    #
    # dh.add_labels_file_based_on_intervals(
    #     intervals={
    #         "1": [
    #             [
    #                 '2018-04-27',
    #                 '10:03:37',
    #                 '10:03:38'
    #             ],
    #             [
    #                 '2018-04-27',
    #                 '10:03:39',
    #                 '11:09:00'
    #             ]
    #         ],
    #         '2' : [
    #             [
    #                 '2018-04-27',
    #                 '11:09:01',
    #                 '12:19:00'
    #             ]
    #         ],
    #         '3' : [
    #             [
    #                 '2018-04-27',
    #                 '12:19:01',
    #                 '14:28:00'
    #             ]
    #         ]
    #     },
    #     label_mapping={"1":"Both", "2": "Thigh", "3": "Back"}
    # )

    # CLEAN THE TEMPORARY (temp) FOLDER
    # dh.cleanup_temp_folder()
