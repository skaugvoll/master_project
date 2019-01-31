import os, sys
import cwa_converter
from pipeline.DataHandler import DataHandler





if __name__ == '__main__':
    dh = DataHandler()
    print('created datahandler')

    unzipped_path = dh.unzip_7z_archive(
        filepath=os.path.join(os.getcwd(), '../data/input/testSNTAtle.7z'),
        unzip_to_path='../data/temp',
        cleanup=False
    )

    print('UNZIPPED PATH RETURNED', unzipped_path)


    cwa_converter.convert_cwas_to_csv_with_temp(
        subject_dir=unzipped_path,
        out_dir=unzipped_path,
        paralell=True
    )

    # dh.cleanup_temp_folder()
