import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from pipeline.DataHandler import DataHandler
from src import models
import cwa_converter




def something_dataset(rel_filepath, label_interval, label_mapping):
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


def get_features_and_labels(df, dh, columns_back=[0,1,2,6], columns_thigh=[3,4,5,7], column_label=[8]):
    back_feat = dh.get_rows_and_columns(dataframe=df, columns=columns_back).values

    thigh_feat = dh.get_rows_and_columns(dataframe=df, columns=columns_thigh).values

    labels = dh.get_rows_and_columns(dataframe=df, columns=column_label).values

    return back_feat, thigh_feat, labels






################
# SCRIPT STARTS EXECUTING HERE
################
if __name__ == '__main__':
    try:
        print("Deleting temp folders")
        os.system("rm -rf ../data/temp/testSNTAtle.7z/")
        os.system("rm -rf ../data/temp/testVegar.7z/")
        print("...DONE")
        print("Deleting output folders")
        os.system("rm -rf ../data/output/testSNTAtle/")
        print("...DONE")
    except:
        print("hmm")



    df_train, dh_train = something_dataset(rel_filepath='../data/input/testSNTAtle.7z',
                                           label_interval={
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
                                           },
                                           label_mapping={"1": "Both", "2": "Thigh", "3": "Back"}
                                           )

    # remove rows that does not have label
    print("df_train: ", type(df_train))
    df_train.dropna(subset=['label'], inplace=True)
    print(df_train)

    print(">>>>>>>>>>>>>>>>>>>>>>|<<<<<<<<<<<<<<<<<<<<<<")

    df_test, dh_test = something_dataset(rel_filepath='../data/input/testVegar.7z',
                                           label_interval={
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
                                           },
                                           label_mapping={"1": "Both", "2": "Thigh", "3": "Back"}
                                           )

    # remove rows that does not have label
    print("df_test: ", type(df_test))
    df_test.dropna(subset=['label'], inplace=True)
    print(df_test)


    # Get the model
    model = models.get( "RFC", {} )

    back_feat_train, thigh_feat_train, label_train = get_features_and_labels(df_train, dh_train)
    back_feat_test, thigh_feat_test, label_test = get_features_and_labels(df_train, dh_train)

    ##########################
    #
    #
    ##########################

    # Do some magic numbering
    sampling_frequency = 50
    window_length = 120
    tempearture_reading_rate = 120
    samples_pr_second = 1/(tempearture_reading_rate/sampling_frequency)
    samples_pr_window = int(window_length*samples_pr_second)

    ##########################
    #
    #
    ##########################

    # pass to the model, for training
    model.train(back_training_feat=back_feat_train,
                thigh_training_feat=thigh_feat_train,
                labels=label_train,
                samples_pr_window=samples_pr_window,
                train_overlap=0.8,
                number_of_trees=100
                )


    ##########################
    #
    #
    ##########################

    # pass to the model, for predictions
    model.test(back_test_feat=back_feat_test,
               thigh_test_feat=thigh_feat_test,
               labels=label_test,
               samples_pr_window=samples_pr_window,
               train_overlap=0.8)

    preds = model.predictions
    ground_truth = model.test_ground_truth_labels
    print("Calculating accuracy...")
    acc = model.calculate_accuracy()
    print("Done, ACC: {:.4f}".format(acc))

    print("Creating confusion matrix")
    conf_mat = model.calculate_confusion_matrix()
    print("DONE ")
    print(conf_mat)

    print("\nsamples_pr_window: {}\nsamples_pr_second: {}\nwindow_length: {}\n".format(samples_pr_window,
                                                                                       samples_pr_second,
                                                                                       window_length))
