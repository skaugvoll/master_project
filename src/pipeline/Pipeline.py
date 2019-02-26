import sys, os
# try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
# except: print("SAdsadsadhsa;hkldasjkd")
import re
import numpy as np
import cwa_converter
import time
import random
import pickle
from collections import Counter
from multiprocessing import Process, Queue, current_process, freeze_support
from pipeline.DataHandler import DataHandler
from src import models
import utils.temperature_segmentation_and_calculation as temp_feature_util

class Pipeline:
    def __init__(self):
        print("HELLO FROM PIPELINE")
        # Create a data handling object for importing and manipulating dataset ## PREPROCESSING
        print('CREATING datahandler')
        self.dh = DataHandler()
        print('CREATED datahandler')
        self.dataframe = None



    def unzip_extractNconvert_temp_merge_dataset(self, rel_filepath, label_interval, label_mapping, unzip_path='../data/temp', unzip_cleanup=False, cwa_paralell_convert=True):
        # unzip cwas from 7z arhcive
        unzipped_path = self.dh.unzip_7z_archive(
            filepath=os.path.join(os.getcwd(), rel_filepath),
            unzip_to_path=unzip_path,
            cleanup=unzip_cleanup
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
            paralell=cwa_paralell_convert
        )

        ##########################
        #
        #
        ##########################


        # Timesynch and concate csvs
        self.dh.merge_csvs_on_first_time_overlap(
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

        df = self.dh.get_dataframe_iterator()
        print(df.head(5))
        # input("looks ok ? \n")


        ##########################
        #
        #
        ##########################

        self.dh.convert_ADC_temp_to_C(
            dataframe=df,
            dataframe_path=None,
            normalize=False,
            save=True
        )

        df = self.dh.get_dataframe_iterator()
        print(df.head(5))
        # input("looks ok ? \n")

        ##########################
        #
        #
        ##########################


        print('SET INDEX TO TIMESTAMP')
        #test that this works with a dataframe and not only path to csv
        # thus pre-loaded and makes it run a little faster
        self.dh.convert_column_from_str_to_datetime_test(
                dataframe=df,
        )

        self.dh.set_column_as_index("time")
        print('DONE')

        ##########################
        #
        #
        ##########################


        print('MAKE NUMERIC')
        self.dh.convert_column_from_str_to_numeric(column_name="btemp")

        self.dh.convert_column_from_str_to_numeric(column_name="ttemp")
        print('DONE')

        ##########################
        #
        #
        ##########################


        print('ADDING LABELS')
        self.dh.add_new_column()
        print('DONE')

        self.dh.add_labels_file_based_on_intervals(
            intervals=label_interval,
            label_mapping=label_mapping
        )


        # ##########################
        # #
        # #
        # ##########################

        # dh.show_dataframe()
        df = self.dh.get_dataframe_iterator()

        return df, self.dh

    # TODO create method for unzip_extractNconvert_temp_stack_dataset() or adopt the above def..

    def get_features_and_labels(self, df, dh=None, columns_back=[0,1,2,6], columns_thigh=[3,4,5,7], column_label=[8]):
        if dh is None:
            dh = DataHandler()

        back_feat, thigh_feat, labels = None, None, None

        print(columns_back, columns_thigh, column_label)

        if columns_back:
            back_feat = dh.get_rows_and_columns(dataframe=df, columns=columns_back).values
        if columns_thigh:
            thigh_feat = dh.get_rows_and_columns(dataframe=df, columns=columns_thigh).values
        if column_label:
            labels = dh.get_rows_and_columns(dataframe=df, columns=column_label).values

        return back_feat, thigh_feat, labels


    ####################################################################################################################
    #                                   vPARALLELL PIPELINE EXECUTE WINDOW BY WINDOW CODEv                             #
    ####################################################################################################################

    # MODEL Klassifisering
    def model_classification_worker(self, input_q, output, model):
        for idx, window in iter(input_q.get, 'STOP'):
            print("model_classification_worker executing")

            # print("MODLE CLASSIFICATION: ", idx, window)

            # Vil ha in ett ferdig window, aka windows maa lages utenfor her og addes til queue
            res = model.window_classification(window)

            # SUBMIT TASKS FOR ACTIVITY CLASSIFICATION
            output.put((idx, res))
            print("worker done")

    # AKTIVITET Klassifisering
    def activity_classification_worker(self, input_q):
        # todo implement this to do LSTM classifications!
        '''
        models: 1: both, 2:thigh, 3:back
        :param input_q:
        :return:
        '''
        for window in iter(input_q.get, 'STOP'):
            window_idx, model = window[0], window[1]
            # time.sleep(0.5 * random.random())
            print("WINDOW IDEX TO DF: {} \t LSTM MODLE TO USE: {} \n DFR: {}".format(window_idx, model, self.dataframe.iloc[window_idx, [0,1,2,3,4,5]]))


    def parallel_pipeline_classification_run(self, dataframe, model_path, samples_pr_window, train_overlap):
        self.dataframe = dataframe
        NUMBER_OF_PROCESSES_models = 3
        NUMBER_OF_PROCESSES_class = 1

        # Create queues
        model_queue = Queue()
        activity_queue = Queue()


        # Submit tasks for model klassifisering
        back_feat, thigh_feat, label = self.get_features_and_labels(self.dataframe) # returns numpy arrays
        back_feat = temp_feature_util.segment_acceleration_and_calculate_features(back_feat,
                                                                    samples_pr_window=samples_pr_window,
                                                                    overlap=train_overlap)

        thigh_feat = temp_feature_util.segment_acceleration_and_calculate_features(thigh_feat,
                                                                    samples_pr_window=samples_pr_window,
                                                                    overlap=train_overlap)

        # concatinates example : [[1,2,3],[4,5,6]] og [[a,b,c], [d,e,f]] --> [[1,2,3,a,b,c], [4,5,6,d,e,f]]
        # akas rebuild the dataframe shape
        both_features = np.hstack((back_feat, thigh_feat))


        for idx, window in enumerate(both_features):
            model_queue.put((idx, window))

        # Lists to maintain processes
        processes_model = []
        processes_class = []

        # CREATE a worker processes on model klassifisering
        for _ in range(NUMBER_OF_PROCESSES_class):
            processes_class.append(Process(target=self.activity_classification_worker, args=(activity_queue,)
                                           )
                                   )

        # START the worker processes
        for process in processes_class:
            process.start()

        # CREATE a worker processes on model klassifisering
        for _ in range(NUMBER_OF_PROCESSES_models):
            # todo fix the path here, to be a input parameter
            RFC = pickle.load(open(model_path, 'rb'))
            processes_model.append(Process(target=self.model_classification_worker, args=(model_queue,
                                                                                          activity_queue,
                                                                                          RFC
                                                                                          )
                                           )
                                   )

        # START the worker processes
        for process in processes_model:
            process.start()

        # waith for tasks_queue to become empty before sending stop signal to workers
        while not model_queue.empty():
            pass

        # Tell child processes to stop waiting for more jobs
        for _ in range(NUMBER_OF_PROCESSES_models):
            model_queue.put('STOP')

        # LET ALL PROCESSES ACTUALLY TERMINATE AKA FINISH THE JOB THEIR DOING
        while any([p.is_alive() for p in processes_model]):
            pass

        # waith for activity_queue to become empty before sending stop signal to workers
        while not activity_queue.empty():
            pass

        # Tell child processes to stop waiting for more jobs
        for _ in range(NUMBER_OF_PROCESSES_class):
            activity_queue.put('STOP')

        # LET ALL PROCESSES ACTUALLY TERMINATE AKA FINISH THE JOB THEIR DOING
        while any([p.is_alive() for p in processes_class]):
            pass

        all_processes = processes_model + processes_class

        # join the processes aka block the threads, do not let them take on any more jobs
        for process in all_processes:
            process.join()

        print("\nALL PROCESSES STATUS BEFORE TERMINATING:\n{}".format(all_processes))

        # Kill all the processes to release memory or process space or something. its 1.15am right now
        for process in all_processes:
            process.terminate()

        # continue the pipeline work
        # ...
        # ...
        # ...


    ####################################################################################################################
    #                                   ^PARALLELL PIPELINE EXECUTE WINDOW BY WINDOW CODE^                             #
    ####################################################################################################################

    def create_large_dafatframe_from_multiple_input_directories(self,
                                                                list_with_subjects,
                                                                back_keywords=['Back'],
                                                                thigh_keywords = ['Thigh'],
                                                                label_keywords = ['GoPro', "Labels"],
                                                                out_path=None,
                                                                merge_column = None,
                                                                master_columns = ['bx', 'by', 'bz'],
                                                                slave_columns = ['tx', 'ty', 'tz'],
                                                                rearrange_columns_to = None,
                                                                save=False,
                                                                added_columns_name=["new_col"]
                                                                ):
        # list_with_subjects = [
        #     '../data/input/006',
        #     '../data/input/008'
        # ]


        subjects = {}
        for subject in list_with_subjects:
            if not os.path.exists(subject):
                print("Could not find Subject at path: ", subject)

            files = {}
            for sub_files_and_dirs in os.listdir(subject):
                print(sub_files_and_dirs)
                words = re.split("[_ .]", sub_files_and_dirs)
                words = list(map(lambda x: x.lower(), words))

                check_for_matching_word = lambda words, keywords: [True if keyword.lower() == word.lower() else False
                                                                   for word in words for keyword in keywords]

                if any(check_for_matching_word(words, back_keywords)):
                    files["backCSV"] = sub_files_and_dirs

                elif any(check_for_matching_word(words, thigh_keywords)):
                    files["thighCSV"] = sub_files_and_dirs

                elif any(check_for_matching_word(words, label_keywords)):
                    files["labelCSV"] = sub_files_and_dirs

            subjects[subject] = files

        # print(subjects)

        merged_df = None
        dh = DataHandler()
        dh_stacker = DataHandler()
        for idx, root_dir in enumerate(subjects):
            subject = subjects[root_dir]
            print("SUBJECT: \n", subject)

            master = os.path.join(root_dir, subject['backCSV'])
            slave = os.path.join(root_dir, subject['thighCSV'])
            label = os.path.join(root_dir, subject['labelCSV'])

            # dh = DataHandler()
            dh.merge_csvs_on_first_time_overlap(
                master,
                slave,
                out_path=out_path,
                merge_column=merge_column,
                master_columns=master_columns,
                slave_columns=slave_columns,
                rearrange_columns_to=rearrange_columns_to,
                save=save,
                left_index=True,
                right_index=True
            )

            dh.add_columns_based_on_csv(label, columns_name=added_columns_name, join_type="inner")

            if idx == 0:
                merged_df = dh.get_dataframe_iterator()
                continue

            merged_old_shape = merged_df.shape
            # vertically stack the dataframes aka add the rows from dataframe2 as rows to the dataframe1
            merged_df = dh_stacker.vertical_stack_dataframes(merged_df, dh.get_dataframe_iterator(),
                                                             set_as_current_df=False)

            print(
            "shape merged df: ", merged_df.shape, "should be ", dh.get_dataframe_iterator().shape, "  more than old  ",
            merged_old_shape)

        print("Final merge form: ", merged_df.shape)
        return merged_df


if __name__ == '__main__':
    p = Pipeline()