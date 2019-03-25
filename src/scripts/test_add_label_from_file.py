import os, re
from src.pipeline.DataHandler import DataHandler


#########
#
# PIPELINE FUNCTION IN THE MAKING
#
#########

list_with_subjects = [
    '../data/input/006',
    '../data/input/008'
]

back_keywords = ['Back']
thigh_keywords = ['Thigh']
label_keywords = ['GoPro', "Label", 'labels', 'target', 'targets']

subjects = {}
for subject in list_with_subjects:
    if not os.path.exists(subject):
        print("Could not find Subject at path: ", subject)

    files = {}
    for sub_files_and_dirs  in os.listdir(subject):
        print(sub_files_and_dirs)
        words = re.split("[_ .]", sub_files_and_dirs)
        words = list(map(lambda x: x.lower(), words))

        check_for_matching_word = lambda words, keywords : [True if keyword.lower() == word.lower() else False for word in words for keyword in keywords]

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
                                out_path=None,
                                merge_column=None,
                                master_columns=['bx', 'by', 'bz'],
                                slave_columns=['tx', 'ty', 'tz'],
                                rearrange_columns_to=None,
                                save=False,
                                left_index=True,
                                right_index=True
                                )

    dh.add_columns_based_on_csv(label, columns_name=["label"], join_type="inner")

    if idx == 0:
        merged_df = dh.get_dataframe_iterator()
        continue


    merged_old_shape = merged_df.shape
    # vertically stack the dataframes aka add the rows from dataframe2 as rows to the dataframe1
    merged_df = dh_stacker.vertical_stack_dataframes(merged_df, dh.get_dataframe_iterator(), set_as_current_df=False)

    print("shape merged df: ", merged_df.shape, "should be ", dh.get_dataframe_iterator().shape, "  more than old  ", merged_old_shape)

print("Final merge form: ", merged_df.shape)



