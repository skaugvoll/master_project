import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier as RFC



class HARRandomForrest():
    def __init__(self):
        pass

    def max(self, array):
        '''
        Finds and returns the highest value in the array
        :param array:
        :return:
        '''
        return np.max(array)

    def min(self, array):
        '''
        Finds and returns the lowest value in the array
        :param array:
        :return:
        '''
        return np.min(array)

    def first_last_delta(self, array):
        temperature_first_sample_in_window = array[0]
        temperature_last_sample_in_window = array[-1]
        return (temperature_last_sample_in_window - temperature_first_sample_in_window)

    def max_min_delta(self, array):
        '''
        # The variation in temperature from the warmest to the coldest
        :param array:
        :return:
        '''
        #max_temp = max(array)
        #min_temp = min(array)
	
        max_temp = np.amax(array)
        min_temp = np.amin(array)
        return (max_temp - min_temp)


    def segment_acceleration_and_calculate_features(self,
                                                    sensor_data,
                                                    samples_pr_window=50,
                                                    overlap=0.0,
                                                    remove_sign_after_calculation=True):
        '''

        :param sensor_data:
        :param samples_pr_window:
        :param overlap:
        :param remove_sign_after_calculation:
        :return:
        '''

        # print("len sensor data: ", sensor_data.shape)
        functions = [
            self.max,
            self.min,
            self.max_min_delta,
            self.first_last_delta,
        ]

        # window_samples = int(sampling_rate * window_length)
        window_samples = samples_pr_window

        # print("Windows samples ", window_samples)
        step_size = int(round(window_samples * (1.0 - overlap)))

        all_features = []

        for window_start in np.arange(0, sensor_data.shape[0], step_size):
            # print("Window start: ", window_start, "Sensor_data.shape[0]: ", sensor_data.shape[0], step_size)
            window_start = int(round(window_start))
            window_end = window_start + int(round(window_samples))
            if window_end > sensor_data.shape[0]:
                break
            window = sensor_data[window_start:window_end]

            # print("Window", window)
            extracted_features = []
            # print("Windows start: ")
            index_of_function = 0
            n = 0
            for func in functions:
                value = func(window)
                extracted_features.append(value)

            all_features.append(np.hstack(extracted_features))

        one_large_array = np.vstack(all_features)

        if remove_sign_after_calculation:
            np.absolute(one_large_array, one_large_array)

        print(one_large_array.shape)

        return one_large_array

    def segment_labels(self, label_data, overlap=0.0, samples_pr_window = 50):
        # window_samples = int(sampling_rate * window_length)
        window_samples = samples_pr_window
        step_size = int(round(window_samples * (1.0 - overlap)))

        labels = []

        for window_start in np.arange(0, label_data.shape[0], step_size):
            window_start = int(round(window_start))
            window_end = window_start + int(round(window_samples))
            if window_end > label_data.shape[0]:
                break
            window = label_data[window_start:window_end]
            # print(window)
            top = self.find_majority_activity(window)
            labels.append(top)

        return np.array(labels)

    def find_majority_activity(self, window):
        sensor_labels_list = window.tolist()
        labels_without_list = []
        for sensor_label in sensor_labels_list:
            labels_without_list.append(sensor_label[0])
        counts = Counter(labels_without_list)
        top = counts.most_common(1)[0][0]
        return top

    def train(self,
              back_training_feat,
              thigh_training_feat,
              labels,
              samples_pr_window,
              train_overlap
              ):

            print("RFC TRAIN BTF: ", back_training_feat)

            back_training_feat = self.segment_acceleration_and_calculate_features(back_training_feat,
                                                                                  samples_pr_window=samples_pr_window,
                                                                                  overlap=train_overlap)

            thigh_training_feat = self.segment_acceleration_and_calculate_features(thigh_training_feat,
                                                                                    samples_pr_window=samples_pr_window,
                                                                                    overlap=train_overlap)

            labels = self.segment_labels(labels, samples_pr_window=samples_pr_window, overlap=train_overlap)
           
            both_features = np.hstack((back_training_feat, thigh_training_feat))

            print("Hopefully this RFC will create and fit")
            RFC_classifier = RFC(n_estimators=100,
                                 class_weight="balanced",
                                 random_state=0,
                                 n_jobs=-1
                                 ).fit(both_features, labels)
            print("I kinda diiiid! ")

