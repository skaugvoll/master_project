import numpy as np

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


    def max_min_delta(self, array):
        '''
        # The variation in temperature from the warmest to the coldest
        :param array:
        :return:
        '''
        max_temp = max(array)
        min_temp = min(array)
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

    def train(self, data, samples_pr_window, train_overlap):
        '''

        :param data: one file containing all the training data
        :param samples_pr_window:
        :param train_overlap:
        :return:
        '''

