import multiprocessing
import numpy as np
import utils.temperature_segmentation_and_calculation as temp_feature_util
# from collections import Counter
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle



class HARRandomForrest():
    def __init__(self):
        self.RFC_classifier = None
        self.predictions = None
        self.test_ground_truth_labels = None
        self.accuracy = None
        self.confusion_matrix = None
        self.model_path= None


    def save_model(self, path="./trained_rfc.save"):
        self.model_path = path
        pickle.dump(self.RFC_classifier, open(self.model_path, 'wb'))

    def load_model(self, path="./trained_rfc.save"):
        self.RFC_classifier = pickle.load(open(path, 'rb'))

    def get_model_path(self):
        return self.model_path


    def train_old(self,
                  back_training_feat,
                  thigh_training_feat,
                  labels,
                  samples_pr_window,
                  train_overlap,
                  number_of_trees=100,
                  verbose=2
                  ):

            # print("RFC TRAIN BTF: ", back_training_feat)

            back_training_feat = temp_feature_util.segment_acceleration_and_calculate_features_old(back_training_feat,
                                                                                  samples_pr_window=samples_pr_window,
                                                                                  overlap=train_overlap)

            thigh_training_feat = temp_feature_util.segment_acceleration_and_calculate_features_old(thigh_training_feat,
                                                                                    samples_pr_window=samples_pr_window,
                                                                                    overlap=train_overlap)

            labels = temp_feature_util.segment_labels(labels, samples_pr_window=samples_pr_window, overlap=train_overlap)
            if self.test_ground_truth_labels is None:
                self.test_ground_truth_labels = labels
           
            both_features = np.hstack((back_training_feat, thigh_training_feat))

            print("Hopefully this RFC will create and fit")
            self.RFC_classifier = RFC(n_estimators=number_of_trees,
                                 class_weight="balanced",
                                 random_state=0,
                                 n_jobs=-1,
                                 verbose=verbose
                                 ).fit(both_features, labels)

            print("I kinda diiiid! ")

    def train(self,
              back_training_feat,
              thigh_training_feat,
              back_temp,
              thigh_temp,
              labels,
              samples_pr_window,
              train_overlap,
              sampling_freq=50,
              number_of_trees=100,
              snt_memory_seconds=600,
              use_acc_data=True,
              verbose=2
              ):

        # print("RFC TRAIN BTF: ", back_training_feat)

        back_training_feat = temp_feature_util.segment_acceleration_and_calculate_features(back_training_feat,
                                                                                           temp=back_temp,
                                                                                           samples_pr_window=samples_pr_window,
                                                                                           sampling_frequency=sampling_freq,
                                                                                           overlap=train_overlap,
                                                                                           seconds_to_remember=snt_memory_seconds,
                                                                                           use_acc_data=use_acc_data)

        thigh_training_feat = temp_feature_util.segment_acceleration_and_calculate_features(thigh_training_feat,
                                                                                            temp=thigh_temp,
                                                                                            samples_pr_window=samples_pr_window,
                                                                                            sampling_frequency=sampling_freq,
                                                                                            overlap=train_overlap,
                                                                                            seconds_to_remember=snt_memory_seconds,
                                                                                            use_acc_data=use_acc_data)


        labels = temp_feature_util.segment_labels(labels, samples_pr_window=samples_pr_window, overlap=train_overlap)
        if self.test_ground_truth_labels is None:
            self.test_ground_truth_labels = labels

        both_features = np.hstack((back_training_feat, thigh_training_feat))


        self.RFC_classifier = RFC(n_estimators=number_of_trees,
                                  class_weight="balanced",
                                  random_state=0,
                                  n_jobs=-1,
                                  verbose=verbose
                                  ).fit(both_features, labels)



    def test(self,
             back_test_feat,
             thigh_test_feat,
             temps,
             labels,
             samples_pr_window,
             sampling_freq=50,
             snt_memory_seconds=600,
             use_acc_data=True,
             train_overlap=.8):

        # print("RFC TEST BTF: ", back_test_feat)

        back_test_feat = temp_feature_util.segment_acceleration_and_calculate_features(back_test_feat,
                                                                                       samples_pr_window=samples_pr_window,
                                                                                       sampling_frequency=sampling_freq,
                                                                                       temp=temps[0],
                                                                                       overlap=train_overlap,
                                                                                       seconds_to_remember=snt_memory_seconds,
                                                                                       use_acc_data=use_acc_data)

        thigh_test_feat = temp_feature_util.segment_acceleration_and_calculate_features(thigh_test_feat,
                                                                                        temp=temps[1],
                                                                                        samples_pr_window=samples_pr_window,
                                                                                        sampling_frequency=sampling_freq,
                                                                                        overlap=train_overlap,
                                                                                        seconds_to_remember=snt_memory_seconds,
                                                                                        use_acc_data=use_acc_data)

        self.test_ground_truth_labels = temp_feature_util.segment_labels(labels, samples_pr_window=samples_pr_window, overlap=train_overlap)

        both_features = np.hstack((back_test_feat, thigh_test_feat))

        print("Hopefully this RFC test and predict")
        self.predictions = self.RFC_classifier.predict(both_features)
        print("I kinda diiiid! ")
        print("PREDICTIONS: \n{}".format(self.predictions))
        return self.predictions, self.test_ground_truth_labels, self.calculate_confusion_matrix()

    def classify(self,
                 back_test_feat,
                 thigh_test_feat,
                 temps,
                 samples_pr_window,
                 sampling_freq,
                 snt_memory_seconds,
                 use_acc_data,
                 train_overlap):

        back_test_feat = temp_feature_util.segment_acceleration_and_calculate_features(back_test_feat,
                                                                                       temp=temps[0],
                                                                                       samples_pr_window=samples_pr_window,
                                                                                       sampling_frequency=sampling_freq,
                                                                                       overlap=train_overlap,
                                                                                       seconds_to_remember=snt_memory_seconds,
                                                                                       use_acc_data=use_acc_data)

        thigh_test_feat = temp_feature_util.segment_acceleration_and_calculate_features(thigh_test_feat,
                                                                                        temp=temps[1],
                                                                                        samples_pr_window=samples_pr_window,
                                                                                        sampling_frequency=sampling_freq,
                                                                                        overlap=train_overlap,
                                                                                        seconds_to_remember=snt_memory_seconds,
                                                                                        use_acc_data=use_acc_data)
        both_features = np.hstack((back_test_feat, thigh_test_feat))
        self.predictions = self.RFC_classifier.predict(both_features)
        return self.predictions


    def calculate_accuracy(self):
        '''
        This cannot be called after training, only after test or classify, as .fit returnes the trained RFC object
        :return:
        '''
        # TODO check that object attributes are not None, raise exception
        gt = self.test_ground_truth_labels
        preds = self.predictions

        self.accuracy = accuracy_score(gt, preds)
        return self.accuracy

    def calculate_confusion_matrix(self):
        # TODO check that object attributes are not None, raise exception
        gt = self.test_ground_truth_labels
        preds = self.predictions

        self.confusion_matrix = confusion_matrix(gt, preds)
        return self.confusion_matrix

    def window_classification(self, window):
        res = self.RFC_classifier.predict([window])
        return res

