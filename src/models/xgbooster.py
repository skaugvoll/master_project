import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, DMatrix
from src.utils import temperature_segmentation_and_calculation as temp_feature_util

class MetaXGBooster:
    def __init__(self):
        self.classifier = None


    def _prepare_data(self,
                      back_training_feat,
                      thigh_training_feat,
                      back_temp,
                      thigh_temp,
                      labels,
                      samples_pr_window,
                      sampling_freq,
                      train_overlap):
        back_training_feat = temp_feature_util.segment_acceleration_and_calculate_features(back_training_feat,
                                                                                           temp=back_temp,
                                                                                           samples_pr_window=samples_pr_window,
                                                                                           sampling_frequency=sampling_freq,
                                                                                           overlap=train_overlap)

        thigh_training_feat = temp_feature_util.segment_acceleration_and_calculate_features(thigh_training_feat,
                                                                                            temp=thigh_temp,
                                                                                            samples_pr_window=samples_pr_window,
                                                                                            sampling_frequency=sampling_freq,
                                                                                            overlap=train_overlap)

        labels = temp_feature_util.segment_labels(labels, samples_pr_window=samples_pr_window, overlap=train_overlap)

        both_features = np.hstack((back_training_feat, thigh_training_feat))
        # We need to convert the dataframe into a DMatrix
        dmatrix = DMatrix(both_features, label=labels)
        return dmatrix, labels

    def train(self,
              back_training_feat,
              thigh_training_feat,
              back_temp,
              thigh_temp,
              labels,
              evaluation=None,
              samples_pr_window=250,
              train_overlap=.8,
              sampling_freq=50,
              number_of_trees=100,
              verbose=2
              ):
        # print("RFC TRAIN BTF: ", back_training_feat)

        dtrain, labels = self._prepare_data(
            back_training_feat,
            thigh_training_feat,
            back_temp,
            thigh_temp,
            labels,
            samples_pr_window=samples_pr_window,
            sampling_freq=sampling_freq,
            train_overlap=train_overlap
        )

        dtest, eval_label = None, None
        if evaluation:
            dtest, eval_label = self._prepare_data(
                evaluation[0],
                evaluation[1],
                evaluation[2],
                evaluation[3],
                evaluation[4],
                samples_pr_window=samples_pr_window,
                sampling_freq=sampling_freq,
                train_overlap=train_overlap
            )

        params = {
            'max_depth': 2,
            'eta': 0.7, # is learning rate
            "verbosity": 2,
        }
        # params['nthread'] = 4 # deafults to max num available
        params['eval_metric'] = 'auc'

        num_rounds = 10
        bst = xgb.train(params, dtrain, num_rounds)
        self.classifier = bst

        return bst



    def test(self,
              back_training_feat,
              thigh_training_feat,
              back_temp,
              thigh_temp,
              labels,
              samples_pr_window=250,
              train_overlap=.8,
              sampling_freq=50,
              ):
        # print("RFC TRAIN BTF: ", back_training_feat)

        dtest, labels = self._prepare_data(
            back_training_feat,
            thigh_training_feat,
            back_temp,
            thigh_temp,
            labels,
            samples_pr_window=samples_pr_window,
            sampling_freq=sampling_freq,
            train_overlap=train_overlap
        )

        print(">>>>>>>>>> EVAL")
        res = self.classifier.predict(dtest)
        labels = dtest.get_label()
        print(res)
        print()
        print(labels)
        print()
        print("error=%f" % (sum(1 for i in range(len(res)) if int(res[i] > 0.5) != labels[i] / float(len(res)))))

        return res




