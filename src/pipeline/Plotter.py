import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")


import os, re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class Plotter():
    def __init__(self):
        # self.root_dir = os.path.dirname(os.path.abspath(__file__))
        print
        self.plotter = None
        self.figure = None
        self.axis = None

    def plot_temperature(self, temp_dataframe, outputname):
        df = pd.read_csv(temp_dataframe)
        # df.columns = ['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz', 'label']
        # df[['time', 'btemp']].plot(style=['r-'])
        df[['bx', 'by', 'bz', 'tx', 'ty', 'tz', ]].plot(style=['r-', 'g--', 'b-.', 'c', 'y', 'm'])
        plt.savefig(outputname)



    ##########################

    def print_weekly_view(self, filename):
        root_dir = os.path.dirname(os.path.abspath(__file__))

        maxSampleDay = 14400  # 14400 for 50Hz - assumes only missing data the first and the last day

        subject_file = os.path.join(root_dir, filename)
        subjectid = list(map(int, re.findall('\d+', filename))).pop().__str__()

        labelled_timestamp = pd.read_csv(subject_file, parse_dates=[0], header=None, names=['timestamp', 'label'])
        labelled_timestamp['date'] = labelled_timestamp.loc[:, 'timestamp'].dt.date

        labelled_timestamp = labelled_timestamp.replace({'label': {
            3: 6,
            4: 1,
            5: 1,
            10: 6,
            11: 9,
            12: 9,
            14: 13,
            15: 6,
            16: 9,
            17: 9,
            18: 9
        }})

        days = labelled_timestamp['date'].drop_duplicates()

        # classes
        # 1:walking
        # 2:running
        # 6:standing
        # 7:sitting
        # 8:lying
        # 9:transition
        # 13:cycling

        no_to_color_dict = {
            1: "forestgreen",
            2: "red",
            6: "lightyellow",
            7: "lightcyan",
            8: "skyblue",
            13: "darkorange",
            99: "white"
        }

        labelled_timestamp = labelled_timestamp.replace({'label': no_to_color_dict})

        first_day = labelled_timestamp.loc[labelled_timestamp['date'] == days.iloc[0]]
        first_day = first_day[['timestamp', 'label']]
        first_day = first_day.set_index('timestamp')

        # add no-wear time to the first day
        missingdatapoints = maxSampleDay - first_day.count()
        timestamp_startnoweartime = first_day.first_valid_index() - pd.Timedelta(
            seconds=(missingdatapoints.__int__()) * 6)
        # create new data frame
        missingdata = pd.DataFrame({'timestamp': pd.date_range(start=timestamp_startnoweartime, freq='6s',
                                                               periods=missingdatapoints.__int__())})
        missingdata['label'] = 'white'
        missingdata = missingdata.set_index('timestamp')
        first_day = pd.concat([missingdata, first_day])
        no_of_days = days.count()

        # Make a figure and axes with dimensions as desired.
        # start with one
        fig = plt.figure(figsize=(20, 10))
        st = fig.suptitle('Subject #' + subjectid, fontsize="x-large")

        ax = fig.add_subplot(111)
        ####### First DAY ##########
        cmap = mpl.colors.ListedColormap(first_day.reset_index().label.tolist())

        # If a ListedColormap is used, the length of the bounds array must be
        # one greater than the length of the color list.  The bounds must be
        # monotonically increasing.
        bounds = first_day.reset_index().index.tolist()
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb0 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        # to use 'extend', you must
                                        # specify two extra boundaries:
                                        # boundaries= bounds,
                                        # ticks=bounds,  # optional
                                        # spacing='proportional',
                                        orientation='horizontal')

        a = ['', '2:24am', '4:48am', '7:12am', '9:36am', '12pm', '2:24pm', '4:48pm', '7:12pm', '9:36pm']
        ax.set_xticklabels(a)
        cb0.set_label('Date: ' + days.iloc[
            0].__str__() + ' (orange: cycling, red: running, green: walking, yellow: standing, light blue:sitting, blue: lying)')

        # now later you get a new subplot; change the geometry of the existing
        for c in range(no_of_days - 1):
            current = c + 1
            one_day = labelled_timestamp.loc[labelled_timestamp['date'] == days.iloc[current]]
            one_day = one_day[['timestamp', 'label']]
            one_day = one_day.set_index('timestamp')

            if (one_day.count().__int__() < maxSampleDay):
                missingdatapoints = maxSampleDay - one_day.count().__int__()
                # create new data frame
                missingdataend = pd.DataFrame({'timestamp': pd.date_range(start=one_day.last_valid_index(), freq='6s',
                                                                          periods=missingdatapoints.__int__())})
                missingdataend['label'] = 'white'
                missingdataend = missingdataend.set_index('timestamp')
                one_day = pd.concat([one_day, missingdataend])

            n = len(fig.axes)
            for i in range(n):
                fig.axes[i].change_geometry(n + 1, 1, i + 1)

            # add the new
            ax = fig.add_subplot(n + 1, 1, n + 1)
            ####### Next DAYs ##########
            cmap = mpl.colors.ListedColormap(one_day.reset_index().label.tolist())
            bounds = one_day.reset_index().index.tolist()
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cb0 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                            norm=norm,
                                            orientation='horizontal')
            ax.set_xticklabels(a)
            cb0.set_label('Date: ' + days.iloc[
                current].__str__() + ' (orange: cycling, red: running, green: walking, yellow: standing, light blue:sitting, blue: lying)')

        # shift subplots down:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        plt.savefig("Daily-Chart-" + subjectid)

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues,
                              figure=None,
                              axis=None
                              ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        if not figure and not axis:
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = axis

        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        # plt.savefig("{}.png".format(title))

        return ax

    def plotter_show(self):
        plt.show()

    def plotter_save(self, name="test.png"):
        plt.savefig(name)

    def start_multiple_plots(self, num_rows, num_columns, figsize=None):
        self.figure, self.axis = plt.subplots(num_rows, num_columns, figsize=figsize)
        # self.figure.subplots_adjust(left=0.88, right=0.98, wspace=0.3)
        self.num_rows = num_rows
        self.num_columns = num_columns
        return self.figure, self.axis

    def get_figure(self):
        return self.figure

    def get_axis_at_row_column(self, row, column):
        print(self.axis)

        print()
        print(row, column)
        print()

        if column == None:
            return self.axis[row]

        else:
            return self.axis[row][column]


    def add_plot_to_multiple_plots(self, row, column, axis, title):
        pass







if __name__ == '__main__':
    pl = Plotter()
    # pl.plot_temperature('../../data/temp/merged/res006.csv', 'Original')
    # pl.plot_temperature('../../data/temp/merged/resampled006.csv', 'Resampeled.png')
    # pl.plot_temperature('../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B.csv')
    # pl.print_weekly_view("../../data/output/4000181_timestamped_predictions.csv")


