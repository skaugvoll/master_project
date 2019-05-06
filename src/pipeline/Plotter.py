import sys, os


try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os, re
import pandas as pd

class Plotter():
    def __init__(self):
        # self.root_dir = os.path.dirname(os.path.abspath(__file__))
        print


    def plot_lines(self, dataframe, columns_to_print, styles_to_print, outputname):
    # columns_to_print = ['bx', 'by', 'bz', 'tx', 'ty', 'tz']
    # styles_to_print = ['r-', 'g--', 'b-.', 'c', 'y', 'm']

        df = pd.read_csv(dataframe)
        # df.columns = ['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz', 'label']
        # df[['time', 'btemp']].plot(style=['r-'])
        df[columns_to_print].plot(style=styles_to_print)
        plt.savefig(outputname)

    def plot_snt_barchart(self, values, outputname, title):
        # plt.rcdefaults()

        objects = ('1 - All', '2 - Thigh', '3 - Back', '4 - None')
        x = np.arange(len(objects))

        def millions(x, pos):
            'The two args are the value and tick position'
            return '%1.1fM' % (x * 1e-6)

        formatter = FuncFormatter(millions)

        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(formatter)
        bars = ax.bar(x, values, align='center', color='white')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 10,
                    '%d' % int(height),
                    ha='center', va='bottom')
        plt.bar(x, values)
        plt.xticks(x, objects)
        plt.savefig(outputname)
        plt.title(title)

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

    # create single
if __name__ == '__main__':
    pl = Plotter()
    pl.plot_snt_barchart([17398515, 3089850, 2946800, 29461400], 'SNT.png', 'SNT dataset distribution')
    pl.plot_snt_barchart([4077200, 333950, 431850, 3797000], '001.png', 'Recording 001')
    pl.plot_snt_barchart([5930850, 1506850, 883650, 6390650], '002.png', 'Recording 002')
    pl.plot_snt_barchart([1629400, 425600, 369200, 6072600], '003.png', 'Recording 003')
    pl.plot_snt_barchart([1233515, 279000, 265200, 2311000], '004.png', 'Recording 004')
    pl.plot_snt_barchart([2520400, 544450, 996900, 10650250], '005.png', 'Recording 005')
    pl.plot_snt_barchart([375200, 0, 0, 131800], '006.png', 'Recording 006')
    pl.plot_snt_barchart([1631950, 0, 0, 108100], '007.png', 'Recording 007')




    # pl.plot_lines('../../data/temp/merged/res006.csv')
    # pl.plot_temperature('../../data/temp/merged/rresampled006.csv')
    # pl.plot_temperature('../../data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B.csv')
    # pl.print_weekly_view("../../data/output/4000181_timestamped_predictions.csv")


