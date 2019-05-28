import sys, os

try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

import pandas as pd
import time
from matplotlib import pyplot as plt
from src.utils.progressbar import printProgressBar


def create_dataFrame_with_n_examples(numExamples=None, columns=[], testValues=[]):

    assert len(columns) > 0, "Must specify column names"

    df = pd.DataFrame(columns=columns)

    if numExamples == None or numExamples == 0:
        return df

    if not testValues:
        return df

    assert len(columns) == len(testValues), "Length of columns and test values do not match"

    for _ in range(numExamples):
        row = {}
        for c, v in zip(columns, testValues):
            row[c] = v

        df.loc[len(df)] = row

    return df


def append_n_examples_to_dataframe(dataframe, numExamples=1, values=[]):
    assert len(dataframe.columns) == len(values), "number of values must match number of columns"

    if numExamples == None or numExamples == 0:
        return dataframe

    print("Adding more rows...")
    for i in range(numExamples):
        row = {}
        for c, v in zip(dataframe.columns, values):
            row[c] = v

        dataframe.loc[len(dataframe)] = row
        sys.stdout.write("\rAdding row #{} of {}".format(i, numExamples))
    print()

def write_to_feather(df, path):
    start_time = time.time()
    df.to_feather('{}.feather'.format(path))
    cost = time.time() - start_time
    fw = "--- Feather savetime: {} seconds ---".format(cost)
    return cost



def write_to_pickle(df, path):
    start_time = time.time()
    df.to_pickle("{}.pkl".format(path))
    cost = time.time() - start_time
    pw = "--- Pickle savetime: {} seconds ---".format(cost)
    return cost



def read_from_feather(path):
    start_time = time.time()
    _ = pd.read_feather(path + '.feather')
    cost = time.time() - start_time
    fr = "--- Feather savetime: {} seconds ---".format(cost)
    return cost

def read_from_pickle(path):
    start_time = time.time()
    _ = pd.read_pickle(path + '.pkl')
    cost = time.time() - start_time
    pr = "--- Pickle savetime: {} seconds ---".format(cost)
    return cost

def cleanup(path):
    os.remove(path)


def main(test_to_n_example, num_examples_to_add, columns, values, name="MemoryPerformanceExperiment"):
    feather_file_size = []
    feather_write_costs = []
    feather_read_costs = []
    pickle_file_size = []
    pickle_write_costs = []
    pickle_read_costs = []


    xaxis = []

    df = create_dataFrame_with_n_examples(None, columns=columns, testValues=[])

    print("Starting experiment")
    experiment_start = time.time()
    for i in range(0, test_to_n_example, num_examples_to_add):
        xaxis.append(i)

        append_n_examples_to_dataframe(df, num_examples_to_add, values)


        ####
        # GET PERFORMANCE
        ####
        # TODO check if dir exists or create it
        feather_write_costs.append(write_to_feather(df, "../data/temp/MemoryPerformance/{}".format(name)))
        feather_read_costs.append(read_from_feather("../data/temp/MemoryPerformance/{}".format(name)))
        feather_file_size.append(os.path.getsize("../data/temp/MemoryPerformance/{}.{}".format(name, "feather")) / 1000)
        cleanup("../data/temp/MemoryPerformance/{}.{}".format(name, "feather"))

        pickle_write_costs.append(write_to_pickle(df, "../data/temp/MemoryPerformance/{}".format(name)))
        pickle_read_costs.append(read_from_pickle("../data/temp/MemoryPerformance/{}".format(name)))
        pickle_file_size.append(os.path.getsize("../data/temp/MemoryPerformance/{}.{}".format(name, "pkl")) / 1000)
        cleanup("../data/temp/MemoryPerformance/{}.{}".format(name, "pkl"))

        printProgressBar(i, test_to_n_example, 20, "Performing for i = {}".format(i))
    printProgressBar(test_to_n_example, test_to_n_example, 20, "Performing for i = {}".format(test_to_n_example))
    experiment_end = time.time()

    print("Saving result image to path: {}".format("../data/temp/MemoryPerformance/"))
    fig1, ax1 = plt.subplots()
    ax1.plot(xaxis, feather_read_costs, 'r', label="Feather read")
    ax1.plot(xaxis, feather_write_costs, 'g', label="Feather write")

    ax1.plot(xaxis, pickle_read_costs, 'b', label="Pickle read")
    ax1.plot(xaxis, pickle_write_costs, 'y', label="Pickle write")

    ax1.legend()
    ax1.set_title("Writing and Reading Speed Comparison")
    ax1.set_xlabel("Number of rows")
    ax1.set_ylabel("Seconds")
    fig1.savefig("../data/temp/MemoryPerformance/{}.png".format(name+"Speeds"))

    fig2, ax2 = plt.subplots()
    ax2.plot(xaxis, feather_file_size, 'c', label="Feather Size (KB)")
    ax2.plot(xaxis, pickle_file_size, 'm', label="Pickle Size (KB)")

    ax2.legend()
    ax2.set_title("File Size Comparison")
    ax2.set_xlabel("Number of rows")
    ax2.set_ylabel("KB")
    fig2.savefig("../data/temp/MemoryPerformance/{}.png".format(name+"FileSize"))




    ### Memory Clean up
    del df
    del feather_write_costs
    del feather_read_costs
    del pickle_read_costs
    del pickle_write_costs
    print("Total runtime = {} seconds".format(experiment_end-experiment_start))



if __name__ == '__main__':
    # df = create_dataFrame_with_n_examples(2, columns=['a', 'b'], testValues=[1, 2])
    # append_n_examples_to_dataframe(df, 3, [4, 5])
    #
    # print(df)

    # main(1000, 100, ['a', 'b'], [1, 2])


    main(1000000, 180000, ['timestart', 'timeend', 'conf', 'target'], ["2018-01-09 16:02:00.000",
                                                                        "2018-01-09 16:02:04.980",
                                                                        "0.37766775488853455",
                                                                        "3"])






