import io
import feather
import pandas as pd
import time

# print('hallo')
#
df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'B': [1, 2, 3, 4, 5, 6, 7, 8, 9]}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])

##HDF
# df.to_hdf('./trained_models/data.h5', key='df', mode='w')
start_time = time.time()
df = pd.read_hdf('./trained_models/data.h5', 'df')
print("--- HDF savetime: {} seconds ---".format(time.time() - start_time))
print(df)


##FEATHER
# df = df.reset_index()
# df.to_feather('./trained_models/data.feather')
start_time = time.time()
df = pd.read_feather('./trained_models/data.feather')
print("--- Feather savetime: {} seconds ---".format(time.time() - start_time))
df = df.set_index('index')
print(df)

##PICKLE
# df.to_pickle("./trained_models/data.pkl")
start_time = time.time()
df = pd.read_pickle('./trained_models/data.pkl')
print("--- Pickle savetime: {} seconds ---".format(time.time() - start_time))
print(df)


# df.to_csv("./trained_models/data.csv")
start_time = time.time()
df = pd.read_csv('./trained_models/data.csv')
print("--- CSV savetime: {} seconds ---".format(time.time() - start_time))
df.columns = ['index', 'A', 'B']
df = df.set_index('index')
print(df)



