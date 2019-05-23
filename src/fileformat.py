import io
import feather
import pandas as pd
import time

# print('hallo')
#
# df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'B': [1, 2, 3, 4, 5, 6, 7, 8, 9]}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])

name = '4003601_c'
##HDF
# df.to_hdf('./trained_models/data.h5', key='df', mode='w')
start_time = time.time()
df = pd.read_hdf('../data/output/'+name+'.h5', 'df')
print("--- HDF savetime: {} seconds ---".format(time.time() - start_time))
# print(df)


##FEATHER
# df = df.reset_index()
# df.to_feather('./trained_models/data.feather')
start_time = time.time()
df = pd.read_feather('../data/output/'+name+'.feather')
print("--- Feather savetime: {} seconds ---".format(time.time() - start_time))
df = df.set_index('index')
# print(df)

##PICKLE
# df.to_pickle("./trained_models/data.pkl")
start_time = time.time()
df = pd.read_pickle('../data/output/'+name+'.pkl')
print("--- Pickle savetime: {} seconds ---".format(time.time() - start_time))
# print(df)


# df.to_csv("./trained_models/data.csv")
start_time = time.time()
df = pd.read_csv('../data/output/'+name+'.csv', header='infer')
print("--- CSV savetime: {} seconds ---".format(time.time() - start_time))
# df.columns = ['index', 'A', 'B']
# df = df.set_index('index')
# print(df)



