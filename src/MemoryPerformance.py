import time

import pandas as pd

name = 'bigfile'


######
# WRITING SPEED
######

result_df = pd.read_csv("bigfile.txt", header=None, names=["timestart", "timeend", "confidence", "target"])

  # CSV
start_time = time.time()
result_df.to_csv("../data/output/{}.csv".format(name))

csvT = "--- CSV savetime: {} seconds ---".format(time.time() - start_time)
print(csvT)

 #PICKLE
start_time = time.time()
result_df.to_pickle("../data/output/{}.pkl".format(name))
pickleT = "--- Pickle savetime: {} seconds ---".format(time.time() - start_time)
print(pickleT)

 #HDF
start_time = time.time()
result_df.to_hdf('../data/output/{}.h5'.format(name), key='df', mode='w')
HDFT = "--- HDF savetime: {} seconds ---".format(time.time() - start_time)
print(HDFT)

 #FEATHER
result_df = result_df.reset_index()
start_time = time.time()
result_df.to_feather('../data/output/{}.feather'.format(name))
featherT = "--- Feather savetime: {} seconds ---".format(time.time() - start_time)
print(featherT)


with open('../data/output/' + name + 'time.txt', 'w') as f:
   f.write(name + '\n')
   f.write("WRITING SPEED: \n")
   f.write(csvT + '\n')
   f.write(pickleT + '\n')
   f.write(HDFT + '\n')
   f.write(featherT + '\n')

print(result_df)



######
# READING SPEED
######


##HDF

start_time = time.time()
df = pd.read_hdf('../data/output/'+name+'.h5', 'df')
HDFRT = "--- HDF savetime: {} seconds ---".format(time.time() - start_time)
# print(df)


##FEATHER
# df = df.reset_index()
# df.to_feather('./trained_models/data.feather')
start_time = time.time()
df = pd.read_feather('../data/output/'+name+'.feather')
featherRT = "--- Feather savetime: {} seconds ---".format(time.time() - start_time)
df = df.set_index('index')
# print(df)

##PICKLE
# df.to_pickle("./trained_models/data.pkl")
start_time = time.time()
df = pd.read_pickle('../data/output/'+name+'.pkl')
pickleRT = "--- Pickle savetime: {} seconds ---".format(time.time() - start_time)
# print(df)


# df.to_csv("./trained_models/data.csv")
start_time = time.time()
df = pd.read_csv('../data/output/'+name+'.csv', header='infer')
csvRT = "--- CSV savetime: {} seconds ---".format(time.time() - start_time)
# df.columns = ['index', 'A', 'B']
# df = df.set_index('index')
# print(df)


with open('../data/output/' + name + 'time.txt', 'a') as f:
   f.write('\n')
   f.write("READING SPEED: \n")
   f.write(csvRT + '\n')
   f.write(pickleRT + '\n')
   f.write(HDFRT + '\n')
   f.write(featherRT + '\n')