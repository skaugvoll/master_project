

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM # if on GPU version, CuDNNLSTM
import math

def mnist_example():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # print("X TRAIN SHAPE", x_train.shape)
    # # print(x_train[0])
    # print("X TRAIN [0] SHAPE ", x_train[0].shape)
    #
    #
    # print("Y TRAIN SHAPE", y_train.shape)
    # print("Y TRAIN [0]", y_train[0])
    # print("Y TRAIN [0] SHAPE", y_train[0].shape)

    ## X Train DATASET LOOKS LIKE
    '''
    60000 arrays / rows in one arary
    each array /rows has 28 rows in them with 28 rows
    '''

    ## Y Train DATASET LOOKS LIKE
    '''
    60000 arrays/rows in one array
    only one element per array / row (label)
    '''



    print(x_train.shape[1:])
    input_shape = x_train.shape[1:] # number of rows and features for each element to train on # mnist (28,28)


    # Noramlize the data!
    x_train = x_train / 255.0
    x_test = x_test / 255.0


    # BUILD THE MODEL
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_shape), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu', return_sequences=False)) # do not want to return_sequence, because the next layer is dense
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))

    number_of_classes = 10
    model.add(Dense(number_of_classes, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5) # decay is how much to lower the lr over time

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))



class LSTMTEST:
    def __init__(self):
        pass

    def train(self, training_data, epochs, batch_size, sequence_lenght, split):
        # print(training_data.shape)

        labels = training_data.iloc[:, [-1]].values
        # print(labels)

        # print(training_data.head(5))

        # remove unwanted columns / features
        training_data.drop(columns=['btemp', 'ttemp', 'label'], inplace=True)

        # We need to reshape the data from (rows, features) to (observations/row, rows pr observation, features )
        training_data = training_data.values
        training_data = training_data.reshape(training_data.shape[0], 1, training_data.shape[1])
        # print(training_data.shape)

        # Split into training and testing
        # print(training_data.shape[0])
        split_on = math.floor(training_data.shape[0] * split)
        # print(split_on)
        x_train = training_data[:split_on]
        x_test = training_data[split_on:]
        print("new shapes features train and test: ", x_train.shape, x_test.shape)

        y_train = labels[:split_on]
        y_test = labels[split_on:]
        print("new shapes labels train and test: ", y_train.shape, y_test.shape)

        # Should probably normalize features
        # ...
        # ...

        ## CREATE BATCHES
        batches = tf.data.Dataset.batch(x_train, batch_size=512)

        print("Created batches, not sure what to do with it")
        # Creating an iterator to access data in batch
        ##




        input_shape = x_train.shape[1:]  # number of rows and features for each element to train on # mnist (28,28)
        # print("INPUT SHAPE: ", input_shape)

        # build the model
        model = Sequential()
        model.add(CuDNNLSTM(12, input_shape=(input_shape), return_sequences=True))
        model.add(Dropout(0.2))

        model.add(CuDNNLSTM(12, return_sequences=False))  # do not want to return_sequence, because the next layer is dense
        model.add(Dropout(0.2))

        model.add(Dense(10, activation='relu'))

        number_of_classes = 3
        model.add(Dense(number_of_classes, activation='softmax'))

        opt = tf.keras.optimizers.Adam(lr=0.7, decay=1e-5)  # decay is how much to lower the lr over time

        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))



    def trainBatch(self, training_data, epochs, batch_size, sequence_lenght, split):
        # print(training_data.shape)

        labels = training_data.iloc[:, [-1]].values
        # print(labels)

        # print(training_data.head(5))

        # remove unwanted columns / features
        training_data.drop(columns=['btemp', 'ttemp', 'label'], inplace=True)

        # We need to reshape the data from (rows, features) to (observations/row, rows pr observation, features )
        training_data = training_data.values
        training_data = training_data.reshape(training_data.shape[0], 1, training_data.shape[1])
        # print(training_data.shape)




if __name__ == '__main__':
    print("Hello from testLSTM model file")