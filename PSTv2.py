'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adadelta
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from keras.initializers import RandomUniform

from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from random import shuffle

import argparse
import os
import numpy

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def get_auth_number():
    directory = "Text"
    count = 0
    for subdir in next(os.walk(directory))[1]:
        if len(os.listdir(os.path.join(directory, subdir))) > 0:
            count = count+1
    return count

def load_data():
    print('\n[Data Loading]')

    # Load and separate the dataset in 3 different vectors : test_set, train_set and valid_set

    slidingWindow = False
    slidingWindowSize = 10

    directory = "Text"

    train_set = []
    test_set = []
    validation_set = []
    test_files = []
    validation_files = []

    print("\n    [Authors]")
    print("       [Number of authors]   %d" %get_auth_number())

    count_author = -1
    letter_vector = [0] * len(alphabet)
    target_vector = []
    example_vector_train = []
    example_vector_valid = []
    example_vector_test = []

    target_vector_train = []
    target_vector_valid = []
    target_vector_test = []

    print("\n    [Texts]")

    if not slidingWindow:

        print("       [Input Type]          Vectors of letters\n")

        for subdir in next(os.walk(directory))[1]:
            if os.listdir(os.path.join(directory, subdir)):
                count_author = count_author + 1
            for file in os.listdir(os.path.join(directory, subdir)):
                if '_input' in file:
                    with open(os.path.join(directory, subdir, file), "r") as text:
                        target = count_author
                        for line in text:
                            for word in line:
                                for character in word:
                                    if character in alphabet:
                                        letter_vector = [0] * len(alphabet)
                                        letter_vector[alphabet.index(character)] = 1
                                        example_vector_train.append(letter_vector)
                                        target_vector_train.append(target)
                elif '_test' in file:
                    with open(os.path.join(directory, subdir, file), "r") as text:
                        target = count_author
                        for line in text:
                            for word in line:
                                for character in word:
                                    if character in alphabet:
                                        letter_vector = [0] * len(alphabet)
                                        letter_vector[alphabet.index(character)] = 1
                                        example_vector_test.append(letter_vector)
                                        target_vector_test.append(target)

    else:

        print("       [Input Type]          Vectors of letters, based on sliding window")

        memory = []
        letter_nb = 0

        print("       [Input type]          Size of the window: %d" %slidingWindowSize)
        print("       [Input type]          Step: 1")

        for subdir in next(os.walk(directory))[1]:
            if os.listdir(os.path.join(directory, subdir)):
                count_author = count_author + 1
            for file in os.listdir(os.path.join(directory, subdir)):
                if '_input' in file:
                    with open(os.path.join(directory, subdir, file), "r") as text:
                        target = count_author
                        for line in text:
                            for word in line:
                                for character in word:
                                    if character in alphabet:
                                        if len(memory) < slidingWindowSize:
                                            memory.append(character)
                                        else:
                                            for read_char in memory:
                                                letter_vector[alphabet.index(read_char)] = 1
                                            example_vector_train.append(letter_vector)
                                            target_vector_train.append(target)
                                            letter_vector = [0] * len(alphabet)
                                            memory = memory[1:]
                                            memory.append(character)
                elif '_test' in file:
                    with open(os.path.join(directory, subdir, file), "r") as text:
                        target = count_author
                        for line in text:
                            for word in line:
                                for character in word:
                                    if character in alphabet:
                                        if len(memory) < slidingWindowSize:
                                                memory.append(character)
                                        else:
                                            for read_char in memory:
                                                letter_vector[alphabet.index(read_char)] = 1
                                            example_vector_test.append(letter_vector)
                                            target_vector_test.append(target)
                                            letter_vector = [0] * len(alphabet)
                                            memory = memory[1:]
                                            memory.append(character)                                                                    

    train_set_x = numpy.array(example_vector_train)
    train_set_y = numpy.array(target_vector_train)
    test_set_x = numpy.array(example_vector_test)
    test_set_y = numpy.array(target_vector_test)

    train_set_x = numpy.array(train_set_x)
    train_set_y = numpy.array(train_set_y)
    test_set_x = numpy.array(test_set_x)
    test_set_y = numpy.array(test_set_y)

    print('\n[/Data Loading]')
    print('\n[Data Information]')


    print("\n     Does train_set contains all author ?")
    if all(value in train_set_y for value in range(get_auth_number())):
        print("         Yes")
    else:
        print("         No")
    
    print("\n     Does test_set contains all author ?")
    if all(value in test_set_y for value in range(get_auth_number())):
        print("         Yes")
    else:
        print("         No")

    rval = [train_set_x, train_set_y, test_set_x, test_set_y]

    return rval

def print_confusion_matrix(model, X_test, Y_test):

    y_pred = model.predict_classes(X_test)
    print(y_pred)

    target_names = ['Zola', 'Flaubert', 'Verne', 'Maupassant', 'Hugo']
    print("\n\n     [Reports]")
    print("             [Classification Report]\n")
    print(classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=target_names))
    print("             [Confusion matrix]\n")
    print(confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred))

    print(model.get_weights())


def train_model(x_train, y_train, x_test, y_test):
    # set parameters:
    max_features = 1014
    vect_size = 26
    batch_size = 128
    filters = 256
    kernel_size = [7, 3]
    hidden_dims = 1024
    epochs = 30
    pool_size = 2

    print("\n     [Dataset information]")
    print("         [Shape]")
    print("             [Train set]\n")
    print("X :\n", x_train)
    print("\nY :\n", y_train)
    print("\n\n             Shapes : ", x_train.shape, y_train.shape)
    print("\n             [Test set]\n")
    print("X :\n", x_test)
    print("\n Y :\n", y_test)
    print("\n\n             Shapes : ", x_test.shape, y_test.shape)

    print("\n         [Sequence]")
    print("\n             ", len(x_train), 'train sequences')
    # print(len(x_test), 'test sequences')

    print('\n              Pad sequences (samples x time)')

    # print('x_train shape:', x_train.shape)
    # print('x_test shape:', x_test.shape)

    print("\n[/Data Information]")
    print('\n[Building Model]')
    model = Sequential()

    random_uni = RandomUniform(minval=-1.0, maxval=1.0)

    # model.add(Embedding(1014,
    #                     get_auth_number(),
    #                     input_length=26))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size[0],
                     padding='same',
                     activation='relu',
                     kernel_initializer=random_uni,
                     input_shape=(26,1)))
    # we use max pooling:
    model.add(MaxPooling1D(pool_size = pool_size, strides=None))
    
    model.add(Conv1D(filters,
                     kernel_size[1],
                     padding='same',
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     padding='same',
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))
    model.add(Conv1D(filters,
                     kernel_size[1],
                     padding='same',
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     border_mode='same',
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))

    # # we use max pooling:
    model.add(MaxPooling1D(pool_size = 3, strides=None))


    model.add(Flatten())

    model.add(Dense(hidden_dims, activation='sigmoid', kernel_initializer=random_uni))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims, activation='sigmoid', kernel_initializer=random_uni))
    model.add(Dropout(0.5))
    model.add(Dense(get_auth_number(), activation='softmax', kernel_initializer=random_uni))

    sgd = SGD(lr=0.01, momentum=0.9)
    adadelta = Adadelta()

    print("     [Model shape]")
    print("\n       Model input shape : ", model.input_shape)
    print("       Model output shape : ", model.output_shape)
    print("\n     [Summary]")
    print("\n")
    model.summary()
    print('\n\n\n[/Data Information]')
    print('\n[Training Model]')

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
    tensorVizualisation = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    earlyStop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=1, verbose=1, mode='auto')
    plot_model(model, to_file='model.png')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[checkpointer, tensorVizualisation, earlyStop])

    model.save('PSTv2.h5')
    print("\n\n[/Training Model]")

    return model
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', type=bool, default=False, help='Boolean value representing whether the model should be trained or not')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory in which the savefile is located')

    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_data()

    x_train = sequence.pad_sequences(x_train)
    x_test = sequence.pad_sequences(x_test)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    y_train = to_categorical(y_train, get_auth_number())
    y_test = to_categorical(y_test, get_auth_number())

    if args.train == True:
        model = train_model(x_train, y_train, x_test, y_test)
    else:
        print("\n[Reports creation]")
        model = load_model(os.path.join(args.save_dir, 'PSTv2.h5'))
    
    print_confusion_matrix(model, x_train, y_train)