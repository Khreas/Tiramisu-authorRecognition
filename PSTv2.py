#!/usr/bin/env python3

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adadelta
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler, Callback
from keras.utils import plot_model
from keras.initializers import RandomUniform, TruncatedNormal

from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from random import shuffle

import keras.backend as K
import time
import argparse
import signal
import os
import sys
import numpy

# ------------------------------------------------ PARSER ------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', help='Option indicating that the model should be trained', action="store_true")
parser.add_argument('--no_reports_saved', help='Option indicating that no reports should be created', action="store_true")
parser.add_argument('--save_dir', type=str, default='./save', help='Directory in which the model should be saved')
parser.add_argument('--reports_dir', type=str, default='logs', help='Directory in which the reports are located. Default "logs" means that they should be saved only with the logs.')
parser.add_argument('--verbosity', type=str, default='medium', help='Level of verbosity - Can be low, medium or high')
parser.add_argument('--sliding_window', help='Option indicating if the data should be formatted according to a sliding window', action="store_true")
parser.add_argument('--sliding_window_size', type=int, default=10, help='Size of the sliding window')

args = parser.parse_args()

assert args.verbosity.lower() == 'low' or args.verbosity.lower() == 'medium' or args.verbosity.lower() == 'high', "Argument 'verbosity' has an unknown value. Please pick 'low', 'medium' or 'high'."

# -------------------------------------------- HYPERPARAMETERS--------------------------------------------

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ]
filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\nàâäéèëêîìíïôöûüùïöüäçæñ'
# accents = ['à', 'â', 'ä', 'é', 'è', 'ë', 'ê', 'î', 'ì', 'í', 'ï', 'ô', 'ö', 'û', 'ü', 'ù', 'ï', 'ö', 'ü', 'ä', 'ç', 'æ', 'ñ']
# symbols = ['!', "'", '"', '#', '$', '%', '&', '(', ')', '[', ']', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '—', '«', '»', '`', '_']
# numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# alphabet += accents + symbols + numbers


vect_size = len(alphabet)
batch_size = 128
filters = 256
kernel_size = [7, 3]
hidden_dims = 1024
epochs = 300
pool_size = 2
epoch_decay = 60
initial_lr = 0.01
decay_rate = 0.5
max_features = 376

def display_parameters(model, x_train, y_train, x_test, y_test):

    # Data information

    if not args.verbosity == 'low':
    
        print(
          "==========================================================\n" +
          "|                   General information                  |\n" +
          "==========================================================")

        print(
          "|                         Dataset                        |\n" +
          "----------------------------------------------------------\n" +
          "|   Nb authors : %d                                       |" %get_auth_number())
        if args.sliding_window == True:
            print(
          "|    Input type : vector of letters                      |")
        else:
            print(
          "|    Input type : vector of letters, sliding window      |\n" +
          "|    Size of sliding window: %d                          |\n" %int(args.sliding_window_size) +
          "|    Step :                  1                           |")

        if args.verbosity == 'high':
            print(
              "|                                                        |\n" +
              "|    Does the train_set contains all author ?            |")
            if all(value in y_train for value in range(get_auth_number())):
                print(
                  "|        Yes                                             |")
            else:
                print(
                  "|        No                                              |")
            print(
              "|    Does the test_set contains all author ?             |")
            if all(value in y_test for value in range(get_auth_number())):
                print(
                  "|        Yes                                             |")
            else:
                print(
                  "|        No                                              |")  


    # Model information

        print(
          "----------------------------------------------------------\n" +
          "|                         Model                          |\n" +
          "----------------------------------------------------------\n" +
          "|    Train set                                           |\n" +
          "|        Shapes:  %s, %s           |\n" %(str(x_train.shape), str(y_train.shape)) +
          "|    Test set                                            |\n" +
          "|        Shapes:  %s, %s            |\n" %(str(x_test.shape), str(y_test.shape)) +
          "|                                                        |\n" +
          "|    Model shape                                         |\n" +
          "|        Input :  %s                        |\n" %str(model.input_shape) +
          "|        Output : %s                              |\n" %str(model.output_shape) +
          "----------------------------------------------------------")

    
    print("\n\n")

    if args.verbosity == 'medium':
        model.summary()
        print("\n\n")

    if args.verbosity == 'high':

        print("x_train : \n\n", x_train)
        print("\ny_train :\n\n", y_train)
        print("\n\nx_test : \n\n", x_test)
        print("\ny_test :\n\n", y_test)


def get_auth_number():
    directory = "Text"
    count = 0
    for subdir in next(os.walk(directory))[1]:
        if len(os.listdir(os.path.join(directory, subdir))) > 0:
            count = count+1
    return count

def load_data():
    print('\n[Loading data]')

    # Load and separate the dataset in 2 different vectors : test_set, train_set

    slidingWindow = args.sliding_window
    slidingWindowSize = args.sliding_window_size

    directory = "Text"

    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    

    count_author = -1
    letter_vector = [0] * len(alphabet)
    target_vector = []
    example_vector_train = []
    example_vector_test = []

    target_vector_train = []
    target_vector_test = []

    

    if not slidingWindow:

        for subdir in next(os.walk(directory))[1]:
            if os.listdir(os.path.join(directory, subdir)):
                count_author = count_author + 1
            for file in os.listdir(os.path.join(directory, subdir)):
                if '_input' in file:
                    i = 1
                    with open(os.path.join(directory, subdir, file), "r") as text:
                        target = count_author
                        example_vector_train = []
                        for line in text:
                            for word in line:
                                for character in word:
                                    if character in alphabet:
                                        letter_vector = [0] * len(alphabet)
                                        letter_vector[alphabet.index(character)] = 1
                                        example_vector_train.append(letter_vector)
                                        if (i%max_features) == 0:
                                            train_set_x.append(numpy.array(example_vector_train))
                                            train_set_y.append(target)
                                            example_vector_train = []
                                        i+=1

                elif '_test' in file:
                    i = 1
                    with open(os.path.join(directory, subdir, file), "r") as text:
                        target = count_author
                        example_vector_test = []
                        for line in text:
                            for word in line:
                                for character in word:
                                    if character in alphabet:
                                        letter_vector = [0] * len(alphabet)
                                        letter_vector[alphabet.index(character)] = 1
                                        example_vector_test.append(letter_vector)
                                        if (i%max_features) == 0:
                                            test_set_x.append(numpy.array(example_vector_test))
                                            test_set_y.append(target)
                                            example_vector_test = []
                                        i+=1
    else:

        memory = []
        letter_nb = 0

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

    forbidden_vec = [0]*len(alphabet)
    forbidden_vec = numpy.array(forbidden_vec)
    #F11
    assert all(len(element) == max_features for element in train_set_x), "Length of train_set_x matrices isn't equal to max_feature !"
    assert all(len(element) == max_features for element in test_set_x), "Length of test_set_x matrices isn't equal to max_feature !"

    train_set_x = numpy.array(train_set_x)
    train_set_y = numpy.array(train_set_y)
    test_set_x = numpy.array(test_set_x)
    test_set_y = numpy.array(test_set_y)

    

    print('\n[Loading data : done]')

    rval = [train_set_x, train_set_y, test_set_x, test_set_y]

    return rval

def print_confusion_matrix(args, model, X_test, Y_test, save_dir):

    y_pred = model.predict_classes(X_test)
    print("\n\n     [Predictions]")
    print(y_pred)
    print("\n\n     [Expected]")
    print(Y_test)

    target_names = ['Zola', 'Flaubert', 'Verne', 'Maupassant', 'Hugo']
    print("\n\n     [Reports]")
    print("             [Classification Report]\n")
    print(classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=target_names))
    print("             [Confusion matrix]\n")
    print(confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred))

    if args.no_reports_saved == False:

        filename = 'report_' + time.strftime("%d_%m_%Y_") + time.strftime("%H:%M:%S") + '.rep'

        if args.train == False:
            save_dir = args.save_dir

        orig_stdout = sys.stdout
        with open(os.path.join(save_dir, filename), 'w') as output:
            sys.stdout = output
            print("Training : " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + '\n\n')
            print("[ARCHITECTURE]\n\n")
            print(model.summary(), '\n\n')
            print('\n\n[CLASSIFICATION REPORT]\n\n',classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=target_names))
            print('\n\n[CONFUSION MATRIX]\n\n', confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred))
            print('\n\n[HYPERPARAMETERS]', '\n\nInitial learning rate : ' + str(initial_lr) + '\nLast learning rate : ' + str(K.get_value(model.optimizer.lr)), '\nDecay of ' + str(decay_rate) + ' every ' + str(epoch_decay) + ' epochs',
                  '\nMax_feature : ' + str(max_features), '\nVector size : ' + str(vect_size), '\nAlphabet : ' + str(alphabet), '\nHidden dim : ' + str(hidden_dims),
                  '\nEpochs : ' + str(epochs), '\nFilters : ' + str(filters), '\nRandom type : RandomUniform')
        sys.stdout = orig_stdout

def train_model(x_train, y_train, x_test, y_test):

    print('\n[Building Model]')
    model = Sequential()

    random_uni = TruncatedNormal(mean = 0.0, stddev = 0.05, seed = None)

    model.add(Conv1D(filters,
                     kernel_size[0],
                     activation='relu',
                     kernel_initializer=random_uni,
                     input_shape=(max_features,vect_size)))
    
    model.add(MaxPooling1D(pool_size = 2, strides=None))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))

    model.add(MaxPooling1D(pool_size = 2, strides=None))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     activation='relu',
                     kernel_initializer=random_uni,
                     strides=1))

    # # we use max pooling:
    model.add(MaxPooling1D(pool_size = 2, strides=None))


    model.add(Flatten())

    model.add(Dense(hidden_dims, activation='sigmoid', kernel_initializer=random_uni))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims, activation='sigmoid', kernel_initializer=random_uni))
    model.add(Dropout(0.5))
    layer = Dense(get_auth_number(), activation='softmax', kernel_initializer=random_uni)
    model.add(layer)

    sgd = SGD(lr=initial_lr, momentum=0.9)

    last_val_loss = float("inf")

    class BoldScheduler(Callback):
        def __init__(self):
            self.last_val_loss = float("inf")

        def on_epoch_end(self, epoch, logs={}):
                
            curr_val_loss = logs.get('val_loss')
            lr = K.get_value(model.optimizer.lr)

            if(self.last_val_loss > curr_val_loss):
                K.set_value(model.optimizer.lr, lr*1.1)
                print("lr changed from {:.6f} to {:.6f}".format(lr, K.get_value(model.optimizer.lr)))
            elif curr_val_loss - self.last_val_loss > 0.001:
                K.set_value(model.optimizer.lr, lr*0.7)
                print("lr changed from {:.6f} to {:.6f}".format(lr, K.get_value(model.optimizer.lr)))

            self.last_val_loss = curr_val_loss
            return

    def scheduler(epoch):
        
        if epoch%epoch_decay == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr*decay_rate)
            print("lr changed to {}".format(lr*decay_rate)) 
        return K.get_value(model.optimizer.lr)


    checkpointer = ModelCheckpoint(filepath=log_dir + '/model.hdf5', verbose=1, save_best_only=True, monitor='val_categorical_accuracy')
    tensorVizualisation = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto')
    lr_decay = LearningRateScheduler(scheduler)
    bold_decay = BoldScheduler()

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_accuracy'])

    plot_model(model, to_file='model.png')

    display_parameters(model, x_train, y_train, x_test, y_test)


    def signal_handler(signal, frame):
        model.stop_training = True


    signal.signal(signal.SIGINT, signal_handler)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = (x_test, y_test),
              shuffle=True,
              callbacks=[checkpointer, tensorVizualisation])

    model.save(args.save_dir + '/last.hdf5')

    return model
    
if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_data()

    x_train = sequence.pad_sequences(x_train)
    x_test = sequence.pad_sequences(x_test)
    y_train = to_categorical(y_train, get_auth_number())
    y_test = to_categorical(y_test, get_auth_number())

    log_dir = './logs/CNN_MaxF' + str(max_features) + '_BS' + str(batch_size) + '_LenAlph' + str(len(alphabet))
    element_in_dir = []
    if os.path.isdir(log_dir):
        for element in os.listdir(log_dir):
            element_in_dir.append(element)
        if element_in_dir != []:
            log_dir += '/' + str(int(element_in_dir[-1]) + 1)
    else:
        log_dir += '/1'

    print('Logging directory : ' + log_dir)

    if args.train == True:
        model = train_model(x_train, y_train, x_test, y_test)
    model = load_model(os.path.join(args.save_dir, 'last.hdf5'))
    print('\n Successfully loaded model from ' + os.path.join(args.save_dir, 'last.hdf5') + '\n')
    if not args.train:
        display_parameters(model, x_train, y_train, x_test, y_test)

    print("\n[Reports creation]")
    print_confusion_matrix(args, model, x_test, y_test, log_dir)