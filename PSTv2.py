#!/usr/bin/env python3

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Layer
from keras.layers.normalization import BatchNormalization
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adadelta
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler, Callback
from keras.utils import plot_model, np_utils
from keras.initializers import RandomUniform, RandomNormal

from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from random import shuffle
from shutil import copy

import keras.backend as K
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import gzip
import pickle
import time
import argparse
import signal
import os
import sys
import numpy
import h5py

# ------------------------------------------------ PARSER ------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', help='Option indicating that the model should be trained', action="store_true")
parser.add_argument('--no_reports_saved', help='Option indicating that no reports should be created', action="store_true")
parser.add_argument('--save_dir', type=str, default='./save', help='Directory in which the LAST model should be saved.')
parser.add_argument('--reports_dir', type=str, default='logs', help='Directory in which the reports are located. Default "logs" means that they should be saved only with the logs.')
parser.add_argument('--newdata', help='Boolean indicating if we load the data from text files directly', action="store_true")
parser.add_argument('--showbest', help='Show the reports related to the best model obtained so far', action="store_true")
parser.add_argument('-m', type=str, default='', help='Message indicating the purpose of the training. It will be stored in the report file')


args = parser.parse_args()

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
hidden_dims = 2048
epochs = 300
pool_size = 2
epoch_decay = 60
initial_lr = 0.001
decay_rate = 0.5
max_features = 376

target_names = []


def get_auth_number():
    directory = "Text"
    count = 0
    for subdir in next(os.walk(directory))[1]:
        if len(os.listdir(os.path.join(directory, subdir))) > 0:
            count = count+1
    return count

def load_data_save():

    with gzip.GzipFile(os.path.join('Text', 'formatted_data.pkl.gzip'), 'rb') as pkl_file:
      x_train, y_train, x_test, y_test = (pickle.load(pkl_file))

    return [x_train, y_train, x_test, y_test]

def hotToString(matrix=None, author=None):

    if matrix != None:
        sentence = ''
        for element in matrix:
            sentence += alphabet[list(element).index(1)]
    
    if author != None:
        author = target_names[list(author).index(1)]

    return (sentence, author)

def load_data_text():
    # The following loads and format the data stored in the folder named "Text"
    # The architecture must be the following:
    # Text --| Author1 --| Result --| input_*.txt
    #        | Author2 --| Result --| input_*.txt
    #        | Author3 --| Result --| input_*.txt

    print('\n[Loading data]')

    directory = "Text"

    data_vector = []
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []    

    count_author = -1
    letter_vector = [0] * len(alphabet)
    example_vector = []

    for subdir in next(os.walk(directory))[1]:
        if os.listdir(os.path.join(directory, subdir)):
            count_author = count_author + 1
            target_names.append(subdir)
        for file in os.listdir(os.path.join(directory, subdir, 'Result')):
            if 'input_' in file:
                i = 1
                with open(os.path.join(directory, subdir, 'Result', file), "r") as text:
                    target = count_author
                    example_vector = []
                    for line in text:
                        for character in line.lower():
                            if character in alphabet:
                                letter_vector = [0] * len(alphabet)
                                letter_vector[alphabet.index(character)] = 1
                                example_vector.append(letter_vector)
                                if (i%max_features) == 0:
                                    data_vector.append((numpy.array(example_vector), target))
                                    example_vector = []
                                i+=1

    shuffle(data_vector)

    for element in data_vector:
      train_set_x.append(element[0])
      train_set_y.append(element[1])

    dim = 0.7*len(data_vector)

    test_set_x = train_set_x[int(dim):]
    test_set_y = train_set_y[int(dim):]
    train_set_x = train_set_x[:int(dim)]
    train_set_y = train_set_y[:int(dim)]

    train_set_x = sequence.pad_sequences(train_set_x)
    test_set_x = sequence.pad_sequences(test_set_x)
    train_set_y = to_categorical(train_set_y, get_auth_number())
    test_set_y = to_categorical(test_set_y, get_auth_number())

    # for idx, element in enumerate(test_set_x):
    #     for idx_val, val in enumerate(element):
    #         if val == 0:
    #             test_set_x[idx][idx_val] = -1

    # for idx, element in enumerate(train_set_x):
    #     for idx_val, val in enumerate(element):
    #         if val == 0:
    #             test_set_x[idx][idx_val] = -1

    print('\n[Loading data : done]')

    with gzip.GzipFile(os.path.join('Text', 'formatted_data.pkl.gzip'), 'wb') as pkl_file:
      pickle.dump((train_set_x, train_set_y, test_set_x, test_set_y), pkl_file)


    rval = [train_set_x, train_set_y, test_set_x, test_set_y]

    return rval

def print_confusion_matrix(args, model, X_test, Y_test, save_dir):

    # Plot the confusion matrix in a neat way ; can be saved as a PNG file afterwards
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, precision=2):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = numpy.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            numpy.set_printoptions(precision=precision)
            cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
            for idx_vec, vector in enumerate(cm):
                for idx_element, element in enumerate(vector):
                    cm[idx_vec][idx_element] = round(element, 2)
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Initialization of the names of the authors
    if target_names == []:
        for subdir in next(os.walk("Text"))[1]:
            if os.listdir(os.path.join("Text", subdir)):
                target_names.append(subdir)

    def predictAndGetBestWorstBatch(matrices, authors):

        # In order: sentence, author, accuracy
        best_batch = [[], [], 0]
        worst_batch = [[], [], 1]

        y_pred = []

        # For each matrix of the input we'll predict its class and determine if it's the best / worst sample
        # we've dealt with so far
        for idx, element in enumerate(matrices):

            matrix_to_send = []
            matrix_to_send.append(element)

            matrix_to_send = numpy.array(matrix_to_send)

            b_pred = model.predict_classes(matrix_to_send, verbose=0)
            sys.stdout.write("\r" + str(idx) + ' / ' + str(len(matrices)))
            sys.stdout.flush()
            y_pred.append(b_pred[0])

            f = K.function([model.get_input(train=False)], [layer.get_output(train=False)])

            if max(list(model.layers[-1].output)) > best_batch[2]:
                best_batch[0], best_batch[1] = hotToString(matrices[idx], authors[idx])
                best_batch[2] = max(list(model.layers[-1].output))

            elif max(list(model.layers[-1].output)) < worst_batch[2]:
                worst_batch[0], worst_batch[1] = hotToString(matrices[idx], authors[idx])
                worst_batch[2] = max(list(model.layers[-1].output))

        sys.stdout.write("\n")
        return (best_batch, worst_batch, y_pred)

    # y_pred = model.predict_classes(X_test)
    best, worst, y_pred = predictAndGetBestWorstBatch(X_test, Y_test)

    print("\n\n     [Best]")
    print(best)

    print("\n\n     [Worst]")
    print(worst)
    
    print("\n\n     [Predictions]")
    print(y_pred)
    print("\n\n     [Expected]")
    print(Y_test)

    print("\n\n     [Reports]")
    print("             [Classification Report]\n")
    print(classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=target_names))
    print("             [Normalized confusion matrix]\n")
    conf_mat = confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred)
    plt.figure()
    plot_confusion_matrix(conf_mat, classes = target_names, normalize = True, title='Normalized confusion matrix', precision=3)
    plt.show()


    if args.no_reports_saved == False:

        filename = 'report_' + time.strftime("%d_%m_%Y_") + time.strftime("%H:%M:%S") + '.rep'

        if args.train == False:
            save_dir = args.save_dir

        orig_stdout = sys.stdout
        with open(os.path.join(save_dir, filename), 'w') as output:
            sys.stdout = output
            print("Training : " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + '\n\n')
            print(args.m)
            print("[ARCHITECTURE]\n\n")
            print(model.summary(), '\n\n')
            print('\n\n[CLASSIFICATION REPORT]\n\n',classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=target_names))
            print('\n\n[CONFUSION MATRIX]\n\n', confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred))
            print('\n\n[HYPERPARAMETERS]', '\n\nInitial learning rate : ' + str(initial_lr) + '\nLast learning rate : ' + str(K.get_value(model.optimizer.lr)), '\nDecay of ' + str(decay_rate) + ' every ' + str(epoch_decay) + ' epochs',
                  '\nMax_feature : ' + str(max_features), '\nVector size : ' + str(vect_size), '\nAlphabet : ' + str(alphabet), '\nHidden dim : ' + str(hidden_dims),
                  '\nEpochs : ' + str(epochs), '\nFilters : ' + str(filters), '\nRandom type : RandomUniform')
        sys.stdout = orig_stdout

def train_model(x_train, y_train, x_test, y_test):


    def signal_handler(signal, frame):
        model.stop_training = True


    signal.signal(signal.SIGINT, signal_handler)

    print('\n[Building Model]')
    model = Sequential()

    random_uni = RandomNormal(mean = 0.0, stddev = 0.05, seed = None)

    def custom_sigmoid_activation(x):
        return 1.7159*K.tanh(2/3*x)

    # We follow the architecture given here : https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
    model.add(Conv1D(filters,
                     kernel_size[1],
                     kernel_initializer=random_uni,
                     input_shape=(max_features,vect_size)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling1D(pool_size = 2, strides=None))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     kernel_initializer=random_uni,
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size = 2, strides=None))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     kernel_initializer=random_uni,
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     kernel_initializer=random_uni,
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(filters,
                     kernel_size[1],
                     kernel_initializer=random_uni,
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size = 2, strides=None))

    model.add(Flatten())

    model.add(Dense(hidden_dims, kernel_initializer=random_uni))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims, kernel_initializer=random_uni))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(get_auth_number(), kernel_initializer=random_uni, activation='softmax'))

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

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_accuracy'])

    plot_model(model, to_file='model.png')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = (x_test, y_test),
              shuffle=True,
              callbacks=[checkpointer, tensorVizualisation])


    copy(log_dir + '/model.hdf5', args.save_dir + '/last.hdf5')

    return model
    
if __name__ == '__main__':

    # Loading and clearing the texts is relatively long -> we load the data from a file in which we have
    # already stored our data, correctly formated
    if args.newdata == True:
      x_train, y_train, x_test, y_test = load_data_text()

    else:
      x_train, y_train, x_test, y_test = load_data_save()

    print("Loading done : " + str(len(x_train)))

    log_dir = './logs/CNN_MaxF' + str(max_features) + '_BS' + str(batch_size) + '_LenAlph' + str(len(alphabet))
    element_in_dir = []
    if os.path.isdir(log_dir):
        for element in os.listdir(log_dir):
            element_in_dir.append(int(element))
        element_in_dir.sort()
        if element_in_dir != []:
            log_dir += '/' + str(int(element_in_dir[-1]) + 1)
    else:
        log_dir += '/1'

    print('Logging directory : ' + log_dir)

    if args.train == True:
        model = train_model(x_train, y_train, x_test, y_test)

    if args.showbest == True:
        model = load_model(os.path.join('best', '75', 'model.hdf5'))  
        print('\n Successfully loaded model from ' + os.path.join('best', '75', 'model.hdf5') + '\n')
  
    else:
        model = load_model(os.path.join(args.save_dir, 'last.hdf5'))
        print('\n Successfully loaded model from ' + os.path.join(args.save_dir, 'last.hdf5') + '\n')

    
    print('\n Successfully loaded model from ' + os.path.join(args.save_dir, 'last.hdf5') + '\n')

    print("\n[Reports creation]")
    print_confusion_matrix(args, model, x_test, y_test, log_dir)
