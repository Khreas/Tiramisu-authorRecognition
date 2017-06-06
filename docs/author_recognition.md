# Author Recognition

## Overview

The author recognition module is based on a convolutional network coupled with a Fully Connected network. It relies on the framework [Keras](https://keras.io) for the neural network implementation and on the library [Sklearn](https://scikit-learn.com) for the analysis of the results.

The aim of this module is simple: **being able to determine the author of a text accordingly to its writing style.**

In order to do that, we need to have a large dataset. Relatively to our tests, we advise you to have at least 3MB of raw text per author.

## Requirements
The required libraries / frameworks for the project are the following:

* [Tensorflow >= 1.0](https://www.tensorflow.org/install/)
	* Note: in order to run faster, please consider installing [CUDA >= 8.5](https://developer.nvidia.com/cuda-downloads) if your GPU supports it. The softwares included in the project can be up to 20x faster. In this case, please also install [CuDNN5.1](https://developer.nvidia.com/cudnn).
* [Keras >= 2.0](https://keras.io/#installation)
* Numpy >= 1.12
	* `sudo pip3 install numpy` will do the trick.
* Sklearn >= 0.18.1
	* `sudo pip3 install scikit-learn` will do the trick.
* Matplotlib >=  2.0.2
	* `sudo pip3 install matplotlib` will do the trick. However, if some error occurs, please consider using `sudo apt install python3-matplotlib`

## Model details

Our model is largely inspired by the paper [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). We adapted it to fit our purpose the best.

### Architecture

Our architecture is 8 layers deep, with 5 convolutional layers and 3 fully-connected layers.

Here is a summary of the described architecture:

|		Layer (type)          |	Output Shape             |Param # |   
|-----------------------------|--------------------------|--------|
|conv1d_1 (Conv1D)            |(None, 250, 256)          |46848   |
|max_pooling1d_1 (MaxPooling1)|(None, 125, 256)          |0       |
|conv1d_2 (Conv1D)            |(None, 123, 256)          |196864  | 
|max_pooling1d_2 (MaxPooling1)|(None, 61, 256)           |0       | 
|conv1d_3 (Conv1D)            |(None, 59, 256)           |196864  |  
|conv1d_4 (Conv1D)            |(None, 57, 256)           |196864  |  
|conv1d_5 (Conv1D)            |(None, 55, 256)           |196864  |
|max_pooling1d_3 (MaxPooling1)|(None, 27, 256)           |0       |  
|flatten_1 (Flatten)          |(None, 6912)              |0       |  
|dense_1 (Dense)              |(None, 1024)              |7078912 |  
|dropout_1 (Dropout)          |(None, 1024)              |0       |  
|dense_2 (Dense)              |(None, 1024)              |1049600 |  
|dropout_2 (Dropout)          |(None, 1024)              |0       |
|dense_3 (Dense)              |(None, 5)                 |5125    |

### Dataset and Input

The solution we've implemented is character-level based. This means that we encode each character and then group some of them together in a matrix and then give it to the network. The encoding used is the [one-hot encoding](https://en.wikipedia.org/wiki/One-hot). We've decided to only deal with the 26 letters of the alphabet, lowercased. Here is a convenient example:

	Alphabet : ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

	a -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	b -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	...
	y -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	z -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

Once our characters have been converted, we group them together. For instance, the sentence *I am the* will be encoded as:

	[[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #i
	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #a
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #m
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #t
	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #h
	[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] #e

Let's say that our dataset contains texts from 3 authors: Maupassant, Hugo and Verne. Those authors can be encoded just like the letters. For instance:

	[1,0,0] # Maupassant
	[0,1,0] # Hugo
	[0,0,1] # Verne

In the end, samples composing our dataset have the following shape:

	(Matrix of encoded letters, Encoded author)

	([[0,0,...,1,0], 
	  [0,0,...,0,0],
	       ...			# Matrix of letters     
	  [0,1,...,0,0],
	  [0,0,...,0,0]]
	  ,
	  [0,0,1])			# Author (also called target)

---

The dimension of a matrix is (*alphabetSize*, *maxFeature*). MaxFeature is the number of consecutive letters we give to the network. You can imagine that as a fixed visual field: the network will determine an author based on this number of consecutive characters only.

In the paper we've used as a basis, maxFeature was set to 1014. The value we've used has been obtained the following way:

	TiramisuMaxFeature = (PaperMaxFeature/PaperAlphabetLen) * TiramisuAlphabetLen
	(1014/77) * 26 = 376

According to our tests, this value seems the most adapted to our problem.

We have estimated that at least 3MB of raw txt files per author is required to ensure good results.

### Output

The last layer of the network will output a vector of size equal to the number of authors in the dataset. This vector will contain values ranged from 0 to 1, indicating the likelihood relative to each author. Here is an example of the output:

			   Auth1   Auth2  Auth3  Auth4  Auth5
	Results: [ 0.670 , 0.030, 0.050, 0.120, 0.130]

## Data processing

#### `load_data()`

**params:** None

**return:** *list*, a list containing respectively the training matrices, the training targets, the validation matrices and the validation targets.

<br>

#### **Behaviour**

This function will load the data and convert it to the format discussed [above](author_recognition.md#model-details).

In order to easily perform some tasks, the data will be stored as numpy arrays instead of simple lists.

---

## Model training

#### `train_model(x_train, y_train, x_test, y_test)`

**params:**

- **x_train**, *list of numpy arrays*, list of size (alphabet_length, maxfeature). This will be the matrices of the training set.
- **y_train**, *list of numpy arrays*, list representing the authors of the x_train matrices. This will be the targets of the training set.
- **x_test**, *list of numpy arrays*, list of size (alphabet_length, maxfeature). This will be the matrices of the test set.
- **y_test**, *list of numpy arrays*, list representing the authors of the x_test matrices. This will be the targets of the test set.

**return:** *Keras model object*, the compiled and trained model we've created

<br>

#### **Behaviour**

This function will build and train the network. A bunch of functionnality have been implemented, such as degressive learning rates, custom activation or early / manual stopping of the training.

All the configuration of the network can be done within this function. The values found in the github are considered as our standard.

For more information about the compilation, the effective training or the tuning of the model, please refer to the [Keras documentation](https://keras.io/getting-started/sequential-model-guide/).

---

## Analysis

#### `print_confusion_matrix(args, model, X_test, Y_test, save_dir)`

**params:**

- **args**, *list of strings*, a parsed version of the entries given in the command line. For more information, follow [this link](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args).
- **model**, *Keras model object*, the model we want to analyze
- **X_test**, *list of numpy arrays*, list of size (alphabet_length, maxfeature). This will be the matrices of the test set.
- **Y_test**, *list of numpy arrays*, list representing the authors of the x_test matrices. This will be the targets of the test set.
- **save_dir**, *string*, path to the directory in which a report will be created

**return:** *Keras model object*, the compiled and trained model we've created

<br>

#### **Behaviour**

This function will generate a classification report, a confusion matrix (that can be saved as a png file) as well as a report file, listing the architecture and the hyperparameters of the model.

Here is an example of a generated confusion matrix:

![Confusion Matrix](img/conv_example.png)

For more information about the classification report or confusion matrix creation, please refer to the [Sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).

---

## Miscellaneous

#### `get_auth_number()`

**params:** None

**return:** *int*, number of distinct authors found in the dataset

<br>

#### **Behaviour**

Simply iterate over the subfolders of *Text/* and count the number of folder found.  

---

#### `display_parameters()`

**params:** None

**return:** None

<br>

#### **Behaviour**

Display all the parameters of the model in the terminal.

**Deprecated function**. Might be enabled again when reworked.