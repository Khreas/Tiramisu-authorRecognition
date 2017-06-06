# Getting started with the author recognition software

*The dataset used for the following can be found [here](resources.md#Dataset). It contains texts from Balzac, France, Hugo, Maupassant, Proust, Verne and Zola.*

## What can it do ?

The building of this tool is based on the article [Character-level Convolutional Networks for Text
Classification]() by X. Zhiang and al. Their model was used on various dataset in order to fulfill various tasks, such as Image Classification or Polarity Analysis. 

Tiramisu's model has been optimized for the Author Classification. But what does it mean ?

Given a set of authors, we have trained a neural network to classify texts according to the set of authors. In other words, the network will take a text of length >= 376 characters as an input and will try to determine the author of the set whom most likely wrote it. Currently, its accuracy reaches 80%.


## What is in the dataset ?

As stated above, the network needs to be trained with a group of texts from the authors chosen.

The dataset has been created with the [crawler](crawler_guide.md) of the project Tiramisu. Basically, it is a structure composed of text files containing novels or short stories concatenated together. In this structure, each author has one input file that can be found under Text/Author/Result/inputfile.txt.

The program will run ONLY if the following architecture is followed:

	Text>|----------Author1>|Result>|input_auth1.txt
		 |----------Author2>|Result>|input_auth2.txt
		 |----------Author3>|Result>|input_auth3.txt
	classifier.py
	

## How to use it ?

If you want to try out our solution on our dataset, no modification has to be done. Simply launch the following command:

`python3 classifier.py`

---

If you want to try out our network on a given input and see who most likely wrote it, simply launch the following command:

`python3 classifier.py --analyze FILE_TO_ANALYSE`

---

If you want to try out our network on your own data, you will have to load your data and train the network first. To do that, launch the following command:

`python3 classifier.py --train --newdata`

**[CAUTION]** This can take a LONG amount of time.



---

Need a more concrete / detailed example ? [Follow me !](example_auth.md)