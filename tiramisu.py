#!/usr/bin/env python3

# Import from files in the same directory
from network import train_model, print_confusion_matrix, print_statistics
from utils import load_text_from_save, load_data_test, load_data_text

# Import packages. A bare `pip3 install foo` should do the trick
import logging
import argparse
import os
import json
import signal, psutil
import shutil

from multiprocessing import Process
from flask import Flask, render_template, request
from os import listdir
from os.path import join
from keras.models import load_model

# ------------------------------------------------ LOGGER ------------------------------------------------

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# ------------------------------------------------ PARSER ------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', help='Option indicating that the model should be trained', action="store_true")
parser.add_argument('--test', help='Option indicating that the model should be tested', action="store_true")
parser.add_argument('--no_reports_saved', help='Option indicating that no reports should be created', action="store_true")
parser.add_argument('--save_dir', type=str, default='./save', help='Directory in which the LAST model should be saved.')
parser.add_argument('--reports_dir', type=str, default='logs', help='Directory in which the reports are located. Default "logs" means that they should be saved only with the logs.')
parser.add_argument('--newdata', help='Boolean indicating if we load the data from text files directly', action="store_true")
parser.add_argument('--showbest', help='Show the reports related to the best model obtained so far', action="store_true")
parser.add_argument('-m', type=str, default='', help='Message indicating the purpose of the training. It will be stored in the report file')


args = parser.parse_args()

# # -------------------------------------------- HYPERPARAMETERS -------------------------------------------

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

vect_size = len(alphabet)
batch_size = 128
filters = 256
kernel_size = [3, 2]
hidden_dims = 2048
epochs = 300
pool_size = 2
epoch_decay = 60
initial_lr = 0.001
decay_rate = 0.5
max_features = 376

target_names = []

hyperparameters = {'alphabet':alphabet, 'vect_size':vect_size, 'batch_size':batch_size, 'filters':filters, 'kernel_size':kernel_size, 
'hidden_dims':hidden_dims, 'epochs':epochs, 'pool_size':pool_size, 'epoch_decay':epoch_decay, 'initial_lr':initial_lr,
'decay_rate':decay_rate, 'max_features':max_features, 'target_names':target_names}

# -------------------------------------------- WEB SERVER INIT -------------------------------------------

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def network_launch():

	def kill_child_processes(parent_pid, sig=signal.SIGTERM):
		try:
			parent = psutil.Process(parent_pid)
		except psutil.NoSuchProcess:
			return
		children = parent.children(recursive=True)
		for process in children:
			process.send_signal(sig)

	kill_child_processes(os.getpid())

	data_vector = load_data_test(hyperparameters)
	print("Loading done")

	with open("static/resources/json/probabilities.json", "r") as file:
		values = json.loads(file.read())
		values[1]["state"] = ""
		values[1]["current_index"] = 0
		for i in range(len(values[0][0])):
			values[0][0][i]['value'] = 0
	
	with open("static/resources/json/probabilities.json", "w") as file:
		file.write(json.dumps(values))

	shutil.copy(join("Test", [f for f in listdir("Test") if '.txt' in f][0]), "static/resources/text_processed.txt")
	run_network(data_vector)
	return "Network launched"

# ------------------------------------- WEB SERVER MANAGEMENT ---------------------------

def testNetwork(args, data_vector):
	model = load_model(join(args.save_dir, 'last.hdf5'))
	print('\n Successfully loaded model from ' + join(args.save_dir, 'last.hdf5') + '\n')
	print_statistics(args, model, data_vector, log_dir, hyperparameters)

def run_network(data_vector):
	p2 = Process(target=testNetwork, args=(args, data_vector))
	p2.start()

def run_app():
	with open("static/resources/json/probabilities.json", "r") as file:
		values = json.loads(file.read())
		values[1]["state"] = ""

	with open("static/resources/json/probabilities.json", "w") as file:
		file.write(json.dumps(values))

	app.run(debug=True)

# ------------------------------------ MAIN ---------------------------------------------

if __name__ == '__main__':

	assert not (args.train == True and args.test == True), "Can't train and test at the same time."

	### Logging directory ###
	log_dir = './logs/CNN_MaxF' + str(max_features) + '_BS' + str(batch_size) + '_LenAlph' + str(len(alphabet))
	element_in_dir = []
	if os.path.isdir(log_dir):
		for element in listdir(log_dir):
			element_in_dir.append(int(element))
		element_in_dir.sort()
		if element_in_dir != []:
			log_dir += '/' + str(int(element_in_dir[-1]) + 1)
	else:
		log_dir += '/1'

	print('Logging directory : ' + log_dir)


	### Training ###
	if args.test == False:

		### New data ###

		if args.newdata == True:
			x_train, y_train, x_test, y_test = load_data_text(hyperparameters)

		### Old data ###
		else:
			x_train, y_train, x_test, y_test = load_text_from_save()

		print("Loading done")

		### Training the model
		if args.train == True:
			model = train_model(args, x_train, y_train, x_test, y_test, hyperparameters, log_dir)
		
		### Printing confusion matrix ###
		model = load_model(join(args.save_dir, 'last.hdf5'))
		print('\n Successfully loaded model from ' + join(args.save_dir, 'last.hdf5') + '\n')
		print("\n[Reports creation]")
		print_confusion_matrix(args, model, x_test, y_test, log_dir, hyperparameters)


	### Testing ###
	else:

		Process(target=run_app).start()