import os
# Allows to not  display the debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import chess

from keras.models import Sequential
from keras.layers import Dense

import learning
from learning import learning
import shutil

def environment_info():
    # Display the version of tensorflow (this code is valid for 2.5.0)
    print("tensorflow version:", tf.__version__)

    # Display the version of python chess (this code is valid for 1.6.1)
    print("python-chess version:", chess.__version__)

    # Display the number of GPUs used by the program
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    # Print the number of CPUs in the system
    print("Number of CPUs in the system:", os.cpu_count())
    print()

def environment_init():
    # Delete files and folders
    if os.path.exists("debug_file"):
        shutil.rmtree("debug_file")

    if os.path.exists("loss_evolution"):
        shutil.rmtree("loss_evolution")

    # Created files and folders
    os.mkdir("debug_file")
    os.mkdir("loss_evolution")


def make_model():
    # define the keras model
    model = Sequential()
    model.add(Dense(128, activation='LeakyReLU'))
    model.add(Dense(500, activation='LeakyReLU'))
    model.add(Dense(1000, activation='LeakyReLU'))
    model.add(Dense(1000, activation='LeakyReLU'))
    model.add(Dense(1000, activation='LeakyReLU'))
    model.add(Dense(1000, activation='LeakyReLU'))
    model.add(Dense(500, activation='LeakyReLU'))
    model.add(Dense(200, activation='LeakyReLU'))
    model.add(Dense(1, activation='LeakyReLU'))

    # compile the keras model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    return model


if __name__ == '__main__':
    environment_info()
    environment_init()

    number_cpu_use = 5
    number_parallel_game = 2400
    standard_starting_chess_position = "8/pppppppp/8/8/8/8/PPPPPPPP/8" #"r6r/pppppppp/8/8/8/8/PPPPPPPP/R6R"

    model = make_model()
    learning(starting_position=standard_starting_chess_position,
             model=model,
             number_parallel_game=number_parallel_game,
             number_repetitions=400,
             coef_exp_int=0.3,
             coef_expectation=0.4,
             verbose=["time_learning", "dataset_shape", "save_history"])    #, "save_picture_dataset"