## Character-Level Recurrent Neural Network Text Generator
This repository contains a complete character-level text generation system that learns and mimics writing styles. By processing text one character at a time, the model automatically learns spelling, grammar, and structural cadence from scratch. [1] 
## Project Description
The system downloads a literature dataset, cleans the text, and vectorizes it into binary matrices. It trains a Long Short-Term Memory neural network to predict the next character based on a forty-character history window. A custom temperature sampling function allows you to adjust the text generation output from predictable to highly creative. [2] 
## Technical Stack

* Language: Python 3.11
* Deep Learning Framework: TensorFlow, Keras
* Mathematics and Data Structures: NumPy

## Features

* Character Level Processing: Learns structural layout directly from raw characters instead of word tokens.
* Long Short-Term Memory Network: Utilizes recurrent memory cells to retain long-range content context.
* One Hot Vectorization: Transforms raw text characters into structured boolean arrays.
* Adjustable Creativity Dial: Uses temperature calculations to alter text randomness during inference.
* Automated Workflow: Fetches data, tokenizes, trains, saves the model file, and outputs text samples.

## How to Setup and Run## 1. Installation
Clone the repository and install the required dependencies using your command terminal:

python -m pip install tensorflow numpy

## 2. Execution
Run the script using your Python environment to train the network and generate text:

python main.py
