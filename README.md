## Character-Level Recurrent Neural Network Text Generator

This repository contains a complete character-level text generation system that learns and mimics writing styles. By processing text one character at a time, the model automatically learns spelling, grammar, and structural cadence from scratch.

## Project Description

The system downloads a literature dataset (Shakespeare's works), maps each character to a numeric ID, and organizes the text into overlapping input/target sequences. It trains a Gated Recurrent Unit (GRU) neural network to predict the next character given the preceding sequence. Generated text is produced one character at a time using a temperature-adjustable sampling function, allowing output to range from predictable to highly creative.

## Technical Stack

* Language: Python 3.11
* Deep Learning Framework: TensorFlow, Keras
* Data Structures: NumPy

## Features

* Character-Level Processing: Learns structural patterns directly from raw characters instead of word tokens.
* GRU-Based Recurrent Network: Uses gated recurrent units to retain context across sequences.
* Embedding-Based Tokenization: Maps characters to dense vector representations via a trainable embedding layer, using `StringLookup` for character-to-ID conversion.
* Adjustable Creativity (Temperature Sampling): Alters randomness of generated text during inference.
* Checkpointing: Saves model weights after each epoch for resuming or evaluating training progress.
* Automated Workflow: Downloads data, builds the training dataset, trains the model, and generates text samples in a single run.

## How to Setup and Run

### 1. Installation

Clone the repository and install the required dependencies using your command terminal:
* python -m pip install tensorflow numpy

### 2. Execution

Run the script using your Python environment to train the network and generate text:
python main.py

Training checkpoints are saved to `./training_checkpoints/`. After training completes, the script generates and prints a sample of text starting with the seed string `"ROMEO:"`.

## Example Output

The model generates Shakespeare-style text, learning character names, dialogue formatting, and archaic sentence structure directly from the training data, without any hardcoded rules.

## Notes

* Training on CPU can be slow; a GPU-enabled environment (e.g., via WSL2 or a cloud notebook) is recommended for faster results.
* The number of training epochs, sequence length, and temperature value can be adjusted directly in `main.py` to experiment with output quality and creativity.
