# DNA Sequence Classifier: User Manual

# Overview

The DNA Sequence Classifier is a Python script designed to classify DNA sequences as either "positive" or "negative" based on specific sequence patterns. This script processes input DNA sequences, makes predictions, and calculates the accuracy, precision, and recall of its predictions compared to the true labels provided in a separate file.

# Requirements
1. Python 3.6 or later
2. BioPython package
3. scikit-learn package
4. NumPy package

# Installation
Install Python 3.6 or later from the official website: https://www.python.org/downloads/
Install the required packages using pip:

pip install biopython scikit-learn numpy

# Usage
1. Save the script as intermediate.py.
2. Prepare an input file containing DNA sequences. The input file can be in plain text format with one sequence per line or in FASTA format.
3. Prepare a true labels file containing the true classification labels for each sequence in the input file. The true labels file should be in plain text format with one label per line, corresponding to the sequences in the input file.
4. Run the script from the command line with the following syntax:

python intermediate.py -f input_file -l true_labels_file

Replace input_file with the path to your input file and true_labels_file with the path to your true labels file.

# Example
For an input file sequences.txt containing the following sequences:

ATGCGTACGATCGATCGATCGTAGCTAC
CGTAGCTACGATCGATCGATCGTAGCTA

And a true labels file true_labels.txt containing the following labels:

positive
negative

# Run the script with the following command:

python dna_sequence_classifier.py -f sequences.txt -l true_labels.txt

The script will output the accuracy, precision, and recall of its predictions, as well as a detailed classification report:

Accuracy: 1.0
Precision: [1. 1.]
Recall: [1. 1.]
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00         1
    positive       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

