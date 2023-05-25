import argparse
import os
import unittest
from collections import Counter
from sklearn.metrics import classification_report
from Bio import SeqIO
import warnings
import numpy as np
warnings.filterwarnings('always')

# Function to predict the label for a given DNA sequence
def predict_label(sequence):
    if 'ATG' in sequence and 'TAA' in sequence:
        start_index = sequence.index('ATG')
        end_index = sequence.index('TAA')
        if end_index - start_index <= 22:
            return 'positive'
    return 'negative'

# Function to create a confusion matrix from true labels and predicted labels
def create_confusion_matrix(true_labels, predicted_labels):
    label_mapping = {'positive': 0, 'negative': 1}
    label_pairs = zip(true_labels, predicted_labels)
    counter = Counter(label_pairs)
    confusion_matrix = np.zeros((2, 2))
    
    for (true_label, predicted_label), count in counter.items():
        confusion_matrix[label_mapping[true_label], label_mapping[predicted_label]] = count
        
    return confusion_matrix

# Function to calculate accuracy, precision, and recall from the confusion matrix
def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return accuracy, precision, recall

# Unit test for the predict_label function
class TestPredictLabel(unittest.TestCase):
    def test_predict_label(self):
        seq = 'ATGCGTACGATCGATCGATCGTAGCTAC'
        true_label = 'positive'
        pred_label = predict_label(seq)
        self.assertEqual(pred_label, true_label)

def main():
    
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='A program for processing DNA sequences.')
    parser.add_argument('-f','--input_file', help='Path to the input file')
    parser.add_argument('-l','--true_labels_file', help='Path to the true labels file')
    args = parser.parse_args()

    input_file_path = args.input_file
    true_labels_file_path = args.true_labels_file


    

    # Check for input file existence and raise an error if not found
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f'Input file not found at {input_file_path}')

    # Read DNA sequences from the input file
    if input_file_path.endswith(".fasta"):
        with open(input_file_path) as handle:
            sequences = [str(record.seq) for record in SeqIO.parse(handle, "fasta")]
    else:
        with open(input_file_path) as f:
            sequences = [line.strip() for line in f]

    

    # Predict labels for the given sequences 
    predicted_labels = [predict_label(seq) for seq in sequences]

    # Read true labels from the true labels file
    with open(true_labels_file_path) as f:
        true_labels = [line.strip() for line in f]


    # Check if the number of sequences and true labels are equal
    if len(sequences) != len(true_labels):
        print(f"Warning: The number of sequences ({len(sequences)}) and true labels ({len(true_labels)}) do not match.")
        min_len = min(len(sequences), len(true_labels))
        print(f"*************************Processing the first {min_len} entries from both files.**************************")
        sequences = sequences[:min_len]
        true_labels = true_labels[:min_len] 
        predicted_labels = predicted_labels[:min_len]     

    # Create a confusion matrix and calculate metrics
    confusion_matrix = create_confusion_matrix(true_labels, predicted_labels)
    accuracy, precision, recall = calculate_metrics(confusion_matrix)
    
    # Print the results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(classification_report(true_labels, predicted_labels))





if __name__ == '__main__':
    
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    main()
