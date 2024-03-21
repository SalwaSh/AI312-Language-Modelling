from fractions import Fraction
import operator
from functools import reduce
import re
import numpy as np
from collections import Counter
import re


def preprocess_text(file_path):
    # Read text
    with open(file_path, "r") as f:
        text = f.read()
    # Initialize a variable to store the processed text
    only_text = ""
    text_lines = text.splitlines()  # Split the text into lines
    # Iterate over each line in the text
    for text_line in text_lines:
        # Remove the first 10 characters (assumed to be an ID) and add a period at the end
        only_text += text_line[10:] + ". " # (I think it is better to remove the space here) 
    # Remove text enclosed in square brackets (e.g., [ABC]) using regular expressions
    only_text = re.sub("\[.*?\] ", "", only_text) # (I think it is better to remove the space here) 
    # Remove text enclosed in angle brackets (e.g., <ABC>) using regular expressions
    only_text = re.sub("\<.*?\> ", "", only_text) # (I think it is better to remove the space here)
    only_text.replace("\n", "")  # Remove any remaining newline characters
    # Return the processed text
    return only_text


def count_words(sents):
    # Initialize counters to keep track of word counts and relationships between words
    word_counts = Counter()
    word_before_counts = Counter()
    beginning_word_counts = Counter()
    words_at_end_count = Counter()
    # Iterate over each sentence in the input list of sentences
    for sent in sents:
        prev_word = None  # Keep track of the previous word in the sentence
        for i, token in enumerate(sent):
            word = token.text  # Get the current word from the sentence
            if word not in [".", "'s"]:  # Exclude certain special words from counting
                # Increment the count of the current word
                word_counts[word] += 1
                if prev_word is not None:
                    # Increment the count of the word before the current word
                    word_before_counts[(word, prev_word)] += 1
                prev_word = word  # Update the previous word for the next iteration
            if i == 0:
                # Increment the count of words at the beginning of sentences
                beginning_word_counts[word] += 1
            if i == len(sent) - 2:
                # Increment the count of words at the end of sentences
                words_at_end_count[word] += 1
    # Return the word counts and relationships between words
    return word_counts, word_before_counts, beginning_word_counts, words_at_end_count


def get_unique_words(sents):
    # Initialize an empty list to store unique words
    unique_words = []
    # Iterate over each sentence in the input list of sentences
    for sent in sents:
        for token in sent:
            word = token.text  # Get the current word from the sentence
            # Check if the word is not already in the list and not a special word
            if word not in unique_words and word not in [".", "'s"]:
                # Add the word to the list of unique words
                unique_words.append(word)
    # Return the list of unique words
    return unique_words


def build_matrix(unique_words, word_counts, word_before_counts, beginning_word_counts, words_at_end_count, num_sentences):
    number_of_unique_words = len(unique_words)
    number_of_columns = number_of_rows = number_of_unique_words + 2
    # Create a matrix with the specified number of rows and columns, initialized with zeros
    matrix = np.zeros((number_of_rows, number_of_columns), dtype=object)
    # Set the last cell in the first row as the end-of-sentence marker
    matrix[0, number_of_columns - 1] = "</S>"
    # Set the first cell in the second row as the start-of-sentence marker
    matrix[1, 0] = "<S>"

    # Iterate over the rows and columns of the matrix
    for i in range(number_of_rows):
        if i < len(unique_words):
            # Set the word in the first column for each unique word
            matrix[i + 2, 0] = unique_words[i]
        for j in range(number_of_columns):
            if j < len(unique_words) and i == 0:
                # Set the word in the first row for each unique word
                matrix[i, j + 1] = unique_words[j]

    # Fill in the transition probabilities in the matrix
    for i in range(number_of_unique_words):
        # Get the word (which is the pervious word) from the matrix
        word = matrix[i + 2, 0]
        count = word_counts.get(word, 0)  # Get the count of the pervious word
        for j in range(number_of_unique_words):
            # Get the word pair from the matrix (from first row (current) and first column (pervious))
            word_pair = (matrix[0, j + 1], matrix[i + 2, 0])
            # Get the count of the word pair (current word and pervious word)
            before_count = word_before_counts.get(word_pair, 0)
            if count > 0:
                # Calculate the transition probability
                matrix[i + 2, j + 1] = (before_count + 1) / \
                    (count + number_of_unique_words + 1)

    # Fill in the probabilities of words appearing at the beginning of sentences
    for i in range(number_of_unique_words):
        word = matrix[i + 2, 0]  # Get the word from the matrix
        # Get the count of the word appearing at the beginning
        beginning_count = beginning_word_counts.get(word, 0)
        matrix[1, i + 1] = (beginning_count + 1) / (num_sentences +
                                                    number_of_unique_words + 1)  # Calculate the probability

    # Fill in the probabilities of words appearing at the end of sentences
    for i in range(number_of_unique_words + 1):
        word = matrix[i + 1, 0]  # Get the word from the matrix
        count = word_counts.get(word, 0)  # Get the count of the word
        # Get the count of the word appearing at the end
        end_count = words_at_end_count.get(word, 0)
        if word == "<S>":
            matrix[i + 1, number_of_columns - 1] = (end_count + 1) / (
                num_sentences + number_of_unique_words + 1)  # Calculate the probability
        else:
            matrix[i + 1, number_of_columns - 1] = (end_count + 1) / (
                count + number_of_unique_words + 1)  # Calculate the probability

    # Return the resulting matrix
    return matrix


def print_matrix(matrix):
    for row in matrix:
        print(row)


def print_matrix_to_file(matrix, output_file):
    rows, cols = matrix.shape
    with open(output_file, "w") as f:
        # Write the matrix to the file
        for i in range(rows):
            for j in range(cols):
                f.write(str(matrix[i, j]) + "\t")
            f.write("\n")


def calculate_word_pairs_probability(sentence, sents, unique_words, word_counts, matrix):
    # Create a dictionary to map words to their indices
    word_to_index = {word: index for index, word in enumerate(unique_words)}
    # Split the sentence into individual words
    words = sentence.split()
    # Append the end-of-sentence marker to the list of words
    words.append("</S>")
    # Initialize the previous word variable
    prev_word = None
    # Initialize the multiplication result variable
    multiplication_result = 1

    ### NOTE: -1 means the word not in the training dataset, and None means start-of-sentence marker ###

    # Iterate over the words
    for word in words:
        current_word_index = word_to_index.get(
            word, -1)  # Get the index of the current word
        # Get the index of the previous word if it exists
        prev_word_index = word_to_index.get(
            prev_word, -1) if prev_word is not None else None

        if word == "</S>":  # If the current word is end-of-sentence marker
            if prev_word_index == -1:
                # Multiply the result by the probability of the end-of-sentence marker given the previous word not exist in the training dataset
                multiplication_result *= (1 / (len(unique_words) + 1))
            elif prev_word_index != -1:
                # Multiply the result by the probability in the matrix of the end-of-sentence marker given the previous word
                multiplication_result *= matrix[prev_word_index + 2][-1]

        # If the current word and previous word exist in the training dataset
        elif current_word_index != -1 and prev_word_index != -1:
            if (current_word_index is not None) and (prev_word_index is not None):
                # Multiply the result by the probability in the matrix of the current word given the previous word
                multiplication_result *= matrix[prev_word_index +
                                                2][current_word_index + 1]
            elif (prev_word_index is None) and (current_word_index is not None):
                # Multiply the result by the probability in the matrix of the current word given the start-of-sentence marker
                multiplication_result *= matrix[1][current_word_index + 1]

        # If the current word does not exist in the training dataset but the previous word exists
        elif current_word_index == -1 and prev_word_index != -1:
            if prev_word_index is None:
                # Multiply the result by the probability of the current word given the start-of-sentence marker
                multiplication_result *= (1 /
                                          ((len(unique_words) + 1) + len(sents)))
            elif prev_word_index is not None:
                # Multiply the result by the probability of the current word given the previous word
                multiplication_result *= (1 / ((len(unique_words) + 1) +
                                          word_counts.get(prev_word_index, 0)))

        elif prev_word_index == -1:  # If the previous word does not exist in the training dataset
            # Multiply the result by the probability of the current word given that the previous word does not exist in the dataset
            multiplication_result *= (1 / (len(unique_words) + 1))

        # Update the previous word to the current word
        prev_word = word

    # Return the final multiplication result
    return multiplication_result
