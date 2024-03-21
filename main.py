import spacy
from nlp_utilities import preprocess_text, count_words, get_unique_words, build_matrix, print_matrix, print_matrix_to_file, calculate_word_pairs_probability

# Load NLP model 🧠
nlp_object = spacy.load("en_core_web_sm")

# Pre-process text 📋
file_path = "transcript.txt"
only_text = preprocess_text(file_path)

# Process text with NLP model 🔄
doc = nlp_object(only_text)
sentences = [sent for sent in doc.sents]

# Count word occurrences 📊
word_counts, word_before_counts, beginning_word_counts, words_at_end_count = count_words(
    sentences)

# Get unique words 🆕
unique_words = get_unique_words(sentences)

# Build matrix 🧩
num_sentences = len(sentences)
matrix = build_matrix(unique_words, word_counts, word_before_counts,
                      beginning_word_counts, words_at_end_count, num_sentences)

print("Task 1 📝: Calculate the bi-gram probabilities of all the words/tokens in the BeRP dataset")
# Desired output file path
output_file = "task1_output.txt" 
print_matrix_to_file(matrix, output_file)  # Print the matrix to the file
print_matrix(matrix)  # Print the matrix to the terminal
print("Print the matrix done✅")


print("Task 2 📝: Calculate the probabilities of the sentences:")

sentence_1 = "show me all the Arabic food restaurants"
multiplication_result = calculate_word_pairs_probability(
    sentence_1, sentences, unique_words, word_counts, matrix)
print("2.1 Probability of \"" + sentence_1 + "\": ", multiplication_result)

sentence_2 = "I am learning mathematics"
multiplication_result = calculate_word_pairs_probability(
    sentence_2, sentences, unique_words, word_counts, matrix)
print("2.2 Probability of \"" + sentence_2 + "\": ", multiplication_result)
