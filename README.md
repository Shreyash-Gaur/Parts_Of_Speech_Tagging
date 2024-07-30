

---

# Parts_of_Speech_Tagging

## Project Overview
This project demonstrates the process of Part-of-Speech (POS) tagging, which involves assigning a part-of-speech tag (e.g., Noun, Verb, Adjective) to each word in an input text. POS tagging is essential for understanding the syntactic structure of sentences, which in turn aids in various natural language processing (NLP) tasks such as speech recognition, search queries, and more.

In this project, we:
- Learned how POS tagging works.
- Computed the transition matrix and emission matrix in a Hidden Markov Model (HMM).
- Implemented the Viterbi algorithm for sequence prediction.
- Evaluated the accuracy of our model.

## Table of Contents
1. [Data Sources](#data-sources)
2. [Parts of Speech Tagging](#parts-of-speech-tagging)
   - 2.1 [Training](#training)
   - 2.2 [Testing](#testing)
3. [Hidden Markov Models for POS](#hidden-markov-models-for-pos)
   - 3.1 [Generating Matrices](#generating-matrices)
   - 3.2 [Viterbi Algorithm and Dynamic Programming](#viterbi-algorithm-and-dynamic-programming)
   - 3.3 [Predicting on a Dataset](#predicting-on-a-dataset)
4. [Results](#results)
5. [Usage](#usage)
6. [Contributing](#contributing)

## Data Sources
The project utilizes two tagged datasets from the Wall Street Journal (WSJ):
- **Training Set**: `WSJ-2_21.pos` for training the model.
- **Test Set**: `WSJ-24.pos` for evaluating the model.

Additionally, the training data has been preprocessed to form a vocabulary (`hmm_vocab.txt`), which includes words from the training set that appear two or more times. The vocabulary also contains 'unknown word tokens' to handle words not present in the training data.

## Parts of Speech Tagging

### Training
In this section, we focus on training the POS tagger. The process includes:
- Computing transition counts: The number of times each tag appears next to another tag.
- Computing emission counts: The number of times each word appears with a specific tag.
- Computing tag counts: The number of times each tag appears in the training data.

The `create_dictionaries` function processes the training corpus to generate three dictionaries: `emission_counts`, `transition_counts`, and `tag_counts`.

### Testing
After training, we evaluate the model's accuracy using the test set. The `predict_pos` function assigns POS tags to each word in the preprocessed test corpus (`prep`). It then compares the predicted tags with the actual tags in the test corpus to compute the accuracy.

## Hidden Markov Models for POS

### Generating Matrices
The transition and emission matrices are computed based on the counts obtained during training. These matrices represent the probabilities of transitioning from one tag to another and emitting a specific word given a tag, respectively.

### Viterbi Algorithm and Dynamic Programming
The Viterbi algorithm is implemented to find the most likely sequence of POS tags for a given sentence. It involves three main steps:
1. Initialization: Setting up initial probabilities.
2. Viterbi Forward: Computing the highest probability for each tag at each position in the sentence.
3. Viterbi Backward: Backtracking to find the optimal path of tags.

### Predicting on a Dataset
The final step involves using the trained model and the Viterbi algorithm to predict POS tags for sentences in the test dataset and evaluate the model's accuracy.

## Results
The results section includes:
- Examples of transition and emission counts.
- Examples of ambiguous words and their corresponding tags.
- The overall accuracy of the POS tagger.

## Usage

### Installation
1. Clone the repository.
   ```bash
   git clone https://github.com/Shreyash-Gaur/Parts_Of_Speech_Tagging.git
   cd Parts_of_Speech_Tagging
   ```

2. Ensure you have the required dependencies installed.
   ```bash
   pip install numpy
   ```

### Running the Project
1. Run the `Parts-of-Speech Tagging.ipynb` notebook to train the model and evaluate its performance. You can open the notebook using Jupyter Notebook or Jupyter Lab.
   ```bash
   jupyter notebook Parts-of-Speech\ Tagging.ipynb
   ```

## utils_pos.py
The `utils_pos.py` file contains utility functions used in the project:

### Key Functions

- **initialize_probs(states)**: Initializes transition probabilities for a set of possible POS tags.
- **get_states(tagged_corpus)**: Extracts states (POS tags) and observations (words) from a tagged corpus.
- **get_probabilities(states, observations)**: Generates initial transition and emission probability dictionaries.
- **get_transitions(tagged_sents, states, transition_probability, emission_probability)**: Computes transition and emission probabilities from a tagged dataset.
- **assign_unk(tok)**: Assigns unknown word tokens based on morphology rules.
- **get_word_tag(line, vocab)**: Retrieves the word and its tag from a line of text, handling unknown words.
- **preprocess(vocab, data_fp)**: Preprocesses data by handling unknown words and preparing the dataset for training and evaluation.
- **get_frequency(tagged_corpus)**: Computes the frequency of each word and its possible tags in the tagged corpus.
- **predict_pos(prep, y, tagged_counts)**: Predicts POS tags for a given dataset and calculates the accuracy.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

---

