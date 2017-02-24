from HMM import *


def invert_map(my_map):
    """
    :param my_map: Map to invert, with unique values.
    :return: Map of values -> keys.
    """
    return {v: k for k, v in my_map.iteritems()}

def generate_line_naive(hmm, word_map, num_words):
    """
    Generate a line of length num_words, with no constraints.
    :param hmm: Trained HMM.
    :param word_map: Dictionary of words (strings) to ints.
    :param num_words: Number of words to generate.
    :return: A string that is a space-separated sequence of words generated
    by the HMM.
    """
    int_to_word_map = invert_map(word_map)
    words = []
    state = None
    for i in range(num_words):
        obs, state = hmm.generate_observation(state)
        words.append(int_to_word_map[obs])

    # HMM generates words in reverse order (end of line -> beginning to line)
    # Reverse so it makes sense to us
    words.reverse()

    # Join with spaces
    return ' '.join(words)

def train_n_states(X, word_map):
    """Train several HMMs. Vary on the number of states."""
    open('train/train_n_states_out.txt', )
    n_states_vals = range(2, 21, 2)  # 2, 4, ..., 20
    n_iterations = 100

    for n_states in n_states_vals:
        hmm = unsupervised_HMM(X, n_states, 100)
        generate_line_naive(hmm, word_map, 100)

