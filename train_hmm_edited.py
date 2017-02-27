from baum_welch.HMM import *
import poem
import multiprocessing as mp
import functools
import random
import pronouncing

def invert_map(my_map):
    """
    :param my_map: Map to invert, with unique values.
    :return: Map of values -> keys.
    """
    return {v: k for k, v in my_map.iteritems()}

def generate_line_naive(hmm, word_map, num_syls, start):
    """
    Generate a line of length num_words, with no constraints.
    :param hmm: Trained HMM.
    :param word_map: Dictionary of words (strings) to ints.
    :param num_words: Number of words to generate.
    :return: A string that is a space-separated sequence of words generated
    by the HMM.
    """
    int_to_word_map = invert_map(word_map)
    words = [start]
    state = None
    tot_syls = est_num_syllables(start)
    while tot_syls != 10:
        if tot_syls > 10:
            tot_syls -= est_num_syllables(word)
            last = words.pop()
        obs, state = hmm.generate_observation(state)
        word = int_to_word_map[obs]
        tot_syls += est_num_syllables(word)
        words.append(int_to_word_map[obs])

    # HMM generates words in reverse order (end of line -> beginning to line)
    # Reverse so it makes sense to us
    words.reverse()

    # Join with spaces
    return ' '.join(words)

def train_and_print(X, n_states, n_iters, model_i, word_map):
    """
    Train one hidden model and print the results.
    :param X: Dataset, list of lists.
    :param n_states: Number of hidden states to train with.
    :param n_iters: Number of E-M iterations to train with.
    :param model_i: Model identifier, for multiple models with the same
        parameters.
    :param word_map: Dict of words to ints.
    """
    # Train unsupervised HMM
    hmm, scores = unsupervised_HMM(X, n_states, n_iters)
    print('-' * 70)
    print('{} hidden states, {} iterations, model {}'
          .format(n_states, n_iters, model_i))
    print(scores)
    # Generate rhymes
    rhymes = poem.gen_rhymes(word_map)
    # Check that list of rhyming words is > 1, more than 1 rhymed word
    rhymes = [rhyme for rhyme in rhymes if len(rhyme) > 1]
    random.seed()
    rhyme = random.sample(range(len(rhymes)), 7)
    # Generate line by line
    for i in range(14):
        # Select random rhymes from list of generated rhymes
        #rhyme = random.sample(range(len(rhymes)), 7)
        prev_starts = []
        start = ''
        # Generated rhyming words
        if i == 0 or i == 2:
            # Check rhyme is not repeated with word in same line and previous
            # set of lines
            while start == '' or start in prev_starts:
                start = str(random.sample(rhymes[rhyme[0]], 1)[0])
            prev_starts.append(start)
        if i == 1 or i == 3:
            while start == '' or start in prev_starts:
                start = str(random.sample(rhymes[rhyme[1]], 1)[0])
            prev_starts.append(start)
        if i == 4 or i == 6:
            while start == '' or start in prev_starts:
                start = str(random.sample(rhymes[rhyme[2]], 1)[0])
            prev_starts.append(start)
        if i == 5 or i == 7:
            while start == '' or start in prev_starts:
                start = str(random.sample(rhymes[rhyme[3]], 1)[0])
            prev_starts.append(start)
        if i == 8 or i == 10:
            while start == '' or start in prev_starts:
                start = str(random.sample(rhymes[rhyme[4]], 1)[0])
            prev_starts.append(start)
        if i == 9 or i == 11:
            while start == '' or start in prev_starts:
                start = str(random.sample(rhymes[rhyme[5]], 1)[0])
            prev_starts.append(start)
        if i == 12 or i == 13:
            while start == '' or start in prev_starts:
                start = str(random.sample(rhymes[rhyme[6]], 1)[0])
            prev_starts.append(start)
        # Generate the 10-syllable line with the rhyme
        print(poem.format_line(generate_line_naive(hmm, word_map, 10, start)))
    print

def est_num_syllables(word):
    """
    :param word: A string.
    :return: An estimate of the number of syllables in the word.
        If it's in the dictionary, return the actual number of syllables.
        Else return # characters / 4.
    """
    phones = pronouncing.phones_for_word(word)
    if phones:
        return pronouncing.syllable_count(phones[0])
    else:
        # Pseudo-estimation of lines
        return len(word) / 4

if __name__ == '__main__':
    # Load the words and word_map
    X, word_map = poem.load_sp()
    # Train HMM and print poem
    train_and_print(X, 10, 100, 0, word_map)