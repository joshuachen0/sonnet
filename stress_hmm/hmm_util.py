from __future__ import print_function
from HMM import unsupervised_HMM, supervised_HMM
import numpy as np


def load_sonnets(f_name, supervised=True):
    '''
    Loads the file 'sonnet_words.txt'.

    Returns (possible):
        stresses:   Sequnces of states, i.e. a list of lists.
                    Each sequence represents the stresses of a word.
        stress_map: A hash map that maps each stress combination to an integer.
        words:      Sequences of observations, i.e. a list of lists.
                    Each sequence represents a word in a line.
        word_map:   A hash map that maps each word to an integer.
    '''
    if supervised is True:

        stresses = []
        stress_map = {}

        stress_counter = 0

    words = []
    word_map = {}
    word_counter = 0

    with open(f_name, 'r') as f:
        if supervised is True:
            stress_seq = []
        word_seq = []

        for line in f:
            line = line.strip()
            if line == '' or line == '@':
                # A line has passed. Add the current sequence to
                # the list of sequences.
                if supervised is True:
                    stresses.append(stress_seq)

                words.append(word_seq)
                # Start new sequences.
                stress_seq = []
                word_seq = []
            
            if line == '':
                break
            elif line == '@':
                continue
            
            if supervised is True:
                stress, word = line.split()

            else:
                word = line.lower()

            # print word
            
            if word not in word_map:
                word_map[word] = word_counter
                word_counter += 1

            word_seq.append(word_map[word])

            if supervised is True:
                if stress not in stress_map:
                    if stress[0] == "S":
                        stress_add = -2
                    else:
                        stress_add = -1
                    stress_map[stress] = 2 * len(stress) + stress_add
                    # stress_counter += 1

                # Convert the genre into an integer.
                stress_seq.append(stress_map[stress])

    if supervised is True:
        return stresses, stress_map, words, word_map
        
    return words, word_map


def unsupervised_learning(words, word_map, n_states, n_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''

    # Train the HMM.
    HMM = unsupervised_HMM(words, n_states, n_iters)

    # Print the transition matrix.
    # print("Transition Matrix:")
    # print('#' * 70)
    # for i in range(len(HMM.A)):
    #     print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    # print('')
    # print('')

    # # Print the observation matrix. 
    # print("Observation Matrix:  ")
    # print('#' * 70)
    # for i in range(len(HMM.O)):
    #     print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    # print('')
    # print('')

    return HMM

def make_emission(HMM, word_map, seed):
    # pick a number of syllables (a state)
    # generate a word from this state
    # repeat until we've exhausted 10 syllables
    temp = HMM.generate_emission(10, word_map, seed)

    # for i in temp:
    #     print i

    return temp
    

def make_sonnet(HMM, word_map, rhymes, supervised=False):
    rhymes = pick_rhymes(rhymes)
    
    for i in range(len(rhymes)):
        if supervised is False:
            res = make_emission(HMM, word_map, rhymes[i])
            res.reverse()
        else:
            res = make_emission(HMM, word_map, rhymes[i])

        for i in res:
            print (i, end=' ')
        print(end='\n')
        
        print

def load_rhymes(f_name):
    rhyme_list = []

    with open(f_name, 'r') as f:
        for line in f:
            # print (line)
            rhymes = line.split()
            # print (rhymes)

            if len(rhymes) > 1:
                rhyme_list.append(rhymes)

    # print (rhyme_list)
    return rhyme_list

def pick_rhymes(rhymes):
    seeds = []
    indices = np.random.choice(np.arange(len(rhymes)), 7, replace=False)

    for i in range(0, 5, 2):
        rhyme1_ind = np.random.choice(np.arange(len(rhymes[i])), 2, replace=False)
        rhyme2_ind = np.random.choice(np.arange(len(rhymes[i + 1])), 2, replace=False)

        for index in range(2):
            seeds.append(rhymes[i][rhyme1_ind[index]])
            seeds.append(rhymes[i + 1][rhyme2_ind[index]])

    last_index = indices[-1]
    
    last_ind = np.random.choice(np.arange(len(rhymes[last_index])), 2, replace=False)

    for i in last_ind:
        seeds.append(rhymes[last_index][i])

    return seeds


def main():
    supervised = False
    rhymes = load_rhymes("data/rhymes.txt")

    if supervised is True:
        stresses, stress_map, words, word_map = \
            load_sonnets("data/concat_words_supervised.txt", supervised=True)
        print (len(stress_map.keys()))
        print (stress_map)
    else:
        words, word_map = load_sonnets("data/concat_words.txt", supervised=False)
    
    print (len(word_map.keys()))
    

    if supervised is False:
        HMM = unsupervised_learning(words, word_map, n_states=10, n_iters=1)
    else:
        HMM = supervised_HMM(words, stresses)

    for i in range(5):
        make_sonnet(HMM, word_map, rhymes, supervised=supervised)
        print(end='\n')

    # Get rhyme pair
    # make pair of lines with rhyme pair as "ending" word
    
    

if __name__ == '__main__':
    main()