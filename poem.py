def load_sp(strip_punc=True):
    """
    Load the words from Shakespeare's sonnets in 'shakespeare.txt'
    :param strip_punc: Whether to strip punctuation from the words or not
    :return: words, a list of list of numbers of words in their lines
    :return: word_map, a dictionary of words and their numbers
    """
    words = []
    word_map = {}
    word_counter = 0
    with open("data/shakespeare.txt") as f:
        word_seq = []
        punctuation = ['\'', '\"', '.', ',', '?', '!', ':', ';', '(', ')']
        while True:
            # Read the line
            fline = f.readline()
            # If at end of file, break
            if fline == '':
                break
            # Else, skip if sonnet number or whitespace
            elif is_int(fline.strip()) or '' == fline.strip():
                continue
            # Split the lines into its words
            line = fline.strip().split()
            for word_ in line:
                # Strip punctuation
                word = word_
                if strip_punc:
                    for punc in punctuation:
                        word = word.strip(punc)
                else:
                    for punc in ['\'', ':', ';', '(', ')']:
                        word = word.strip(punc)
                word = word.lower()
                if word not in word_map:
                    word_map[word] = word_counter
                    word_counter += 1
                word_seq.append(word_map[word])
            words.append(word_seq)
            word_seq = []
    return words, word_map

def is_int(s):
    """
    Check if a string is an integer
    :param s: The string to check
    :return: True, if it is an integer, otherwise False
    """
    try:
        int(s)
        return True
    except ValueError:
        return False

def gen_rhymes(word_map):
    """
    Generates a list of list of words that rhyme
    :param word_map: A hashmap of words and their corresponding numbers
    :return: list of list of rhyming words
    """
    # pip install pronouncing
    import pronouncing
    # Initialize lists
    rhymes = []
    rhymed = []
    new_rhyme = True
    # Keep track of found rhymes, dynamic programming!
    found_rhymes = []
    # For each word, num in word_map
    for word, num in word_map.iteritems():
        # Check to see if a previous rhyme for this has been found
        # Loop through all lists of found rhymes
        for rhyme in range(len(rhymes)):
            # Check to see if that word rhymes
            if unicode(word) in found_rhymes[rhyme]:
                rhymes[rhyme].append(word)
                new_rhyme = False
        # If no rhyme has been found
        if new_rhyme:
            # This is a new rhyme
            rhymed.append(word)
            found_rhymes.append(pronouncing.rhymes(word))
        # If a new rhyme has been found
        if rhymed:
            # Add it to the list and reset rhymed and new_rhyme
            rhymes.append(rhymed)
            rhymed = []
            new_rhyme = True
    return rhymes

def write_rhymes(rhymes):
    """
    Write the rhymes to a file so they can just be loaded from a file
    """
    with open('rhymes.txt', 'w') as f_out:
        for rhyme in rhymes:
            f_out.write(' '.join(rhyme) + '\n')

def format_line(line):
    """
    Strips extraneous punctuation and adds capitalization and punctuation.
    :param line: Line to be formatted (properly capitalized and punctuation)
    :return: Formatted line
    """
    words = line.strip().split(' ')
    words[0] = words[0][0].upper() + words[0][1:]
    for i in range(len(words)):
        # Capitalize words after period
        if words[i][-1] == '.' and i != len(words)-1:
            words[i + 1] = words[i + 1][0].upper() + words[i + 1][1:]
        # Capitalize I
        if words[i][0] == 'i' and (len(words[i]) == 1 or not words[i][1].isalpha()):
            words[i] = words[i][0].upper() + words[i][1:]
    # Add end of line punctuation
    if words[-1][-1] not in [',', '.', ';', '?', '!']:
        words[-1] += '.'
    return ' '.join(words)

if __name__ == '__main__':
    print('Loading words...')
    words, word_map = load_sp(True)

    print('Finding rhymes...')
    rhymes = gen_rhymes(word_map)

    print('Writing rhymes to \'rhymes.txt\'')
    write_rhymes(rhymes)
    
    print('Done!')
