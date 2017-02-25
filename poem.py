def load_sp(strip_punc=True):
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
            line = fline.strip().lower().split()
            for word_ in line:
                # Strip punctuation
                word = word_
                if strip_punc:
                    for punc in punctuation:
                        word = word.strip(punc)
                else:
                    for punc in ['\'', ':', ';']:
                        word = word.strip(punc)
                '''
                
                '''
                if word not in word_map:
                    word_map[word] = word_counter
                    word_counter += 1
                word_seq.append(word_map[word])
            words.append(word_seq)
            word_seq = []

    return words, word_map

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def gen_rhymes(word_map):
    # pip install pronouncing
    import pronouncing
    # Initialize lists
    rhymes = []
    rhymed = []
    new_rhyme = True
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
    with open('rhymes.txt', 'w') as f_out:
        for rhyme in rhymes:
            f_out.write(' '.join(rhyme) + '\n')

if __name__ == '__main__':
    print('Loading words...')
    words, word_map = load_sp(True)
    print('Finding rhymes...')
    rhymes = gen_rhymes(word_map)
    print('Writing rhymes to \'rhymes.txt\'')
    write_rhymes(rhymes)
    print('Done!')
