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
            line = fline.strip().split()
            for word_ in line:
                # Strip punctuation
                word = word_
                if strip_punc:
                    for punc in punctuation:
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

if __name__ == '__main__':
    words, word_map = load_sp()
