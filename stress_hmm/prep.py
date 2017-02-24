from hyphenate import hyphenate_word
import string

def remove_end_line_punct(sonnet):
    '''
    Removes punctuation from the end of a sonnet.

    sonnet: the string to strip endline punctuation from.
    '''
    # characters we want to remove from the sonnet
    remove = [',', '.', ':', ';', '\t', '(', ')', '?', '!']

    # Remove end of line punctuation
    sonnet = sonnet.translate(string.maketrans('', ''), ''.join(remove))

    return sonnet

def build_stress_string(stresses, start_stress, length):
    res = ''
    
    for i in range(length):
        res += stresses[start_stress]
        start_stress = (start_stress + 1) % 2

    return res


def tokenize(f_read, f_write, supervised=False):
    f_out = open(f_write, 'w')
    first = True

    if supervised is True:
        # stressed, unstressed since we construct the lines in reverse
        stresses = ['U', 'S']
        curr_stress = 0

    for line in open(f_read).readlines():
        if line.strip() == '' or line.strip().isdigit():
            continue

        line_strip = remove_end_line_punct(line.strip())

        for word in reversed(line_strip.split()):
            if supervised is True:
                res = hyphenate_word(word)
                if '-' in res:
                    res.remove('-')

                hy_length = len(res)
                start_stress = (curr_stress + hy_length) % 2
                f_out.write(build_stress_string(stresses, start_stress, 
                    hy_length) + "\t")
                curr_stress = (curr_stress + hy_length) % 2

            f_out.write(word.strip('\'').lower() + '\n')
        f_out.write("@\n")
        curr_stress = 0

    f_out.close()


def main():
    SUPERVISED = False

    if SUPERVISED is True:
        tokenize('data/concat.txt', 'data/concat_words_supervised.txt', supervised=True)
        # tokenize('data/two_sonnets.txt', 'data/sonnet_words_test.txt', supervised=True)
    else:
        tokenize('data/concat.txt', 'data/concat_words.txt')
        # tokenize('data/two_sonnets.txt', 'data/sonnet_words_test.txt')

if __name__ == '__main__':
    main()


