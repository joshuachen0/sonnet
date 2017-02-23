from hyphenate import hyphenate_word
import string

sonnet = \
'''
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel:
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And tender churl mak'st waste in niggarding:
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.
'''

def remove_end_line_punct(sonnet):
    '''
    Removes punctuation from the end of a sonnet.

    sonnet: the string to strip endline punctuation from.
    '''
    # characters we want to remove from the sonnet
    remove = [',', '.', ':', ';', '\t']

    # Add character to denote end of lines
    sonnet = sonnet.replace('\n', ' @ ')

    # Remove end of line punctuation
    sonnet = sonnet.translate(string.maketrans('', ''), ''.join(remove))
    return sonnet

sonnet = remove_end_line_punct(sonnet)
print sonnet

for word in sonnet.split(" "):
    res = hyphenate_word(word)
    if "-" in res:
        res.remove("-")
        print res
    else:
        print res