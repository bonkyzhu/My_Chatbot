#coding=utf-8
import re

noise_word = ['\n']


def sentence_cleaner(sentence):
    for noise in noise_word:
        sentence = re.sub(noise, '', sentence)
    return sentence



def load_conversation_pairs_qy(file_loc):
    file = open(file_loc, 'r', encoding='utf-8')
    pairs = []
    for line in file:
        pair = []
        split = line.split('|')
        if len(split) == 2:
            pair.append(split[0])
            pair.append(sentence_cleaner(split[1]))
            pairs.append(pair)
    file.close()
    return pairs

if __name__ == '__main__':
    print(load_conversation_pairs_qy('qingyun_seg'))
