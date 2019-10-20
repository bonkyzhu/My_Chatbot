from evaluate import *
from settings import *
from nltk.translate.bleu_score import sentence_bleu

def test(method='BLUE'):
    voc, pairs = loadPrepareData(corpus_name='xhj', datafile='corpus/xhj_seg')
    test_voc, test_pairs = loadPrepareData(corpus_name='qingyun', datafile='corpus/qingyun_seg')
    testsize = len(test_pairs) * 0.1
    trimRareWords(voc, pairs, MIN_COUNT)
    trimRareWords(test_voc, test_pairs, MIN_COUNT)
    encoder, decoder = load_model_from_file(voc, file=EvalFile)
    searcher = GreedySearchDecoder(encoder, decoder)

    # SCORE = {'1-gram':0,'2-gram':0,'3-gram':0,'4-gram':0,}
    SCORE = 0
    total = 0
    line = '\n' + '-' * 80 + '\n'
    for input, target in test_pairs:
        input = ''.join(input.split(' '))
        target = target.split(' ')
        output_list = bot_answer_api(encoder, decoder, input, searcher, voc)
        if output_list:
            _, output = output_list
        if method == 'BLUE':
            blue = BLUE(target, output)
            SCORE += blue
            total += 1
            if total % 1000 == 0:
                print(f"{total/testsize} finished, BLUE score: {SCORE/total}")
            if total == int(testsize):
                break
            # for i, score in enumerate(blue):
            #     BLUE_SCORE[f'{i+1}-gram'] += score
    print(line + f"最终的BLUE得分为：{SCORE/total}" + line)


def BLUE(target, output):
    return sentence_bleu(target, output, (1, 0, 0, 0))

test()




