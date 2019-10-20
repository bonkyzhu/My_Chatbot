'''
实际上未使用
'''
import torch
from math import exp
import numpy as np
import logging
from settings import  *
logging.basicConfig(filename='info.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='[%(asctime)s]%(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
)


def BP(candidate_length, reference_length):
    '''
    返回长度惩罚因子BP的值
    :param candidate: 生成的句子的长度
    :param reference:
    :return: BP
    '''
    pass
    if candidate_length < reference_length:
        BP = 1
    else:
        BP = exp(1 - reference_length/candidate_length)
    return BP

def BLUE(decoder_out, target, mask):
    '''
    Unigram
    :param decoder_out: [10, 100, 25907]
    :param target: [10, 100]
    :param mask:  [10, 100]
    :return: AVG_BLUE
    '''
    logging.info(f'batch size: {batch_size}, max_length: {MAX_LENGTH}')
    logging.debug(f'检查初始维度 {decoder_out.shape}, {target.shape}, {mask.shape}')
    decoder_out.transpose_(0, 1)
    target.t_()
    mask.t_()
    logging.debug(f'检查变换后维度 {decoder_out.shape}, {target.shape}, {mask.shape}')
    # OK, torch.Size([100, 10, 25907]), torch.Size([100, 10]), torch.Size([100, 10])
    BLUE_loss = 0
    for i in range(batch_size):
        current_decoder_out = decoder_out[i]
        current_target = target[i]
        current_mask = mask[i]

        sample_decoder_out = [word_vec.argmax() for word_vec in current_decoder_out]
        logging.debug(f'检查输出语句\n {sample_decoder_out}')

        candidate_length = current_mask.sum().item()
        try:
            reference_length = list(current_target).index(0)
        except:
            reference_length = len(current_target)
        bp = BP(candidate_length, reference_length)
        logging.debug(f'检查BP值: 生成句子长度 {candidate_length}, 参考句子长度 {reference_length}, BP {bp}')

        reference_dict = {}
        for j in range(reference_length):
            if current_target[j] not in reference_dict.keys():
                reference_dict[current_target[j]] = 1 # 如果没有就创建
            else:
                reference_dict[current_target[j]] += 1 # 有则计算数量
        logging.debug(f'检查reference dict: \n{reference_dict}')

        num = 0
        for j in range(candidate_length):
            if sample_decoder_out[j] in reference_dict.keys() : # 如果存在当前词
                if reference_dict[sample_decoder_out[i]] != 0:
                    num += 1 # 计数
                    reference_dict[sample_decoder_out[i]] -= 1 # 不超过原句子相同单词的数量，减减，到0停止计数

        precision = num / candidate_length
        BLUE_value = bp * precision
        logging.debug(f'检查BLUE值: {BLUE_value}, 精度：{precision}')
        BLUE_loss += BLUE_value

    AVG_BLUE = BLUE_loss / batch_size
    logging.info(f'平均BLUE: {AVG_BLUE}')
    AVG_BLUE = torch.Tensor([AVG_BLUE])
    AVG_BLUE.requires_grad = True
    return AVG_BLUE

