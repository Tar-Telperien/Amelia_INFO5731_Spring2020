import torch
import sys

##############################
# imports from trainer.py
import math
import os
from functools import partial

from tqdm import tqdm

# import dataloader
# import model
# import transformer
# import util
# from decoding import Decode, get_decode_fn
# from model import dummy_mask
# from trainer import BaseTrainer

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')
#############################

sys.path.insert(0, './src')

model_fp = 'checkpoints/sigmorphon20-task0/mono-hmm/default/eng.nll_0.0071.acc_96.4858.dist_0.0814.epoch_14'

my_model = torch.load(model_fp)

# decoder
# get_decode_fn
# decode_fn = get_decode_fn(params.decode, params.max_decode_len)
# default.greedy

inputfp = 'task0-data/DEVELOPMENT-LANGUAGES/germanic/swe.dev'

# third option is a Decode class

decode_fn = get_decode_fn()

my_model.decode('dev', f'{model_fp}.decode', decode_fn)

# my_model.predict(inputfp)

# print(my_model.eval())

# pred, _ = decode_fn(my_model, inputfp)


# asking for model.decode()

#######################################

# decode function

# def decode(self, mode, write_fp, decode_fn):
#     self.model.eval()
#     cnt = 0
#     sampler, nb_instance = self.iterate_instance(mode)
#     decode_fn.reset()
#     # with open(f'{write_fp}.{mode}.tsv', 'w') as fp:
#     #  fix alexander kahanek
#     with open('{0}.{1}.tsv'.format(write_fp, mode), 'w') as fp:
#         # fp.write(f'prediction\ttarget\tloss\tdist\n')
#         # fix alexander kahanek
#         fp.write('prediction\ttarget\tloss\tdist\n')
#         for src, trg in tqdm(sampler(), total=nb_instance):
#             pred, _ = decode_fn(self.model, src)
#             dist = util.edit_distance(pred, trg.view(-1).tolist()[1:-1])

#             src_mask = dummy_mask(src)
#             trg_mask = dummy_mask(trg)
#             data = (src, src_mask, trg, trg_mask)
#             loss = self.model.get_loss(data).item()

#             trg = self.data.decode_target(trg)[1:-1]
#             pred = self.data.decode_target(pred)
#             fp.write(
#                 # f'{" ".join(pred)}\t{" ".join(trg)}\t{loss}\t{dist}\n')
#                 # fix alexander kahanek
#                 # tsv file fix spaces??? seems like it is supposed to be that way
#                 '{0}\t{1}\t{2}\t{3}\n'.format(" ".join(pred), " ".join(trg), loss, dist))
#             cnt += 1
#     decode_fn.reset()
#     # self.logger.info(f'finished decoding {cnt} {mode} instance')
#     # fix alexander kahanek
#     self.logger.info(
#         'finished decoding {0} {1} instance'.format(cnt, mode))


#  my_model
#  MonoTagHMMTransducer(
#   (src_embed): Embedding(77, 200, padding_idx=0)
#   (trg_embed): Embedding(77, 200, padding_idx=0)
#   (enc_rnn): LSTM(200, 400, num_layers=2, dropout=0.4, bidirectional=True)
#   (dec_rnn): StackedLSTM(
#     (layers): ModuleList(
#       (0): LSTMCell(240, 400)
#     )
#     (dropout): Dropout(p=0.4, inplace=False)
#   )
-UU-:%%--F1  model_load.py   31% L69    (Python ElDoc) -----------------------------------------------------------------------------------------------

