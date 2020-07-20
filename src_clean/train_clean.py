'''
train
'''
import math
import os
from functools import partial

import torch
from tqdm import tqdm

import dataloader_clean
import model
import transformer
import util
from decoding import Decode, get_decode_fn
from model import dummy_mask
from trainer import BaseTrainer

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


class Data(util.NamedEnum):
    sigmorphon17task1 = 'sigmorphon17task1'


class Arch(util.NamedEnum):
    hardmono = 'hardmono'  # hard monotonic attention
    hmm = 'hmm'  # 0th-order hard attention without input-feeding
    hmmfull = 'hmmfull'  # 1st-order hard attention without input-feeding
    transformer = 'transformer'


class Trainer(BaseTrainer):
    '''docstring for Trainer.'''
    def set_args(self):
        '''
        get_args
        '''
        # yapf: disable
        super().set_args()
        parser = self.parser
        parser.add_argument('--dataset', required=True, type=Data, choices=list(Data))
        parser.add_argument('--max_seq_len', default=128, type=int)
        parser.add_argument('--max_decode_len', default=128, type=int)
        parser.add_argument('--init', default='', help='control initialization')
        parser.add_argument('--dropout', default=0.2, type=float, help='dropout prob')
        parser.add_argument('--embed_dim', default=100, type=int, help='embedding dimension')
        parser.add_argument('--nb_heads', default=4, type=int, help='number of attention head')
        parser.add_argument('--src_layer', default=1, type=int, help='source encoder number of layers')
        parser.add_argument('--trg_layer', default=1, type=int, help='target decoder number of layers')
        parser.add_argument('--src_hs', default=200, type=int, help='source encoder hidden dimension')
        parser.add_argument('--trg_hs', default=200, type=int, help='target decoder hidden dimension')
        parser.add_argument('--label_smooth', default=0., type=float, help='label smoothing coeff')
        parser.add_argument('--tie_trg_embed', default=False, action='store_true', help='tie decoder input & output embeddings')
        parser.add_argument('--arch', required=True, type=Arch, choices=list(Arch))
        parser.add_argument('--nb_sample', default=2, type=int, help='number of sample in REINFORCE approximation')
        parser.add_argument('--wid_siz', default=11, type=int, help='maximum transition in 1st-order hard attention')
        parser.add_argument('--indtag', default=False, action='store_true', help='separate tag from source string')
        parser.add_argument('--decode', default=Decode.greedy, type=Decode, choices=list(Decode))
        parser.add_argument('--mono', default=False, action='store_true', help='enforce monotonicity')
        parser.add_argument('--bestacc', default=False, action='store_true', help='select model by accuracy only')
        # yapf: enable

    def load_data(self, dataset, train, dev, test):
        assert self.data is None
        logger = self.logger
        params = self.params
        # yapf: disable
        if params.arch == Arch.hardmono:
            if dataset == Data.sigmorphon17task1:
                self.data = dataloader.AlignSIGMORPHON2017Task1(train, dev, test, params.shuffle)
            else:
                raise ValueError
        else:
            if dataset == Data.sigmorphon17task1:
                if params.indtag:
                    self.data = dataloader.TagSIGMORPHON2017Task1(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.SIGMORPHON2017Task1(train, dev, test, params.shuffle)
            else:
                raise ValueError
        # yapf: enable
        logger.info('src vocab size %d', self.data.source_vocab_size)
        logger.info('trg vocab size %d', self.data.target_vocab_size)
        logger.info('src vocab %r', self.data.source[:500])
        logger.info('trg vocab %r', self.data.target[:500])

    def build_model(self):
        assert self.model is None
        params = self.params
        if params.arch == Arch.hardmono:
            params.indtag, params.mono = True, True
        kwargs = dict()
        kwargs['src_vocab_size'] = self.data.source_vocab_size
        kwargs['trg_vocab_size'] = self.data.target_vocab_size
        kwargs['embed_dim'] = params.embed_dim
        kwargs['nb_heads'] = params.nb_heads
        kwargs['dropout_p'] = params.dropout
        kwargs['tie_trg_embed'] = params.tie_trg_embed
        kwargs['src_hid_size'] = params.src_hs
        kwargs['trg_hid_size'] = params.trg_hs
        kwargs['src_nb_layers'] = params.src_layer
        kwargs['trg_nb_layers'] = params.trg_layer
        kwargs['nb_attr'] = self.data.nb_attr
        kwargs['nb_sample'] = params.nb_sample
        kwargs['wid_siz'] = params.wid_siz
        kwargs['label_smooth'] = params.label_smooth
        kwargs['src_c2i'] = self.data.source_c2i
        kwargs['trg_c2i'] = self.data.target_c2i
        kwargs['attr_c2i'] = self.data.attr_c2i
        model_class = None
        indtag, mono = True, True
        # yapf: disable
        fancy_classfactory = {
            (Arch.hardmono, indtag, mono): model.HardMonoTransducer,
            (Arch.hmm, indtag, not mono): model.TagHMMTransducer,
            (Arch.hmm, indtag, mono): model.MonoTagHMMTransducer,
            (Arch.hmmfull, indtag, not mono): model.TagFullHMMTransducer,
            (Arch.hmmfull, indtag, mono): model.MonoTagFullHMMTransducer,
        }
        regular_classfactory = {
            Arch.hmm: model.HMMTransducer,
            Arch.hmmfull: model.FullHMMTransducer,
            Arch.transformer: transformer.Transformer,
        }
        # yapf: enable
        if params.indtag or params.mono:
            model_class = fancy_classfactory[(params.arch, params.indtag,
                                              params.mono)]
        else:
            model_class = regular_classfactory[params.arch]
        self.model = model_class(**kwargs)
        if params.indtag:
            self.logger.info('number of attribute %d', self.model.nb_attr)
            self.logger.info('dec 1st rnn %r', self.model.dec_rnn.layers[0])
        self.logger.info('model: %r', self.model)
        self.logger.info('number of parameter %d',
                         self.model.count_nb_params())
        self.model = self.model.to(self.device)

    def dump_state_dict(self, filepath):
        util.maybe_mkdir(filepath)
        self.model = self.model.to('cpu')
        torch.save(self.model.state_dict(), filepath)
        self.model = self.model.to(self.device)
        self.logger.info(f'dump to {filepath}')

    def load_state_dict(self, filepath):
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.logger.info(f'load from {filepath}')

    def setup_evalutator(self):
        arch, dataset = self.params.arch, self.params.dataset
        if arch == Arch.hardmono:
            if dataset == Data.sigmorphon17task1:
                self.evaluator = util.PairBasicEvaluator()
            else:
                raise ValueError
        else:
            self.evaluator = util.BasicEvaluator()

    def evaluate(self, mode, epoch_idx, decode_fn):
        self.model.eval()
        sampler, nb_instance = self.iterate_instance(mode)
        decode_fn.reset()
        results = self.evaluator.evaluate_all(sampler, nb_instance, self.model,
                                              decode_fn)
        decode_fn.reset()
        for result in results:
            self.logger.info(
                f'{mode} {result.long_desc} is {result.res} at epoch {epoch_idx}'
            )
        return results

    def decode(self, mode, write_fp, decode_fn):
        self.model.eval()
        cnt = 0
        sampler, nb_instance = self.iterate_instance(mode)
        decode_fn.reset()
        with open(f'{write_fp}.{mode}.tsv', 'w') as fp:
            fp.write(f'prediction\ttarget\tloss\tdist\n')
            for src, trg in tqdm(sampler(), total=nb_instance):
                pred, _ = decode_fn(self.model, src)
                dist = util.edit_distance(pred, trg.view(-1).tolist()[1:-1])

                src_mask = dummy_mask(src)
                trg_mask = dummy_mask(trg)
                data = (src, src_mask, trg, trg_mask)
                loss = self.model.get_loss(data).item()

                trg = self.data.decode_target(trg)[1:-1]
                pred = self.data.decode_target(pred)
                fp.write(
                    f'{" ".join(pred)}\t{" ".join(trg)}\t{loss}\t{dist}\n')
                cnt += 1
        decode_fn.reset()
        self.logger.info(f'finished decoding {cnt} {mode} instance')

    def select_model(self):
        best_res = [m for m in self.models if m.evaluation_result][0]
        best_acc = [m for m in self.models if m.evaluation_result][0]
        best_devloss = self.models[0]
        for model in self.models:
            if not model.evaluation_result:
                continue
            if type(self.evaluator) == util.BasicEvaluator
                # [acc, edit distance / per ]
                if model.evaluation_result[0].res >= best_res.evaluation_result[0].res and \
                   model.evaluation_result[1].res <= best_res.evaluation_result[1].res:
                    best_res = model
            else:
                raise NotImplementedError
            if model.evaluation_result[0].res >= best_acc.evaluation_result[0].res:
                best_acc = model
            if model.devloss <= best_devloss.devloss:
                best_devloss = model
        if self.params.bestacc:
            best_fp = best_acc.filepath
        else:
            best_fp = best_res.filepath
        return best_fp, set([best_fp])


def main():
    '''
    main
    '''
    trainer = Trainer()
    params = trainer.params
    decode_fn = get_decode_fn(params.decode, params.max_decode_len)
    trainer.load_data(params.dataset, params.train, params.dev, params.test)
    trainer.setup_evalutator()
    if params.load and params.load != '0':
        if params.load == 'smart':
            start_epoch = trainer.smart_load_model(params.model) + 1
        else:
            start_epoch = trainer.load_model(params.load) + 1
        trainer.logger.info('continue training from epoch %d', start_epoch)
        trainer.setup_training()
        trainer.load_training(params.model)
    else:  # start from scratch
        start_epoch = 0
        trainer.build_model()
        if params.init:
            if os.path.isfile(params.init):
                trainer.load_state_dict(params.init)
            else:
                trainer.dump_state_dict(params.init)
        trainer.setup_training()

    trainer.run(start_epoch, decode_fn=decode_fn)


if __name__ == '__main__':
    main()
