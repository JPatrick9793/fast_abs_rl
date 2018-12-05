""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe


def decode(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda):
    start = time()

    # SET UP MODELS
    abstractor, extractor, meta = get_abstractor_extractor_and_meta(beam_size, cuda, max_len, model_dir)

    # SET UP LOADER
    loader, n_data = get_loader_and_ndata(batch_size, split)

    # PREPARE SAVE PATHS AND LOGS
    prepare_save_paths_and_logs(beam_size, diverse, meta, save_path, split)

    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):

            # At this point, raw_article_batch is a list of lists
            # each entry in the list is an article, each list within that list are sentences
            tokenized_article_batch: map = map(tokenize(None), raw_article_batch)
            ext_arts = []
            ext_inds = []

            # so basically, this entire section tokenizes the sentences and removes punctuation, stop words, etc...
            #############################
            # this tokenize the sentences
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                else:
                    ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += [raw_art_sents[i] for i in ext]
            # Here's where the magic happens
            if beam_size > 1:
                # if beamsize is > 1, send through abstracter and rerank
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)
            else:
                # if beamsize = 1, simply send through abstractor
                dec_outs = abstractor(ext_arts)
            #############################

            # assert the batch size and output are proper sizes???
            assert i == batch_size*i_debug

            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                with open(join(save_path, 'output/{}.dec'.format(i)), 'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(i, n_data, i/n_data*100,
                                                                       timedelta(seconds=int(time()-start))),
                      end='')
    print()


def prepare_save_paths_and_logs(beam_size, diverse, meta, save_path, split):
    os.makedirs(join(save_path, 'output'))
    dec_log = {
        'abstractor': meta['net_args']['abstractor'],
        'extractor': meta['net_args']['extractor'],
        'rl': True,
        'split': split,
        'beam': beam_size,
        'diverse': diverse
    }
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)


def get_loader_and_ndata(batch_size, split):
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles

    dataset = DecodeDataset(split)
    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )
    return loader, n_data


def get_abstractor_extractor_and_meta(beam_size, cuda, max_len, model_dir):
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    extractor = RLExtractor(model_dir, cuda=cuda)
    return abstractor, extractor, meta


_PRUNE = defaultdict(
    lambda: 2,
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 4, 7: 3, 8: 3}
)


def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))


def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))


def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return -repeat, lp


if __name__ == '__main__':

    with open("SETTINGS.json") as f: args = json.load(f)["decode_full_model"]

    # split data either testing or validation
    data_split = 'test' if args.get('TEST_OR_VAL', 'test').lower() == 'test' else 'val'

    # run through decoder function
    # TODO include comment descriptions
    decode(
        args['DECODE_OUTPUT_DIR'],
        args['PRE_TRAINED_MODEL_DIR'],
        data_split,
        args.get('BATCH', 32),
        args.get('BEAMS', 1),
        args.get('DIV', 1.0),
        args.get('MAX_DEC_WORD', 30),
        args.get('CUDA', False)
    )
