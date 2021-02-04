"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

from dataset import Dictionary, VQAFeatureDataset_Custom
import base_model
import utils
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=8)
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logits', action='store_true')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--questions', type=str, default='')
    args = parser.parse_args()
    return args


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, label2ans):
    _m, idx = p.max(0)
    return label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, label2ans, image, questions_tensor):
    v = image['features']
    b = image['spatial']
    q = questions_tensor

    v = v.cuda()
    b = b.cuda()
    q = q.cuda()

    print(v.shape, b.shape, q.shape)

    logits, att = model(v, b, q, None)
    return get_answer(logits.data[0], label2ans)





def tokenize(questions, dictionary, max_length=14):
    """Tokenizes the questions.

    This will add q_token in each entry of the dataset.
    -1 represent nil, and should be treated as padding_idx in embedding
    """
    entries = []
    for question in questions:
        tokens = dictionary.tokenize(question, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [dictionary.padding_idx] * (max_length - len(tokens))
            tokens = tokens + padding
        utils.assert_eq(len(tokens), max_length)
        entries.append({"q_token": tokens})
    return entries

def tensorize(entry):
  question = torch.from_numpy(np.array(entry['q_token'])) 
  return question

# if __name__ == '__main__':
#     args = parse_args()

#     torch.backends.cudnn.benchmark = True

#     dictionary = Dictionary.load_from_file('data/dictionary.pkl')
#     eval_dset = VQAFeatureDataset(args.split, dictionary, adaptive=True)

#     n_device = torch.cuda.device_count()
#     batch_size = args.batch_size * n_device

#     constructor = 'build_%s' % args.model
#     model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma).cuda()
#     eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

#     def process(args, model, eval_loader):
#         model_path = args.input+'/model%s.pth' % \
#             ('' if 0 > args.epoch else '_epoch%d' % args.epoch)
    
#         print('loading %s' % model_path)
#         model_data = torch.load(model_path)

#         model = nn.DataParallel(model).cuda()
#         model.load_state_dict(model_data.get('model_state', model_data))

#         model.train(False)

#         logits, qIds = get_logits(model, eval_loader)
#         results = make_json(logits, qIds, eval_loader)
#         model_label = '%s%s%d_%s' % (args.model, args.op, args.num_hid, args.label)

#         if args.logits:
#             utils.create_dir('logits/'+model_label)
#             torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % args.index)
        
#         utils.create_dir(args.output)
#         if 0 <= args.epoch:
#             model_label += '_epoch%d' % args.epoch

#         with open(args.output+'/%s_%s.json' \
#             % (args.split, model_label), 'w') as f:
#             json.dump(results, f)

#     process(args, model, eval_loader)


if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('ban-vqa-demo/data/dictionary.pkl')
    ans2label_path = os.path.join('ban-vqa-demo/data/cache', 'trainval_ans2label.pkl')
    label2ans_path = os.path.join('ban-vqa-demo/data/cache', 'trainval_label2ans.pkl')
    ans2label = pkl.load(open(ans2label_path, 'rb'))
    label2ans = pkl.load(open(label2ans_path, 'rb'))
    num_ans_candidates = len(ans2label)

    eval_dset = VQAFeatureDataset_Custom(dictionary, len(ans2label), adaptive=True)
    print(ans2label)
    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma).cuda()

    questions = [args.questions]
    questions_token = tokenize(questions, dictionary)
    questions_tensor = tensorize(questions_token[0])
    questions_tensor = torch.unsqueeze(questions_tensor, 0)

    image = pkl.load(open(args.image, 'rb'))


    def process(model, label2ans, image, questions_tensor):
        model_path = 'ban-vqa-demo/saved_models/model_epoch12.pth'
    
        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)

        answer = get_logits(model, label2ans, image, questions_tensor)
        return answer



    answer = process(model, label2ans, image, questions_tensor)
    print(answer)
