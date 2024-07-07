# import os
import argparse
import json
from datetime import datetime
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import loader2017
from model.model_uncertainty_fusion_case_t import MyModel
from utils_uncertainty_fusion_case import seed_worker, seed_everything, train, evaluate
import pdb
import netron
import torch.onnx
from torch.autograd import Variable
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
import pickle

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--dataset', type=str, default='twitter2017', choices=['twitter2015', 'twitter2017', 'case'])
parser.add_argument('--encoder_t', type=str, default='bert-base-uncased',
                    choices=['bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--encoder_v', type=str, default='resnet152', choices=['', 'resnet101', 'resnet152'])
parser.add_argument('--stacked', action='store_true', default=True)
parser.add_argument('--rnn',   action='store_true',  default=True)
parser.add_argument('--crf',   action='store_true',  default=True)
parser.add_argument('--aux',   action='store_true',  default=False)
parser.add_argument('--gate',   action='store_true',  default=False)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'AdamW'])
parser.add_argument('--caption',   action='store_true',  default=True)
parser.add_argument('--gpt1',   action='store_true',  default=False)
parser.add_argument('--gpt2',   action='store_true',  default=False)
parser.add_argument('--edl',   action='store_true',  default=True)

# EDL Parameters
parser.add_argument('--etrans_func', default='softplus', type=str, help='type of evidence')
parser.add_argument("--loss", default='edl', type=str, help='train cost function')
parser.add_argument("--use_span_weight", type=str2bool, default=False, help="range: [0,1.0], the weight of negative span for the loss.")
parser.add_argument('--annealing_start', default=0.01, type=float, help='num of random')
parser.add_argument('--annealing_step', default=10, type=float, help='num of random')
parser.add_argument('--with_uc',type=str2bool, default=False, help='')
parser.add_argument('--with_iw', type=str2bool, default=False, help='')
parser.add_argument('--with_kl', type=str2bool, default=True, help='')
parser.add_argument('--gpu',type=str2bool, default=True, help='')
parser.add_argument('--iteration', default=10, type=int, help='num of iteration')

args = parser.parse_args()

# if (args.aux or args.gate) and args.encoder_v == '':
#     raise ValueError('Invalid setting: auxiliary task or gate module must be used with visual encoder (i.e. ResNet)')

if (args.aux) and args.encoder_v == '':
    raise ValueError('Invalid setting: auxiliary task or gate module must be used with visual encoder (i.e. ResNet)')

seed_everything(args.seed)
generator = torch.Generator()
generator.manual_seed(args.seed)

ner_corpus = loader2017.load_ner_corpus(f'resources/datasets/{args.dataset}', load_image=(args.encoder_v != ''))
ner_dev_loader = DataLoader(ner_corpus.dev, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)
optimizer = getattr(torch.optim, args.optim)(params)

model = MyModel.from_pretrained(args)
# model_path = 'model_weights/2017_uncertainty_fusion/2024-04-12_11-09-04_twitter2017/model_bs4_lr1e-05_seed0.pt'
# model_path = 'model_weights/2017_uncertainty_fusion-tic/2024-04-12_11-02-47_twitter2017/model_bs4_lr1e-05_seed0.pt'
# model_path = 'model_weights/2017_uncertainty_fusion-ti/2024-04-12_01-17-44_twitter2017/model_bs4_lr1e-05_seed0.pt'
model_path = 'model_weights/2017_uncertainty_fusion-t/2024-04-17_20-09-40_twitter2017/model_bs4_lr1e-05_seed0_epoch6.pt'

# model_path = 'model_weights/2015_uncertainty_fusion/2024-04-12_00-51-07_twitter2015/model_bs4_lr1e-05_seed0.pt'
# model_path = 'model_weights/2015_uncertainty_fusion-tic/2024-04-12_11-02-23_twitter2015/model_bs4_lr1e-05_seed0.pt'
# model_path = 'model_weights/2015_uncertainty_fusion-tic/2024-04-16_14-19-29_twitter2015/model_bs4_lr1e-05_seed0_epoch2.pt'
# model_path = 'model_weights/2015_uncertainty_fusion-ti/2024-04-12_01-17-21_twitter2015/model_bs4_lr1e-05_seed0.pt'
# model_path = 'model_weights/2015t/model_bs4_lr1e-05_seed0.pt'

model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epoch=1
batch_results = evaluate(model, ner_dev_loader, epoch)



def convert_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach().tolist()  # 将 Tensor 转换为列表
    elif isinstance(obj, dict):
        return {k: convert_tensor(v) for k, v in obj.items()}  # 递归处理字典
    elif isinstance(obj, list):
        return [convert_tensor(v) for v in obj]  # 递归处理列表
    return obj

def save_results_json(batch_results, filename="case_results_t_n_2017.json"):
    converted_results = [convert_tensor(batch) for batch in batch_results]
    with open(filename, 'w') as f:
        json.dump(converted_results, f, indent=4)




save_results_json(batch_results)

