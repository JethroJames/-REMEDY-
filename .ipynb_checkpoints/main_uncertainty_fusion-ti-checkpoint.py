import os
import argparse
import json
from datetime import datetime
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import loader
from model.model_uncertainty_fusionti import MyModel
from utils_uncertainty_fusion_batch import seed_worker, seed_everything, train, evaluate
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
parser.add_argument('--dataset', type=str, default='twitter2015', choices=['twitter2015', 'twitter2017'])
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
parser.add_argument('--num_epochs', type=int, default=10)
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
# 生成时间戳
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_weights_path = f'model_weights/2015_uncertainty_fusion-ti/{current_time}_{args.dataset}/'
model_log_path = f'log/2015_uncertainty_fusion-ti/{current_time}_{args.dataset}/'
os.makedirs(model_weights_path, exist_ok=True)
os.makedirs(model_log_path, exist_ok=True)

if args.num_workers > 0:
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

'''
    Load Dataset
'''
ner_corpus = loader.load_ner_corpus(f'resources/datasets/{args.dataset}', load_image=(args.encoder_v != ''))
ner_train_loader = DataLoader(ner_corpus.train, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers,
                            shuffle=True, worker_init_fn=seed_worker, generator=generator)
ner_dev_loader = DataLoader(ner_corpus.dev, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)
ner_test_loader = DataLoader(ner_corpus.test, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)


model = MyModel.from_pretrained(args)

params = [
    {'params': model.encoder_t.parameters(), 'lr': args.lr},
    {'params': model.head.parameters(), 'lr': args.lr * 100},
]
if args.encoder_v:
    params.append({'params': model.encoder_v.parameters(), 'lr': args.lr})
    params.append({'params': model.proj.parameters(), 'lr': args.lr * 100})
if args.rnn:
    params.append({'params': model.rnn.parameters(), 'lr': args.lr * 100})
if args.crf:
    params.append({'params': model.crf.parameters(), 'lr': args.lr * 100})
if args.gate:
    params.append({'params': model.aux_head.parameters(), 'lr': args.lr * 100})
optimizer = getattr(torch.optim, args.optim)(params)

print(args)
dev_f1s, test_f1s = [], []
ner_losses, itr_losses = [], []
best_dev_f1, best_test_report = 0, None

for epoch in range(1, args.num_epochs + 1):
    ner_loss, uncertainty = train(ner_train_loader, model, optimizer,epoch, task='ner', weight=1.0)
    ner_losses.append(ner_loss)
    print(f'loss of multimodal named entity recognition at epoch#{epoch}: {ner_loss:.2f}')

    dev_f1, dev_report = evaluate(model, ner_dev_loader,epoch)
    dev_f1s.append(dev_f1)
    test_f1, test_report = evaluate(model, ner_test_loader,epoch)
    test_f1s.append(test_f1)
    print(f'f1 score on dev set: {dev_f1:.4f}, f1 score on test set: {test_f1:.4f}')
    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_test_report = test_report

        # 保存模型权重
        model_save_name = f'model_weights/2015_uncertainty_fusion-ti/{current_time}_{args.dataset}/model_bs{args.bs}_lr{args.lr}_seed{args.seed}.pt'
        torch.save(model.state_dict(), model_save_name)
        print(f'Model weights saved to {model_save_name}')

    # 保存不确定性数据
    with open(f'model_weights/2015_uncertainty_fusion-ti/data_epoch{epoch}.pkl','wb') as f:
        pickle.dump(uncertainty,f)
 
print()
print(best_test_report)

results = {
    'config': vars(args),
    'dev_f1s': dev_f1s,
    'test_f1s': test_f1s,
    'ner_losses': ner_losses,
    # 'itr_losses': itr_losses,
}

# 修改文件名以包含时间戳
file_name = f'log/2015_uncertainty_fusion-ti/{current_time}_{args.dataset}/bs{args.bs}_lr{args.lr}_seed{args.seed}.json'
with open(file_name, 'w') as f:
    json.dump(results, f, indent=4)
