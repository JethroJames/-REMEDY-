import os
import random
import numpy as np
import torch
import constants
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
import pdb


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def train(loader, model, optimizer, epoch, task, weight=1.0):
    losses = []
    uncertainty = []
    sentence = []
    batch_data_list=[]
    # model.train()
    for batch in tqdm(loader):
        batch_list = []  # 存储单个批次的数据
        for pair in batch:
            if pair.sentence:  # 确保sentence存在
                # 对于每个sentence，提取其所有tokens的text和label
                true_labels = [constants.ID_TO_LABEL[token.label] for token in pair.sentence]
                batch_list.append(true_labels)  # 将当前pair的数据添加到当前批次列表
                sentence.append([token.text for token in pair.sentence])
        batch_data_list.append(batch_list)  # 将当前批次列表添加到所有批次的数据列表
    
    return batch_data_list,sentence


def evaluate(model, loader, epoch):
    true_labels = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            # pdb.set_trace()
            _, pred, batch_uncertainty = model.ner_forward(batch, epoch)
            # print("pred",pred)
            # true_labels += [[constants.ID_TO_LABEL[token.label] for token in pair.sentence] for pair in batch]
            true_labels += [([constants.ID_TO_LABEL[token.label] for token in pair.sentence] if pair.sentence else []) for pair in batch if pair.sentence]
            pred_labels += pred
    
    # print("true_label:",true_labels)
    # print("pred_label:",pred_labels)
    f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
    report = classification_report(true_labels, pred_labels, digits=4, mode='strict', scheme=IOB2, zero_division=1)

    return f1, report
