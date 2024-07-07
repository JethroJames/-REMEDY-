import os
import random
import numpy as np
import torch
import constants
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
import pdb
from torch.nn.functional import softmax

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
    model.train()
    for batch in tqdm(loader):
        # pdb.set_trace()
        optimizer.zero_grad()
        
        loss, _, batch_uncertainty = getattr(model, f'{task}_forward')(batch, epoch)
        loss *= weight
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        uncertainty.append(batch_uncertainty)
    
    return np.mean(losses), uncertainty


# def evaluate(model, loader, epoch):
#     true_labels = []
#     pred_labels = []
#     batch_data = []
#     uncertainty = []

#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm(loader):
#             _, pred, batch_uncertainty = model.ner_forward(batch, epoch)
            
#             # 保存预测结果
#             pred_labels += pred
            
#             # 保存真实标签
#             current_true_labels = [
#                 ([constants.ID_TO_LABEL[token.label] for token in pair.sentence] if pair.sentence else []) 
#                 for pair in batch if pair.sentence
#             ]
#             true_labels += current_true_labels
            
#             # 保存不确定性
#             uncertainty.append(batch_uncertainty)
            
#             # 保存批次数据，包括句子、真实标签、预测标签和图像ID
#             current_batch_data = [
#                 {
#                     'sentence': [token.text for token in pair.sentence],
#                     'true_label': labels,
#                     'pred_label': preds,
#                     'image_id': pair.image.file_name
#                 }
#                 for pair, labels, preds in zip(batch, current_true_labels, pred)
#             ]
#             batch_data += current_batch_data
    
#     # 计算评估指标
#     f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
#     report = classification_report(true_labels, pred_labels, digits=4, mode='strict', scheme=IOB2, zero_division=1)

#     return f1, report, uncertainty, batch_data

def normalize(values):
    min_val = torch.min(values)
    max_val = torch.max(values)
    normalized = (values - min_val) / (max_val - min_val) if max_val > min_val else values.clone()
    return normalized
def evaluate(model, loader, epoch):
    batch_results = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            _, pred, batch_uncertainty, belief_a, uncertainty_A, evidence_A = model.ner_forward(batch, epoch)
            
            # Apply softmax to the last dimension of evidence_A
            evidence_A_softmax = softmax(evidence_A, dim=-1)  # This applies softmax across the last dimension (dim=2)
            
            # normalized_uncertainty_A = []
            # for idx, pair in enumerate(batch):
            #     if pair.sentence:
                    # token_count = len(pair.sentence)
                    # print("sentence",pair.sentence)
                    # print("token_count",len(pair.sentence))
                    # print(uncertainty_A[idx][:token_count])
                    
                    # Normalize only the relevant portion of uncertainty_A
                    # sentence_uncertainty = normalize(uncertainty_A[idx][:token_count])
                    # If necessary, append the remaining unnormalized values
                    # remaining_uncertainty = uncertainty_A[idx][token_count:]
                    # normalized_uncertainty_A.append(torch.cat([sentence_uncertainty, remaining_uncertainty]))
                    # print("normalized_uncertainty_A",normalized_uncertainty_A)
                # else:
                    # If there is no sentence, keep the original uncertainty values
                    # normalized_uncertainty_A.append(uncertainty_A[idx])

            # Convert the list back to the original type (e.g., tensor, list of tensors)
            # depending on how your model expects uncertainty_A to be formatted
            # uncertainty_A = torch.stack(normalized_uncertainty_A) if isinstance(uncertainty_A, torch.Tensor) else normalized_uncertainty_A
            
            true_labels = [([constants.ID_TO_LABEL[token.label] for token in pair.sentence] if pair.sentence else []) 
                           for pair in batch if pair.sentence]
            pred_labels = pred
            
            # Calculate f1 and classification report for the current batch
            batch_f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
            batch_report = classification_report(true_labels, pred_labels, digits=4, mode='strict', scheme=IOB2, zero_division=1)
            
            # Store batch-specific results including the softmax-transformed evidence_A
            batch_results.append({
                "batch_f1_score": batch_f1,
                "batch_classification_report": batch_report,
                "batch_uncertainty": batch_uncertainty,
                "belief_a": belief_a,
                "uncertainty_A": uncertainty_A,
                "evidence_A_softmax": evidence_A_softmax,  # Adding softmaxed evidence_A to results
                "batch_details": [
                    {
                        'sentence': [token.text for token in pair.sentence],
                        'true_label': labels,
                        'pred_label': preds,
                        'image_id': pair.image.file_name
                    }
                    for pair, labels, preds in zip(batch, true_labels, pred_labels)
                ]
            })

    return batch_results
# def evaluate(model, loader, epoch):
#     batch_results = []

#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm(loader):
#             _, pred, batch_uncertainty, belief_a, uncertainty_A, evidence_A = model.ner_forward(batch, epoch)



            
#             true_labels = [([constants.ID_TO_LABEL[token.label] for token in pair.sentence] if pair.sentence else []) 
#                            for pair in batch if pair.sentence]
#             pred_labels = pred
            
#             # Calculate f1 and classification report for the current batch
#             batch_f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
#             batch_report = classification_report(true_labels, pred_labels, digits=4, mode='strict', scheme=IOB2, zero_division=1)
            
#             # Store batch-specific results
#             batch_results.append({
#                 "batch_f1_score": batch_f1,
#                 "batch_classification_report": batch_report,
#                 "batch_uncertainty": batch_uncertainty,
#                 "belief_a": belief_a,
#                 "uncertainty_A": uncertainty_A,
#                 "batch_details": [
#                     {
#                         'sentence': [token.text for token in pair.sentence],
#                         'true_label': labels,
#                         'pred_label': preds,
#                         'image_id': pair.image.file_name
#                     }
#                     for pair, labels, preds in zip(batch, true_labels, pred_labels)
#                 ]
#             })

#     return batch_results


# import torch
# from tqdm import tqdm
# from seqeval.metrics import f1_score, classification_report
# from seqeval.scheme import IOB2

# # Assuming constants and the necessary imports are defined elsewhere in your code.
# # Assuming model, loader, and epoch are properly defined and passed to this function.

# def normalize(values):
#     min_val = torch.min(values)
#     max_val = torch.max(values)
#     normalized = (values - min_val) / (max_val - min_val) if max_val > min_val else values.clone()
#     return normalized

# def evaluate(model, loader, epoch):
#     batch_results = []

#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm(loader):
#             _, pred, batch_uncertainty, belief_a, uncertainty_A = model.ner_forward(batch, epoch)
            
#             # Create a new list for the normalized uncertainty values
#             normalized_uncertainty_A = []
#             for idx, pair in enumerate(batch):
#                 if pair.sentence:
#                     token_count = len(pair.sentence)
#                     # Normalize only the relevant portion of uncertainty_A
#                     sentence_uncertainty = normalize(uncertainty_A[idx][:token_count])
#                     # If necessary, append the remaining unnormalized values
#                     remaining_uncertainty = uncertainty_A[idx][token_count:]
#                     normalized_uncertainty_A.append(torch.cat([sentence_uncertainty, remaining_uncertainty]))
#                 else:
#                     # If there is no sentence, keep the original uncertainty values
#                     normalized_uncertainty_A.append(uncertainty_A[idx])

#             # Convert the list back to the original type (e.g., tensor, list of tensors)
#             # depending on how your model expects uncertainty_A to be formatted
#             uncertainty_A = torch.stack(normalized_uncertainty_A) if isinstance(uncertainty_A, torch.Tensor) else normalized_uncertainty_A

#             true_labels = [([constants.ID_TO_LABEL[token.label] for token in pair.sentence] if pair.sentence else []) 
#                            for pair in batch if pair.sentence]
#             pred_labels = pred
            
#             batch_f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
#             batch_report = classification_report(true_labels, pred_labels, digits=4, mode='strict', scheme=IOB2, zero_division=1)
            
#             # Store the results for this batch
#             batch_results.append({
#                 "batch_f1_score": batch_f1,
#                 "batch_classification_report": batch_report,
#                 "batch_uncertainty": batch_uncertainty,
#                 "belief_a": belief_a,
#                 "uncertainty_A": uncertainty_A,  # This is now the list of normalized tensors
#                 "batch_details": [
#                     {
#                         'sentence': [token.text for token in pair.sentence],
#                         'true_label': labels,
#                         'pred_label': preds,
#                         'image_id': pair.image.file_name
#                     }
#                     for pair, labels, preds in zip(batch, true_labels, pred_labels)
#                 ]
#             })

#     # Return the accumulated results for all batches
#     return batch_results

# Call the evaluate function with the necessary arguments
# results = evaluate(model, loader, epoch)
