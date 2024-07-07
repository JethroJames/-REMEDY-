from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TokenEmbeddings
from flair.data import Token as FlairToken
from flair.data import Sentence as FlairSentence
from torchcrf import CRF
from data.dataset import MyDataPoint, MyPair,MyToken_Caption
import constants
import pdb
import numpy as np
from model.evidence_worker import Tagger_Evidence
from model.loss_function import get_dc_seq_loss 
# constants for model
CLS_POS = 0
SUBTOKEN_PREFIX = '##'
IMAGE_SIZE = 224
VISUAL_LENGTH = (IMAGE_SIZE // 32) ** 2


def use_cache(module: nn.Module, data_points: List[MyDataPoint]):
    for parameter in module.parameters():
        if parameter.requires_grad:
            return False
    for data_point in data_points:
        if data_point.feat is None:
            return False
    return True


def resnet_encode(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = x.view(x.size()[0], x.size()[1], -1)
    x = x.transpose(1, 2)

    return x

def weighted_evidence_fusion(evidence_dict, uncertainty_dict):
    # 计算每个视图的加权因子（这里简单地使用不确定性的倒数加1e-6避免除以0）
    weights_list = [1.0 / (uncertainty + 1e-6) for uncertainty in uncertainty_dict.values()]
    total_weight = sum(weights_list)
    normalized_weights_list = [weight / total_weight for weight in weights_list]

    # 使用加权因子对证据进行加权融合
    fused_evidence = torch.zeros_like(next(iter(evidence_dict.values())))
    for evidence, weight in zip(evidence_dict.values(), normalized_weights_list):
        fused_evidence += evidence * weight.unsqueeze(-1)

    return fused_evidence


def pad_sequences(feats_views, max_length, hid_dim_t, device='cpu'):
    """
    Pad the sequences in feats_views to make them all have the same seq_length.
    
    Args:
    - feats_views (dict): A dictionary containing the views data.
    - max_length (int): The target sequence length.
    - hid_dim_t (int): The dimension of each token/feature.
    - device (str): The device to create tensors on.

    Returns:
    - A dictionary with the same keys as feats_views, but with padded data.
    """
    padded_views = {}
    zero_tensor = torch.zeros(max_length * hid_dim_t, device=device)

    for view, data in feats_views.items():
        # Assuming data shape is [bs, seq_length, dim], and we need to pad seq_length to max_length
        batch_size, current_seq_length, dim = data.shape
        assert dim == hid_dim_t, "Dimension mismatch between data and hid_dim_t"
        
        # Calculate how much padding is needed
        num_padding = max_length - current_seq_length
        if num_padding > 0:
            # Create the padding tensor
            padding = zero_tensor[:hid_dim_t * num_padding].view(1, num_padding, hid_dim_t).repeat(batch_size, 1, 1)
            # Concatenate the original data with padding
            padded_data = torch.cat([data, padding], dim=1)
        else:
            padded_data = data
        
        # Save the padded data back to the dictionary
        padded_views[view] = padded_data
    
    return padded_views


class MyModel(nn.Module):
    def __init__(
            self,
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            encoder_t: PreTrainedModel,
            hid_dim_t: int,
            encoder_v: nn.Module = None,
            hid_dim_v: int = None,
            token_embedding: TokenEmbeddings = None,
            rnn: bool = None,
            crf: bool = None,
            gate: bool = None,
            caption: bool = None,
            gpt1: bool = None,
            gpt2: bool = None,
            edl: bool = None,
            edl_model: nn.Module = None,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.encoder_t = encoder_t
        self.hid_dim_t = hid_dim_t
        self.encoder_v = encoder_v
        self.hid_dim_v = hid_dim_v
        self.token_embedding = token_embedding
        self.proj = nn.Linear(hid_dim_v, hid_dim_t) if encoder_v else None
        self.aux_head = nn.Linear(hid_dim_t, 2)
        if self.token_embedding:
            self.hid_dim_t += self.token_embedding.embedding_length
        if rnn:
            hid_dim_rnn = 256
            num_layers = 2
            num_directions = 2
            self.rnn = nn.LSTM(self.hid_dim_t, hid_dim_rnn, num_layers, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hid_dim_rnn * num_directions, constants.LABEL_SET_SIZE)
        else:
            self.rnn = None
            self.head = nn.Linear(self.hid_dim_t, constants.LABEL_SET_SIZE) # Teacher share the same head with student

        self.crf = CRF(constants.LABEL_SET_SIZE, batch_first=True) if crf else None
        self.gate = gate
        self.caption = caption
        self.gpt1 = gpt1
        self.gpt2 = gpt2
        self.edl = edl
        self.edl_model = edl_model
        self.to(device)

    @classmethod
    def from_pretrained(cls, args):
        device = torch.device(f'cuda:{args.cuda}')
        models_path = 'resources/models'
        encoder_t_path = f'{models_path}/transformers/{args.encoder_t}'
        tokenizer = AutoTokenizer.from_pretrained(encoder_t_path)
        encoder_t = AutoModel.from_pretrained(encoder_t_path)
        config = AutoConfig.from_pretrained(encoder_t_path)
        hid_dim_t = config.hidden_size

        if args.edl:
            edl_model = Tagger_Evidence(args, constants.LABEL_SET_SIZE)

        if args.encoder_v:
            encoder_v = getattr(torchvision.models, args.encoder_v)()
            encoder_v.load_state_dict(torch.load(f'{models_path}/cnn/{args.encoder_v}.pth'))
            hid_dim_v = encoder_v.fc.in_features
        else:
            encoder_v = None
            hid_dim_v = None

        if args.stacked:
            flair.cache_root = 'resources/models'
            flair.device = device
            token_embedding = StackedEmbeddings([
                WordEmbeddings('crawl'),
                WordEmbeddings('twitter'),
                FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')
            ])
        else:
            token_embedding = None

        return cls(
            device=device,
            tokenizer=tokenizer,
            encoder_t=encoder_t,
            hid_dim_t=hid_dim_t,
            encoder_v=encoder_v,
            hid_dim_v=hid_dim_v,
            token_embedding=token_embedding,
            rnn=args.rnn,
            crf=args.crf,
            gate=args.gate,
            caption=args.caption,
            gpt1=args.gpt1,
            gpt2=args.gpt2,
            edl=args.edl,
            edl_model = edl_model
        )

    def fuse_and_process_edl_losses(self, stu_logits_batch, tea_logits_batch, labels_batch, mask_batch, epoch):
        evidences = {}
        uncertainties = {}
        edl_loss = 0

        # Student EDL processing
        edl_loss_stu, belief, evidence_stu, uncertainty_stu = self.edl_model.loss(stu_logits_batch, labels_batch, mask_batch, epoch)
        edl_loss += edl_loss_stu
        evidences[0] = evidence_stu
        uncertainties[0] = uncertainty_stu

        # Teacher EDL processing
        for view, tea_logits in tea_logits_batch.items():
            edl_loss_tea, belief_tea, evidence_tea, uncertainty_tea = self.edl_model.loss(tea_logits, labels_batch, mask_batch, epoch)
            edl_loss += edl_loss_tea
            evidences[view] = evidence_tea
            uncertainties[view] = uncertainty_tea

        # Uncertainty-driven Knowledge Fusion
        fused_evidence = weighted_evidence_fusion(evidences, uncertainties)
        
        # Combine uncertainties for all views
        all_uncertainties = list(uncertainties.values())
        # all_evidences = list(evidences.values())
        all_evidences = evidences
        return fused_evidence, all_evidences, all_uncertainties, edl_loss




    def process_teacher_features(self, sentence_batch, ids_batch, offset_batch, mask_batch, feat_batches):
        processed_feats = {key: None for key in feat_batches}  # 准备字典来存储每个特征批次的处理结果

        # 首先计算最大的特征列表长度
        max_length = 0
        for key, feat_batch in feat_batches.items():
            for ids, offset, mask in zip(ids_batch, offset_batch, mask_batch):
                ids = ids[mask]
                offset = offset[mask]
                subtokens = self.tokenizer.convert_ids_to_tokens(ids)
                length = len(subtokens)
                i = 0
                current_length = 0
                while i < length:
                    j = i + 1
                    while j < length and (offset[j][0] != 0 or subtokens[j].startswith(SUBTOKEN_PREFIX)):
                        j += 1
                    current_length += 1
                    i = j
                max_length = max(max_length, current_length)

        # 处理每个特征批次
        for key, feat_batch in feat_batches.items():
            # 预分配足够大的张量
            all_features_tensor = torch.zeros((len(sentence_batch), max_length, self.hid_dim_t), dtype=torch.float)

            for idx, (sentence, ids, offset, mask, fea) in enumerate(zip(sentence_batch, ids_batch, offset_batch, mask_batch, feat_batch)):
                ids = ids[mask]
                offset = offset[mask]
                subtokens = self.tokenizer.convert_ids_to_tokens(ids)
                length = len(subtokens)
                
                feat = fea[mask]  # 应用掩码
                feat_list = []
                i = 0
                while i < length:
                    j = i + 1
                    while j < length and (offset[j][0] != 0 or subtokens[j].startswith(SUBTOKEN_PREFIX)):
                        j += 1
                    feat_list.append(torch.mean(feat[i:j], dim=0))
                    i = j

                assert len(sentence) == len(feat_list)
                
                # Embedding with FlairSentence
                flair_sentence = FlairSentence(str(sentence))
                self.token_embedding.embed(flair_sentence)
                enhanced_feat_list = []
                for flair_token, original_feat in zip(flair_sentence, feat_list):
                    # Concatenate the original feature with the Flair embedding
                    enhanced_feat = torch.cat((original_feat, flair_token.embedding), dim=0)
                    enhanced_feat_list.append(enhanced_feat)

                # 用enhanced_feat填充预分配的张量
                for feat_idx, enhanced_feat in enumerate(enhanced_feat_list):
                    all_features_tensor[idx, feat_idx, :] = enhanced_feat

            processed_feats[key] = all_features_tensor

        return processed_feats

    def process_views(self, feats_views, lengths):
        """
        Process each view in feats_views with RNN.
        
        Args:
        - feats_views (dict): A dictionary containing the padded views data.
        - lengths (list): A list of original sequence lengths before padding.
        - rnn_model (nn.Module): An RNN model to process the sequences.
        
        Returns:
        - A dictionary with the same keys as feats_views, but where each item has been
        processed by the RNN.
        """
        processed_views = {}

        for view, data in feats_views.items():
            # Assuming data is already padded and has shape [bs, seq_length, dim]
            
            # Convert lengths to a tensor
            lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
            
            # Pack the sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(data, lengths_tensor, batch_first=True, enforce_sorted=False)
            
            # Process sequences through the RNN
            packed_output, _ = self.rnn(packed_input.to(self.device))
            
            # Unpack the sequences
            unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Save the processed data back to the dictionary
            processed_views[view] = unpacked_output
        
        return processed_views

# 

    def _prepare_components(self, inputs, pairs):
        images = [pair.image for pair in pairs]
        caption_batch = [pair.caption for pair in pairs]
        gpt1_batch = [pair.gpt1 for pair in pairs]
        gpt2_batch = [pair.gpt2 for pair in pairs]
        # query
        textual_embeds = self.encoder_t.embeddings.word_embeddings(inputs.input_ids)
        # image
        images = [pair.image for pair in pairs]
        visual_embeds = torch.stack([image.data for image in images]).to(self.device) 
        if not use_cache(self.encoder_v, images):
            visual_embeds = resnet_encode(self.encoder_v, visual_embeds) # [bs, 49, 768]
        visual_embeds = self.proj(visual_embeds)
        # caption
        caption_inputs = self.tokenizer(caption_batch, padding=True, return_tensors='pt', return_special_tokens_mask=True)
        caption_input_ids = caption_inputs.input_ids.to(self.device)
        caption_length = caption_input_ids.size()[1]
        caption_attention_mask = caption_inputs.attention_mask.to(self.device)
        textual_embeds_captions = self.encoder_t.embeddings.word_embeddings(caption_input_ids)
        # gpt1
        gpt1_batch = [pair.gpt1 for pair in pairs]
        gpt1_inputs = self.tokenizer(gpt1_batch, padding=True, return_tensors='pt', return_special_tokens_mask=True)
        gpt1_input_ids = gpt1_inputs.input_ids.to(self.device)
        gpt1_length = gpt1_input_ids.size()[1]
        gpt1_attention_mask = gpt1_inputs.attention_mask.to(self.device)
        textual_embeds_gpt1 = self.encoder_t.embeddings.word_embeddings(gpt1_input_ids)
        # gpt2
        gpt2_batch = [pair.gpt2 for pair in pairs]
        gpt2_inputs = self.tokenizer(gpt2_batch, padding=True, return_tensors='pt', return_special_tokens_mask=True)
        gpt2_input_ids = gpt2_inputs.input_ids.to(self.device)
        gpt2_length = gpt2_input_ids.size()[1]
        gpt2_attention_mask = gpt2_inputs.attention_mask.to(self.device)
        textual_embeds_gpt2 = self.encoder_t.embeddings.word_embeddings(gpt2_input_ids)
        return {
            "textual_embeds":textual_embeds,
            "visual_embeds": visual_embeds,
            "textual_embeds_captions": textual_embeds_captions, "caption_inputs": caption_inputs,"caption_length":caption_length,
            "textual_embeds_gpt1": textual_embeds_gpt1, "gpt1_inputs": gpt1_inputs,"gpt1_length":gpt1_length,
            "textual_embeds_gpt2": textual_embeds_gpt2, "gpt2_inputs": gpt2_inputs,"gpt2_length":gpt2_length,
        }
    def _bert_forward(self,inputs, components, view=None):
        textual_embeds=components["textual_embeds"]
        visual_embeds=components["visual_embeds"]
        textual_embeds_captions=components["textual_embeds_captions"]
        textual_embeds_gpt1=components["textual_embeds_gpt1"]
        textual_embeds_gpt2=components["textual_embeds_gpt2"]
        caption_length = components["caption_length"]
        gpt1_length = components["gpt1_length"]
        gpt2_length = components["gpt2_length"]
        if view == "T + I":
            inputs_embeds = torch.concat((textual_embeds, visual_embeds), dim=1)
            batch_size = visual_embeds.size()[0]
            visual_length = visual_embeds.size()[1]

            attention_mask = inputs.attention_mask
            visual_mask = torch.ones((batch_size, visual_length), dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat((attention_mask, visual_mask), dim=1)

            token_type_ids = inputs.token_type_ids
            visual_type_ids = torch.ones((batch_size, visual_length), dtype=token_type_ids.dtype, device=self.device)
            token_type_ids = torch.cat((token_type_ids, visual_type_ids), dim=1)

            return self.encoder_t(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
        elif view == "T + C":
            inputs_embeds = torch.cat((textual_embeds, textual_embeds_captions), dim=1)
            batch_size = inputs_embeds.size(0)
            attention_mask = inputs.attention_mask
            caption_mask = torch.ones((batch_size, caption_length), dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat((attention_mask, caption_mask), dim=1)

            token_type_ids = inputs.token_type_ids
            caption_type_ids = torch.ones((batch_size, caption_length), dtype=token_type_ids.dtype, device=self.device)
            token_type_ids = torch.cat((token_type_ids, caption_type_ids), dim=1)

            return self.encoder_t(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            ),caption_length
        elif view == "T + C + G1":
            inputs_embeds = torch.cat((textual_embeds, textual_embeds_captions, textual_embeds_gpt1), dim=1)
            batch_size = inputs_embeds.size(0)
            total_length = caption_length + gpt1_length

            attention_mask = torch.cat((inputs.attention_mask,
                                    torch.ones((batch_size, total_length), dtype=inputs.attention_mask.dtype, device=self.device)), dim=1)

            token_type_ids = torch.cat((inputs.token_type_ids,
                                    torch.ones((batch_size, caption_length), dtype=inputs.token_type_ids.dtype, device=self.device),
                                    torch.ones((batch_size, gpt1_length), dtype=inputs.token_type_ids.dtype, device=self.device)), dim=1)
        

            return self.encoder_t(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            ),total_length
        elif view == "T + C + G2":
            inputs_embeds = torch.cat((textual_embeds, textual_embeds_captions, textual_embeds_gpt1, textual_embeds_gpt2), dim=1)
            batch_size = inputs_embeds.size(0)
            total_length = caption_length + gpt1_length + gpt2_length

            attention_mask = torch.cat((inputs.attention_mask,
                                    torch.ones((batch_size, total_length), dtype=inputs.attention_mask.dtype, device=self.device)), dim=1)

            token_type_ids = torch.cat((inputs.token_type_ids,
                                    torch.ones((batch_size, caption_length), dtype=inputs.token_type_ids.dtype, device=self.device),
                                    torch.ones((batch_size, gpt1_length), dtype=inputs.token_type_ids.dtype, device=self.device),
                                    torch.ones((batch_size, gpt2_length), dtype=inputs.token_type_ids.dtype, device=self.device)),dim=1)
        

            return self.encoder_t(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            ),total_length


    def ner_encode(self, pairs: List[MyPair], gate_signal=None):
        sentence_batch = [pair.sentence for pair in pairs]  # len = bs
        tokens_batch = [[token.text for token in sentence] for sentence in sentence_batch] # len = bs
        
        # T view  inputs['input_ids'].size() = [bs,max_seq_len]
        inputs = self.tokenizer(tokens_batch, is_split_into_words=True, padding=True, return_tensors='pt',
                                return_special_tokens_mask=True, return_offsets_mapping=True).to(self.device)
        
        T = self.encoder_t(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            token_type_ids=inputs.token_type_ids,
            return_dict=True
        )
        T_feat_batch = T.last_hidden_state
        
        components =  self._prepare_components(inputs, pairs)

        # T + I view
        view="T + I"
        INT = self._bert_forward(inputs,components, view)
        INT_feat_batch = INT.last_hidden_state[:, :-VISUAL_LENGTH]
        
        # T + C view 
        view="T + C"
        INC,caption_length = self._bert_forward(inputs,components,view)
        INC_feat_batch = INC.last_hidden_state[:, :-caption_length]

        # T + C + G1 view
        view="T + C + G1"
        ICG1, ICG1_length = self._bert_forward(inputs,components,view)
        ICG1_feat_batch = ICG1.last_hidden_state[:, :-ICG1_length]

        # T + C + G2 view
        view="T + C + G2"
        ICG2, ICG2_length = self._bert_forward(inputs,components,view)
        ICG2_feat_batch = ICG2.last_hidden_state[:, :-ICG2_length]

        # collect all views
        # shape: torch.Size([4, 30, 768])
        Teachers_feats ={
            1: INT_feat_batch, # T+I
            2: INC_feat_batch, # T+C
            3: ICG1_feat_batch, # T+C+G1
            4: ICG2_feat_batch # T+C+G2
        }    
        
        ids_batch = inputs.input_ids
        offset_batch = inputs.offset_mapping
        mask_batch = inputs.special_tokens_mask.bool().bitwise_not()

        # Get Teacher features
        Teacher_feats = self.process_teacher_features(
            sentence_batch,ids_batch,offset_batch,mask_batch,Teachers_feats)
       

        # Obtain Student features
        for sentence, ids, offset, mask, feat in zip(sentence_batch, ids_batch, offset_batch, mask_batch, T_feat_batch):
            ids = ids[mask]
            offset = offset[mask]
            feat = feat[mask]
            subtokens = self.tokenizer.convert_ids_to_tokens(ids)
            length = len(subtokens)

            token_list = []
            feat_list = []
            i = 0
            while i < length:
                j = i + 1
                # the 'or' condition is for processing Korea characters
                while j < length and (offset[j][0] != 0 or subtokens[j].startswith(SUBTOKEN_PREFIX)):
                    j += 1
                token_list.append(''.join(subtokens[i:j]))
                feat_list.append(torch.mean(feat[i:j], dim=0))
                i = j
            assert len(sentence) == len(token_list)

            # Put feat in feat_list into MyPair - MyToken Objects
            for token, token_feat in zip(sentence, feat_list):
                token.feat = token_feat
            
            # Enhance tokens embedding in MyPairs 
            if self.token_embedding is not None:
                flair_sentence = FlairSentence(str(sentence))
                flair_sentence.tokens = [FlairToken(token.text) for token in sentence]
                self.token_embedding.embed(flair_sentence)
                for token, flair_token in zip(sentence, flair_sentence):
                    token.feat = torch.cat((token.feat, flair_token.embedding))
        
        
        return Teacher_feats, mask_batch


    def ner_forward(self, pairs: List[MyPair], epoch):
        gate_signal = None
        feats_views, mask_batch = self.ner_encode(pairs, gate_signal)

        # fetch batch MyPairs
        sentences = [pair.sentence for pair in pairs]
        batch_size = len(sentences)
        lengths = [len(sentence) for sentence in sentences]
        max_length = max(lengths)

        # Obtain T tokens
        feat_list = []
        zero_tensor = torch.zeros(max_length * self.hid_dim_t, device=self.device)
        for sentence in sentences:
            # Obtain equal length of feature vectors
            feat_list += [token.feat for token in sentence]
            num_padding = max_length - len(sentence)
            if num_padding > 0:
                padding = zero_tensor[:self.hid_dim_t * num_padding]
                feat_list.append(padding)


        feats = torch.cat(feat_list).view(batch_size, max_length, self.hid_dim_t)
        padded_teachers_views = pad_sequences(feats_views, max_length, self.hid_dim_t, device=self.device)
        
        
        if self.rnn is not None:
            # Student view
            feats = nn.utils.rnn.pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
            feats, _ = self.rnn(feats)
            feats, _ = nn.utils.rnn.pad_packed_sequence(feats, batch_first=True)

            # Teachers view
            processed_teachers_views = self.process_views(padded_teachers_views, lengths)


        
        # Obtain student logits
        stu_logits_batch = self.head(feats) # stu_logits_batch: [bs, seq_length, class_num]
        tea_logits_batch = {}
        for view, data in processed_teachers_views.items():
            tea_logits_batch[view] = self.head(data)
        
 

        # Obtain labels
        labels_batch = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        for i, sentence in enumerate(sentences):
            labels = torch.tensor([token.label for token in sentence], dtype=torch.long, device=self.device)
            labels_batch[i, :lengths[i]] = labels


        # EDL
        if self.edl:
            '''
                Uncertainty-driven Knowledge Fusion            
            '''
            # README 
            # fused_evidence: [bs,seq_lengtn,class_num]
            # evidences: dict -> keys: [0,1,2,3,4]
            # all_uncertainties: list -> len: view_num, element: Tensor -> size:[bs,seq_length]

            fused_evidence, evidences, all_uncertainties, edl_loss = self.fuse_and_process_edl_losses(stu_logits_batch, tea_logits_batch, labels_batch, mask_batch, epoch)
            
            
            evidence_a = fused_evidence
            edl_loss_A, belief_a, evidence_A, uncertainty_A = self.edl_model.loss(evidence_a, labels_batch, mask_batch, epoch)
            edl_loss += edl_loss_A
            edl_loss = edl_loss / (len(evidence_a) + 1) # fusion view & tea-stu views
            # Robust conflict multi-view learning
            # rcml_loss = get_dc_seq_loss(evidences, self.device)

        if self.crf:
            mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=self.device)
            for i in range(batch_size):
                mask[i, :lengths[i]] = 1
            crf_loss = -self.crf(evidence_a, labels_batch, mask, reduction='mean')
            pred_ids = self.crf.decode(stu_logits_batch, mask)
            pred = [[constants.ID_TO_LABEL[i] for i in ids] for ids in pred_ids]


        else:
            loss = torch.zeros(1, device=self.device)
            for logits, labels, length in zip(stu_logits_batch, labels_batch, lengths):
                loss += F.cross_entropy(logits[:length], labels[:length], reduction='sum')
            loss /= batch_size
            pred_ids = torch.argmax(stu_logits_batch, dim=2).tolist()
            pred = [[constants.ID_TO_LABEL[i] for i in ids[:length]] for ids, length in zip(pred_ids, lengths)]

        loss = edl_loss + crf_loss

        return loss, pred, all_uncertainties
