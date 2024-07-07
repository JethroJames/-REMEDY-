import re
import csv
import sys
sys.path.append("./data")
from pathlib import Path
from collections import Counter
from data.dataset import MyToken, MySentence, MyImage, MyPair, MyDataset, MyCorpus,MyDataset_Caption,MyPair_Caption,MyCorpus_Caption,MyDataset_gpt1,MyPair_gpt1,MyCorpus_gpt1,MyDataset_gpt2,MyPair_gpt2,MyCorpus_gpt2
import constants
from PIL import Image
from PIL import UnidentifiedImageError
from pathlib import Path
import pdb


# constants for preprocessing
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
IMGID_PREFIX = 'IMGID:'
URL_PREFIX = 'http://t.co/'
UNKNOWN_TOKEN = '[UNK]'
image_id_pattern = re.compile(r'^O_[0-9]+$|^[0-9]+_[0-9]+_[0-9]+$|^[0-9]+_[0-9]+_[0-9]+_[0-9]+$|^[0-9]+$')


def normalize_text(text: str):
    # remove the ending URL which is not part of the text
    url_re = r' http[s]?://t.co/\w+$'
    text = re.sub(url_re, '', text)
    return text

def load_ner_dataset(path_to_txt: Path) -> MyDataset:
    tokens = []
    pairs = []
    collecting_caption = False
    caption = ""

    with open(str(path_to_txt), encoding='utf-8') as txt_file:
        for line in txt_file:
            line = line.rstrip()  # strip '\n'

            if line:
                # print("1111",line)
                text, label = line.split()
                if text == '' or text.isspace() or text in SPECIAL_TOKENS or text.startswith(URL_PREFIX):
                    text = UNKNOWN_TOKEN
                if label == 'B-X' and not collecting_caption:
                    collecting_caption = True
                    if text != "<EOS>":
                        caption += text  
                elif collecting_caption:
                    if text != "<EOS>":
                        caption += " " + text
                else:
                    tokens.append(MyToken(text, constants.LABEL_TO_ID[label]))
            else:
                if tokens or caption:
                    sentence = MySentence(tokens)
                    pairs.append(MyPair(MySentence(tokens), MyImage('1.jpg'), caption, '', ''))
                    tokens = []  
                    caption = ""  
                    collecting_caption = False
    
    # Handle the last sentence if there is any
    if tokens or caption:
        sentence = MySentence(tokens)
        pairs.append(MyPair(MySentence(tokens), None, caption, '', ''))

    return MyDataset(pairs, '')


def load_ner_corpus(path: str, load_image: bool = False) -> MyCorpus:
    path = Path(path)
    path_to_train_file = path / 'train.txt'
    path_to_dev_file = path / 'dev.txt'
    path_to_test_file = path / 'test.txt'

    assert path_to_train_file.exists()
    assert path_to_dev_file.exists()
    assert path_to_test_file.exists()

    train = load_ner_dataset(path_to_train_file)
    dev = load_ner_dataset(path_to_dev_file)
    test = load_ner_dataset(path_to_test_file)

    return MyCorpus(train, dev, test)

def load_captions(caption_file: Path) -> dict:
    caption_dict = {}
    with open(str(caption_file), encoding='utf-8') as cap_file:
        for line in cap_file:
            image_id, caption = line.split(":", 1)
            image_id = image_id.split('.')[0].strip()
            caption_dict[image_id] = caption.strip()
    return caption_dict

def load_gpt_data(path_to_gpt: Path) -> dict:
    gpt_dict = {}
    with open(str(path_to_gpt), encoding='utf-8') as gpt_file:
        current_id = None
        content_lines = []
        for line in gpt_file:
            line = line.rstrip()
            if image_id_pattern.match(line):  
                if current_id:
                    gpt_dict[current_id] = '\n'.join(content_lines)
                current_id = line
                content_lines = []
            else:
                content_lines.append(line)
        if current_id:
            gpt_dict[current_id] = '\n'.join(content_lines)
    return gpt_dict



def type_count(dataset: MyDataset) -> str:
    tags = [token.label for pair in dataset for token in pair.sentence]
    counter = Counter(tags)

    num_total = len(dataset)
    num_per = counter['B-PER']
    num_loc = counter['B-LOC']
    num_org = counter['B-ORG']
    num_misc = counter['B-MISC']

    return f'{num_total}\t{num_per}\t{num_loc}\t{num_org}\t{num_misc}'


def token_count(dataset: MyDataset) -> str:
    lengths = [len(pair.sentence) for pair in dataset]
    num_sentences = len(lengths)
    num_tokens = sum(lengths)

    return f'{num_sentences}\t{num_tokens}'


if __name__ == "__main__":
    twitter2015 = load_ner_corpus('resources/datasets/snap')
    print('text',twitter2015.test.pairs[0].sentence.tokens[1].text)
    print('caption',twitter2015.test.pairs[0].caption)
    # print('gpt2',twitter2015.train.pairs[20].gpt2)
#     twitter2015 = load_ner_corpus('resources/datasets/twitter2015')
#     twitter2015_train_statistic = type_count(twitter2015.train)
#     twitter2015_dev_statistic = type_count(twitter2015.dev)
#     twitter2015_test_statistic = type_count(twitter2015.test)
#     assert twitter2015_train_statistic == '4000\t2217\t2091\t928\t940'
#     assert twitter2015_dev_statistic == '1000\t552\t522\t247\t225'
#     assert twitter2015_test_statistic == '3257\t1816\t1697\t839\t726'

#     print('-----------------------------------------------')
#     print('2015\tNUM\tPER\tLOC\tORG\tMISC')
#     print('-----------------------------------------------')
#     print('TRAIN\t' + twitter2015_train_statistic)
#     print('DEV\t' + twitter2015_dev_statistic)
#     print('TEST\t' + twitter2015_test_statistic)
#     print('-----------------------------------------------')

#     print()

    # twitter2017 = load_ner_corpus('resources/datasets/twitter2017')
    # twitter2017_train_statistic = token_count(twitter2017.train)
    # twitter2017_dev_statistic = token_count(twitter2017.dev)
    # twitter2017_test_statistic = token_count(twitter2017.test)
    # assert twitter2017_train_statistic == '4290\t68655'
    # assert twitter2017_dev_statistic == '1432\t22872'
    # assert twitter2017_test_statistic == '1459\t23051'

    # print('------------------------')
    # print('2017\tSENT.\tTOKEN')
    # print('------------------------')
    # print('TRAIN\t' + twitter2017_train_statistic)
    # print('DEV\t' + twitter2017_dev_statistic)
    # print('TEST\t' + twitter2017_test_statistic)
    # print('------------------------')

