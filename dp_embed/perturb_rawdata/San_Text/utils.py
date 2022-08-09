from tqdm import tqdm
import os
import unicodedata
from collections import Counter
import pandas as pd

def word_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def get_vocab(data_dir, data_file, tokenizer, tokenizer_type="subword"):
    vocab = Counter()
    total_num = sum([1 for i in open(data_file, "r", encoding='ISO-8859-1')])
    print("get_vocab from:",data_file)
    print(data_dir+'/corpus_vocab')
    if os.path.exists(os.path.join(data_dir, 'corpus_vocab')):
        print("read file from %s"%data_dir+'/corpus_vocab')
        with open(os.path.join(data_dir, 'corpus_vocab'),encoding='ISO-8859-1') as f:
            lines = f.readlines()
            for line in lines:
                w,fre = line.strip().split()
                if(int(fre)>=5):
                    vocab[w] = int(fre)

    else:
        with open(data_file, 'r', encoding='ISO-8859-1') as f:
            for idx, line in tqdm(enumerate(f), total=total_num):
                ele_list = line.strip().split('\t\t')
                if (len(ele_list) < 2):
                    continue
                text = ele_list[1]
                if tokenizer_type == "subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type == "word":
                    tokenized_text = [token.text.lower() for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token] += 1
            if tokenizer_type == "subword":
                for token in tokenizer.vocab:
                    vocab[token] += 1
            with open(os.path.join(data_dir, 'corpus_vocab'), 'w', encoding='ISO-8859-1') as f:
                for w, fre in vocab.items():
                    f.write(w)
                    f.write('\t\t')
                    f.write(str(fre))
                    f.write('\r')

    return vocab


def get_vocab_weibo(dataset,data_dir, data_file, tokenizer, tokenizer_type="subword"):
    vocab = Counter()
    print("get_vocab from:",data_file)
    print(data_dir+'/%s_corpus_vocab'%dataset)
    if os.path.exists(os.path.join(data_dir, dataset + '_corpus_vocab')):
        print("read file from %s"%data_dir+'/%s_corpus_vocab' % dataset)
        with open(os.path.join(data_dir, dataset + '_corpus_vocab'),encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                eles = line.strip().split()
                if len(eles)>1:
                    w = eles[0]
                    fre = eles[1]
                    if(int(fre)>=5):
                        vocab[w] = int(fre)

    else:
        text = pd.DataFrame(pd.read_csv(data_file)).loc[:,'con'].values.tolist()
        for line in tqdm(text):
            if tokenizer_type == "subword":
                tokenized_text = tokenizer.tokenize(line)
            elif tokenizer_type == "word":
                tokenized_text = [token.text.lower() for token in tokenizer(line)]
            for token in tokenized_text:
                vocab[token] += 1
        if tokenizer_type == "subword":
            for token in tokenizer.vocab:
                vocab[token] += 1
        with open(os.path.join(data_dir, dataset + '_corpus_vocab'), 'w', encoding='utf-8') as f:
            for w, fre in vocab.items():
                f.write(w)
                f.write('\t\t')
                f.write(str(fre))
                f.write('\r')

    return vocab



def get_vocab_foursquare(data_dir,data_file,tokenizer,tokenizer_type="subword"):
    if not os.path.exists(os.path.join(data_dir,'corpus_vocab')):
        vocab = Counter()
        with open(data_file,'r',encoding='ISO-8859-1') as f:
            num_lines = f.readlines()
        for line in num_lines:
            text = line.strip().split()[2].replace('_',' ')
            if tokenizer_type == "subword":
                tokenized_text = tokenizer.tokenize(text)
            elif tokenizer_type == "word":
                tokenized_text = [token.text.lower() for token in tokenizer(text)]
            for token in tokenized_text:
                vocab[token] +=1
        if tokenizer_type == "subword":
            for token in tokenizer.vocab:
                vocab[token]+=1
        with open(os.path.join(data_dir,'corpus_vocab'), 'w',encoding='ISO-8859-1') as f:
            for w,fre in vocab.items():
                f.write(w)
                f.write('\t\t')
                f.write(str(fre))
                f.write('\r')
    else:
        vocab = Counter()
        with open(os.path.join(data_dir,'corpus_vocab'), 'r',encoding='ISO-8859-1') as f:
            lines = f.readlines()
            for line in lines:
                for combine in line.strip().split('\\r'):
                    print(combine)
                    w, fre = combine.strip().split('\\t\\t')
                    vocab[w] = str(fre)
    return vocab


def get_vocab_twitter1(data_dir,data_file,tokenizer,tokenizer_type="subword"):
    vocab = Counter()
    total_num = sum([1 for i in open(data_file, "r",encoding='ISO-8859-1')])
    with open(data_file,'r',encoding='ISO-8859-1') as f:
        for idx,line in tqdm(enumerate(f),total=total_num):
            ele_list = line.strip().split('\t')
            if(len(ele_list)<2):
                continue
            text = ele_list[1].replace("_", " ")
            text_list = list(text.strip().split())
            if(len(text_list)>200):
                text = " ".join(text_list[0:200])
            if tokenizer_type == "subword":
                tokenized_text = tokenizer.tokenize(text)
            elif tokenizer_type == "word":
                tokenized_text = [token.text.lower() for token in tokenizer(text)]
            for token in tokenized_text:
                vocab[token] +=1
        if tokenizer_type == "subword":
            for token in tokenizer.vocab:
                vocab[token]+=1
        with open(os.path.join(data_dir,'corpus_vocab'), 'w',encoding='ISO-8859-1') as f:
            for w,fre in vocab.items():
                f.write(w)
                f.write('\t\t')
                f.write(str(fre))
                f.write('\r')

    return vocab







