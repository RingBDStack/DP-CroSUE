import argparse
import torch
import random
import numpy as np
import logging
import os
logger = logging.getLogger(__name__)
from tqdm import tqdm
from scipy.special import softmax
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from San_Text.utils import word_normalize, get_vocab,get_vocab_weibo
from spacy.lang.en import English
from transformers import BertTokenizer, BertForMaskedLM
from San_Text.SanText import SanText_plus,SanText_plus_init
import en_core_web_lg
import zh_core_web_lg
import pandas as pd

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix




def make_santext(dataset, data_dir, data_file, epsilon, p, sensitive_word_percentage,outpath):
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=data_dir,
        type=str,
        help="The input dir"
    )

    parser.add_argument(
        "--data_file",
        default=data_file,
        type=str,
        help="The input dir"
    )

    parser.add_argument(
        "--bert_model_path",
        default="bert-base-uncased",
        type=str,
        help="bert model name or path. leave it bank if you are using Glove"
    )

    parser.add_argument(
        "--word_embedding_path",
        default='./data/glove.840B.300d.txt',
        type=str,
        help="The pretrained word embedding path. leave it blank if you are using BERT",
    )

    parser.add_argument(
        "--word_embedding_size",
        default=300,
        type=int,
        help="The pretrained word embedding size. leave it blank if you are using BERT",
    )

    parser.add_argument(
        '--method',
        choices=['SanText', 'SanText_plus'],
        default='SanText_plus',
        help='Sanitized method'
    )

    parser.add_argument(
        '--embedding_type',
        choices=['glove', 'bert', 'spacy'],
        default='spacy',
        help='embedding used for sanitization'
    )

    parser.add_argument('--task',
                        choices=['foursquare', 'twitter'],
                        default=dataset,
                        )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--epsilon", type=float, default=epsilon, help="privacy parameter epsilon")
    parser.add_argument("--p", type=float, default=p,
                        help="SanText+: probability of non-sensitive words to be sanitized")

    parser.add_argument("--sensitive_word_percentage", type=float, default=sensitive_word_percentage,
                        help="SanText+: how many words are treated as sensitive")

    parser.add_argument("--threads", type=int, default=4, help="number of processors")

    parser.add_argument("--outpath",type=str,default=outpath)
    args = parser.parse_args()

    set_seed(args)

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Running method: %s, task: %s,  epsilon = %s, random_seed: %d" % (
        args.method, args.task, args.epsilon, args.seed))

    logger.info("Building Vocabulary...")
    if args.embedding_type == "glove":
        tokenizer = English()
        tokenizer_type = "word"

    elif args.embedding_type == "spacy":
        if args.task == "twitter" or args.task == "foursquare":
            tokenizer = en_core_web_lg.load()
        elif args.task == "weibo1" or args.task == "weibo2":
            tokenizer = zh_core_web_lg.load()
        tokenizer_type = "word"

    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        tokenizer_type = "subword"

    if args.task == "foursquare":
        vocab = get_vocab(args.data_dir, args.data_file, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "twitter":
        vocab = get_vocab(args.data_dir, args.data_file, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "weibo1" or args.task=="weibo2":
        vocab = get_vocab_weibo(args.task,args.data_dir,args.data_file,tokenizer, tokenizer_type=tokenizer_type)
    else:
        raise NotImplementedError

    sensitive_word_count = int(args.sensitive_word_percentage * len(vocab))
    words = [key for key, _ in vocab.most_common()]
    sensitive_words = words[-sensitive_word_count - 1:]
    sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}
    logger.info("#Total Words: %d, #Sensitive Words: %d" % (len(words), len(sensitive_words2id)))

    sensitive_word_embed = []
    all_word_embed = []

    word2id = {}
    sword2id = {}
    sensitive_count = 0
    all_count = 0

    if args.embedding_type == "spacy":
        print("embedding_type:",args.embedding_type)
        for idx,word in enumerate(vocab.keys()):
            if word not in word2id and tokenizer.vocab.strings.__contains__(word):
                word2id[word] = all_count
                if all_count % 10000 == 0:
                    print(all_count)
                all_count += 1
                all_word_embed.append(tokenizer(word).vector)
                if word in sensitive_words2id:
                    sword2id[word] = sensitive_count
                    sensitive_count += 1
                    sensitive_word_embed.append(tokenizer(word).vector)
            assert len(word2id) == len(all_word_embed)
            assert len(sword2id) == len(sensitive_word_embed)
        print(vocab)


    elif args.embedding_type == "glove":
        num_lines = sum(1 for _ in open(args.word_embedding_path))
        logger.info("Loading Word Embedding File: %s" % args.word_embedding_path)

        with open(args.word_embedding_path) as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for row in tqdm(f, total=num_lines - 1):
                content = row.rstrip().split(' ')
                cur_word = word_normalize(content[0])
                if cur_word in vocab and cur_word not in word2id:
                    word2id[cur_word] = all_count
                    all_count += 1
                    emb = [float(i) for i in content[1:]]
                    all_word_embed.append(emb)
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)
                assert len(word2id) == len(all_word_embed)
                assert len(sword2id) == len(sensitive_word_embed)
            f.close()
    else:
        logger.info("Loading BERT Embedding File: %s" % args.bert_model_path)
        model = BertForMaskedLM.from_pretrained(args.bert_model_path)
        embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

        for cur_word in tokenizer.vocab:
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = all_count
                emb = embedding_matrix[tokenizer.convert_tokens_to_ids(cur_word)]
                all_word_embed.append(emb)
                all_count += 1

                if cur_word in sensitive_words2id:
                    sword2id[cur_word] = sensitive_count
                    sensitive_count += 1
                    sensitive_word_embed.append(emb)
            assert len(word2id) == len(all_word_embed)
            assert len(sword2id) == len(sensitive_word_embed)

    all_word_embed = np.array(all_word_embed, dtype='f')
    sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')

    logger.info("All Word Embedding Matrix: %s" % str(all_word_embed.shape))
    logger.info("Sensitive Word Embedding Matrix: %s" % str(sensitive_word_embed.shape))

    logger.info("Calculating Prob Matrix for Exponential Mechanism...")
    prob_matrix = cal_probability(all_word_embed, sensitive_word_embed, args.epsilon)
    threads = min(args.threads, cpu_count())

    if args.method == "SanText":
        args.sensitive_word_percentage = 1.0
        out_path = args.data_dir + '/' + "text_eps_%.2f" % args.epsilon


    if args.task=="foursquare" or args.task=="twitter":

        logger.info("Processing file: %s. Will write to: %s" % (args.data_file, args.outpath))
        with open(args.data_file, 'r', encoding='ISO-8859-1') as rf:
            users = []
            texts = []
            total_num = sum([1 for i in open(args.data_file, "r", encoding='ISO-8859-1')])
            print("get_vocab from:", args.data_file)
            if args.task == "foursquare" or args.task == "twitter":
                for idx, line in tqdm(enumerate(rf), total=total_num):
                    ele_list = line.strip().split('\t\t')
                    if (len(ele_list) < 2):
                        continue
                    text = ele_list[1]
                    user = str(ele_list[0])
                    if args.embedding_type == "glove" or args.embedding_type == "spacy":
                        text = [token.text.lower() for token in tokenizer(text)]
                    else:
                        text = tokenizer.tokenize(text)
                    texts.append(text)
                    users.append(user)
            rf.close()

    elif args.task == "weibo1" or args.task=="weibo2":
        texts = pd.DataFrame(pd.read_csv(args.data_file))["con"].values.tolist()
        users = pd.DataFrame(pd.read_csv(args.data_file))["user_id"].values.tolist()
        logger.info("Processing file: %s. Will write to: %s" % (args.data_file, args.outpath))

    with Pool(threads, initializer=SanText_plus_init,
              initargs=(prob_matrix, word2id, sword2id, words, args.p, tokenizer)) as p:
        annotate_ = partial(
            SanText_plus,
        )
        results = list(
            tqdm(
                p.imap(annotate_, texts, chunksize=32),
                total=len(texts),
                desc="Sanitize docs using SanText",
            )
        )
        p.close()

    logger.info("Saving ...")

    print(results)
    if args.task == "foursquare" or args.task == "twitter":
        with open(args.outpath, 'w') as out_file:
            for i, predicted_text in enumerate(results):
                write_content = str(users[i]) + "\t\t"  + predicted_text +"\n"
                out_file.write(write_content)

    elif args.task == "weibo1" or args.task == "weibo2":
        df = pd.DataFrame()
        df["user_id"] = users
        df["con"] = results
        df.to_csv(args.outpath)

