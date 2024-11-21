import yaml
from pprint import pprint
from easydict import EasyDict
import argparse
import io
import tqdm
import numpy as np
import librosa
from gensim.models import KeyedVectors
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='OHGesture')
    # parser.add_argument('--config', default='./configs/OHGesture.yml')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--word2vec_model', type=str, default="./fasttext/crawl-300d-2M.vec")

    args = parser.parse_args()
    return args

def load_word_embeddings(word2vec_model):
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model, binary=False)
    text = """I think the worst food that would be liquefied and drank through a straw."""
    print(text)
    texts = text.split(" ")
    print("Text length", len(texts))
    for word in texts:
        vector = word2vec_model[word]
        print(np.shape(vector))
        break


def load_csv_aligned(csv_aligned_file):
    sentence = []
    with open(csv_aligned_file, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) == 4:
                raw_word, word, start, end = line
                start = float(start)
                end = float(end)
                sentence.append([start, end, raw_word])

    return sentence


# def load_word2vectors(fname):  # take about 03:27
#     print("Loading word2vector ...")
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in tqdm(fin):
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
#     return data

def word2vec(sentence, word2vec_model, length):
    tensor_vec = np.zeros([length, 300])

    for start, end, word in sentence:
        vector = word2vec_model[word]
        tensor_vec[start:end, :] += vector

    return tensor_vec



if __name__ == '__main__':
    '''
    cd mydiffusion_zeggs/
    python word2vec.py --config=./configs/OHGesture.yml --gpu mps
    '''

    args = parse_args()
    # device = torch.device(args.gpu)

    # with open(args.config) as f:
    #     config = yaml.safe_load(f)
    #
    # for k, v in vars(args).items():
    #     config[k] = v
    # pprint(config)
    pprint(args)
    # word2vector = load_word2vectors(fname=args.word2vec_model)
    tsvpath = ""

    audiofile = "./data/train/001_Neutral_0_x_1_0.wav"

    wav, sr = librosa.load(audiofile, sr=16000)
    print(wav.shape)

    audio_length_seconds = len(wav) / sr

    print(f"Audio length: {audio_length_seconds} seconds")

    # load_word_embeddings(args.word2vec_model)
    word2vec_model = KeyedVectors.load_word2vec_format(args.word2vec_model, binary=False)
    align_sentence = load_csv_aligned('./data/train/001_Neutral_0_x_1_0.csv')
    word2vec(align_sentence, word2vec_model, audio_length_seconds)

    # for item in align_sentence:
    #     print(item)

    # tsv = load_tsv(tsvpath.replace('.TextGrid', '.tsv'), word2vector, clip_len)

    # config = EasyDict(config)



