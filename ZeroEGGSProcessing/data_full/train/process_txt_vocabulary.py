import os
import sys
import nltk
import argparse
from pprint import pprint
from gensim.models import KeyedVectors
import contractions
nltk.download('punkt')
nltk.download('punkt_tab')


def parse_args():
    parser = argparse.ArgumentParser(description='OHGesture')
    parser.add_argument('--src', type=str, default='./txt')
    parser.add_argument('--dest', type=str, default='./txt_processed')
    parser.add_argument('--word2vec_model', type=str, default="./fasttext/crawl-300d-2M.vec")

    args = parser.parse_args()
    return args

# def expand_contractions_nltk(text, word2vec_model, file):
#     # Expand contractions
#     text_fixed = contractions.fix(text)
#
#     tokens = nltk.word_tokenize(text_fixed)
#
#     # Rechecks
#     for token in tokens:
#         try:
#             word2vec_model.get_vector(token)
#         except KeyError:
#             with open("./data/train/log.txt", "a") as f:
#                 f.write(f"Inside file {file}, Token {token} not found in word2vec model\n")
#             print(f"Token {token} not found in word2vec model")
#
#     # Rejoin tokens into a sentence
#     expanded_text = ' '.join(text_fixed)
#     return expanded_text
#
#
# def process_txt_contractions(file, word2vec_model):
#     raw_text = ""
#     with open(file, "r") as f:
#         raw_text = f.readlines()
#
#     sentence_result = expand_contractions_nltk(raw_text, word2vec_model, file)
#     return sentence_result

def process_txt_contractions(file, word2vec_model):
    with open(file, "r") as f:
        raw_text = f.readlines()

    # Expand contractions
    text_fixed = contractions.fix(''.join(raw_text))  # Join lines into a single string for contraction expansion

    tokens = nltk.word_tokenize(text_fixed)

    # Recheck tokens in word2vec model
    for token in tokens:
        try:
            word2vec_model.get_vector(token)
        except KeyError:
            with open("./data/train/log.txt", "a") as log_file:
                log_file.write(f"Inside file {file}, Token {token} not found in word2vec model\n")
            print(f"Token {token} not found in word2vec model")

    # Rejoin tokens into a sentence
    expanded_text = ' '.join(tokens)  # Fix this to join the tokens, not the original text
    return expanded_text


if __name__ == '__main__':
    """
    python process_txt_contractions.py --src="./data_full/train/txt" --dest="./data_full/train/corpus" --word2vec_model=./fasttext/crawl-300d-2M.vec
    """
    # Example usage
    # text = "I'll go to the market, and we'll see what happens."
    args = parse_args()
    pprint(args)

    transcribes = os.listdir(args.src)
    files = [os.path.join(args.src, f) for f in transcribes]
    word2vec_model = KeyedVectors.load_word2vec_format(args.word2vec_model, binary=False)

    for file in files:
        print("Processing", file)
        txt_processed = process_txt_contractions(file, word2vec_model)

        name = file.split("/")[-1][:-4]
        lab_file = os.path.join(args.dest, f"{name}.lab")
        with open(lab_file, "w") as f:
            f.write(txt_processed)
            print(f"Saved {lab_file}")
