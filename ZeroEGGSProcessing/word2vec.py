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
import os
from praatio import textgrid
import fasttext


def parse_args():
    parser = argparse.ArgumentParser(description='OHGesture')
    parser.add_argument('--src', type=str, required=True, default='./data/train')
    parser.add_argument('--dest', type=str, required=True, default='./processed/train')
    parser.add_argument('--word2vec_model', required=True, type=str, default="./word_embedding/crawl-300d-2M.vec")
    parser.add_argument('--subword_model', type=str, default="./word_embedding/crawl-300d-2M-subword")

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


def text_grid2tsv(file):
    # Read the TextGrid file using praatio
    tg = textgrid.openTextgrid(file, False)

    # Get the first tier (assuming that's what we want based on original code)
    tier = tg.getTier(tg.tierNames[0])

    # Create output filename by replacing .TextGrid with .tsv
    output_file = file.replace('.TextGrid', '.tsv')

    # Write entries to TSV
    with open(output_file, 'w', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        # Iterate through entries in the tier
        for entry in tier.entries:
            start_time, end_time, label = entry

            # Skip empty labels (similar to original)
            if label == '':
                continue

            tsv_writer.writerow([start_time, end_time, label])


def load_tsv_aligned(file):
    sentence = []
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            if len(line) == 3:
                start, end, raw_word = line
                start = float(start)
                end = float(end)
                sentence.append([start, end, raw_word])

    return sentence


def load_csv_aligned(file):
    sentence = []
    with open(os.path.join(file), "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if len(line) == 5:
                start, end, raw_word, type, speaker = line

                if type == "words":
                    start = float(start)
                    end = float(end)
                    sentence.append([start, end, raw_word])

    return sentence


def word2vec(sentence, word2vec_model, n_frames, fps, csv_file):
    tensor_vec = np.zeros([n_frames, 300])

    for start, end, word in sentence:
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        try:
            # Get the vector for the word
            vector = word2vec_model.get_vector(word.lower())

        except KeyError:
            with open("./log.txt", "a") as f:
                f.write(f"Inside file {csv_file}, Token {word} not found in word2vec model\n")
            print(f"Token {word} not found in word2vec model")
            vector = np.zeros(300)

            # Handle missing words
            # print(f"Keyword '{word}' not found in {tsv_file}")
            # try:
            #     # Find the most similar word in the vocabulary
            #     similar_word = word2vec_model.most_similar(normalized_word, topn=1)[0][0]
            #     print(f"Using similar word '{similar_word}' for '{word}'")
            #     vector = word2vec_model.get_vector(similar_word)
            # except KeyError:
            #     # If no similar word is found, use a zero vector
            #     print(f"No similar word found for '{word}'. Using a zero vector.")
            #     vector = np.zeros(300)

        tensor_vec[start_frame:end_frame, :] = vector

    print("embedding shape", np.shape(tensor_vec))
    return tensor_vec


if __name__ == '__main__':
    '''
    python word2vec.py --src=./data_full/train/aligned  --dest=./processed/train_full/embedding --word2vec_model=./word_embedding/crawl-300d-2M.vec --subword_model=./word_embedding/crawl-300d-2M-subword
    '''
    args = parse_args()
    pprint(args)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    files = os.listdir(args.src)

    wav_files = [file for file in files if file.endswith(".wav")]
    csv_files = [file for file in files if file.endswith(".csv")]
    fps = 20

    # assert len(wav_files) == len(csv_files)

    word2vec_model = KeyedVectors.load_word2vec_format(args.word2vec_model, binary=False)
    # subword_model = fasttext.load_model(args.subword_model)

    # if len(tsv_files) <= 0:
    #     for i, tgd_file in enumerate(tgd_files):
    #         file = os.path.join(args.src, tgd_file)
    #         text_grid2tsv(file)

    for i, wav_file in enumerate(wav_files):
        wav_file_path = os.path.join(args.src, wav_file)
        wav, sr = librosa.load(wav_file_path, sr=16000)
        audio_length_seconds = len(wav) / sr
        print(wav.shape)

        # tsv_file = os.path.join(args.src, f"{wav_file[:-4]}.tsv")
        # align_sentence = load_tsv_aligned(tsv_file)

        csv_file = os.path.join(args.src, f"{wav_file[:-4]}.csv")
        align_sentence = load_csv_aligned(csv_file)

        n_frames = int(audio_length_seconds * fps)
        print(f"Audio length: {audio_length_seconds} -> {audio_length_seconds * fps}")
        sentence_vec = word2vec(align_sentence, word2vec_model, n_frames, fps, csv_file)
        print(np.shape(sentence_vec), " -> saving ", csv_file)
        np.save(os.path.join(args.dest, f"{wav_file[:-4]}.npy"), sentence_vec)
        # # np.savez_compressed
