from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from datetime import datetime


import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, Split
from tokenizers import pre_tokenizers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

MAX_LENGTH = 100

class Voc:
    def __init__(self, name, tokenizer_type, vocab_size=None):
        self.name = name
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size

        if self.tokenizer_type == "custom":
            self.word2index = {"SOS": 0, "EOS": 1, "PAD": 2, "UNK": 3}
            self.word2count = {}
            self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3: "UNK"}
            self.n_words = 4  # Count SOS, EOS, PAD and UNK
        else:
            if self.vocab_size is None:
                raise ValueError("vocab_size must be provided for BPE or Unigram tokenizers")
            self.tokenizer = Tokenizer(BPE() if self.tokenizer_type == "bpe" else Unigram())
            # Custom pre-tokenizer to preserve numbers and alphanumeric codes
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                Split(r"(\d+|[a-zA-Z]\d+|[a-zA-Z]|[^\w\s])", behavior="isolated"),
                Whitespace()
            ])
            self.trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[SOS]", "[EOS]", "[PAD]", "[UNK]"],
                unk_token="[UNK]", 
                show_progress=True
            ) if self.tokenizer_type == "bpe" else UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[SOS]", "[EOS]", "[PAD]", "[UNK]"],
                unk_token="[UNK]", 
                show_progress=True
            )

    def addSentence(self, sentence):
        if self.tokenizer_type == "custom":
            for word in sentence.split(' '):
                self.addWord(word)

    def addWord(self, word):
        if self.tokenizer_type == "custom":
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1
    
    def train(self, corpus):
        if self.tokenizer_type != "custom":
            self.tokenizer.train_from_iterator(corpus, trainer=self.trainer)
            self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, sentence):
        if self.tokenizer_type == "custom":
            return [self.word2index.get(word, UNK_token) for word in sentence.split(' ')]
        else:
            return self.tokenizer.encode(sentence).ids

    def decode(self, ids):
        if self.tokenizer_type == "custom":
            return ' '.join([self.index2word.get(id, "UNK") for id in ids])
        else:
            return self.tokenizer.decode(ids)

    def wordToIndex(self, word):
        if self.tokenizer_type == "custom":
            return self.word2index.get(word, UNK_token)
        else:
            return self.tokenizer.token_to_id(word)

    def ensure_word(self, word):
        """Ensures a word is in the word2index and index2word"""
        if self.tokenizer_type == "custom":
            if word not in self.word2index:
                self.addWord(word)

def readVocs(corpus, corpus_name):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus) as f:
        content = f.readlines()
    lines = [x.strip() for x in content]
    it = iter(lines)
    # pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
    pairs = [[x, next(it)] for x in it]
    voc = Voc(corpus_name)
    return voc, pairs


class EncoderVoc(Voc):
    def __init__(self, name, tokenizer_type, vocab_size=None):
        super().__init__(name, tokenizer_type, vocab_size)
    
    def encode(self, sentence):
        if self.tokenizer_type == "custom":
            return [self.word2index.get(word, UNK_token) for word in sentence.split(' ')] + [EOS_token]
        else:
            return self.tokenizer.encode(sentence).ids  # EOS is handled by the tokenizer's

class DecoderVoc(Voc):
    def __init__(self, name):
        super().__init__(name, tokenizer_type="custom")

def prepareData(corpus, corpus_name, save_dir, tokenizer_type, vocab_size=None):
    print("Reading lines...")
    with open(corpus, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    pairs = [[lines[i].strip(), lines[i+1].strip()] for i in range(0, len(lines), 2)]
    print("Read {!s} sentence pairs".format(len(pairs)))
    
    encoder_voc = EncoderVoc(corpus_name + "_encoder", tokenizer_type=tokenizer_type, vocab_size=vocab_size)
    decoder_voc = DecoderVoc(corpus_name + "_decoder")
    
    if tokenizer_type == "custom":
        for pair in pairs:
            encoder_voc.addSentence(pair[0])
        print("Counted words for encoder:", encoder_voc.n_words)
    else:
        encoder_voc.train([pair[0] for pair in pairs])
    
    for pair in pairs:
        decoder_voc.addSentence(pair[1])
    print("Counted words for decoder:", decoder_voc.n_words)
    
    if tokenizer_type == "custom":
        print("Final encoder vocabulary size (unlimited):", encoder_voc.n_words)
    else:
        print(f"Final encoder vocabulary size (limited to {vocab_size}):", encoder_voc.vocab_size)
    print("Final decoder vocabulary size (unlimited):", decoder_voc.n_words)
    
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    torch.save(encoder_voc, os.path.join(directory, f'{tokenizer_type}_encoder_voc.tar'))
    torch.save(decoder_voc, os.path.join(directory, 'custom_decoder_voc.tar'))
    torch.save(pairs, os.path.join(directory, f'{tokenizer_type}_pairs.tar'))
    return encoder_voc, decoder_voc, pairs

def loadPrepareData(corpus, save_dir, tokenizer_type="custom", vocab_size=None, file_name=None):
    corpus_name = corpus.split('\\')[-1].split('.')[0]
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    print(f"Save directory: {save_dir}")
    print(f"Directory: {directory}")
    print(f"Corpus name: {corpus_name}")

    try:
        if file_name is None:
            encoder_voc_path = os.path.join(directory, f'{tokenizer_type}_encoder_voc.tar')
            decoder_voc_path = os.path.join(directory, 'custom_decoder_voc.tar')
            pairs_path = os.path.join(directory, f'{tokenizer_type}_pairs.tar')
        else:
            encoder_voc_path = os.path.join(directory, f'{tokenizer_type}_encoder_voc_{file_name}.tar')
            decoder_voc_path = os.path.join(directory, f'custom_decoder_voc_{file_name}.tar')
            pairs_path = os.path.join(directory, f'{tokenizer_type}_pairs_{file_name}.tar')

        print(f"Encoder Voc path: {encoder_voc_path}")
        print(f"Decoder Voc path: {decoder_voc_path}")
        print(f"Pairs path: {pairs_path}")

        if os.path.exists(encoder_voc_path) and os.path.exists(decoder_voc_path) and os.path.exists(pairs_path):
            print(f"{tokenizer_type.capitalize()} Pairs and Vocs exist --- Start loading training data ...")
            encoder_voc = torch.load(encoder_voc_path)
            decoder_voc = torch.load(decoder_voc_path)
            pairs = torch.load(pairs_path)
            return encoder_voc, decoder_voc, pairs
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        pass

    print("Saved data not found, start preparing training data ...")
    encoder_voc, decoder_voc, pairs = prepareData(corpus, corpus_name, save_dir, tokenizer_type=tokenizer_type, vocab_size=vocab_size)
    return encoder_voc, decoder_voc, pairs



def test_tokenizer(encoder_voc, decoder_voc, pairs, tokenizer_type):
    print("Testing tokenizer encoding and decoding:")

    sample_pair = random.choice(pairs)
    input_sentence, target_sentence = sample_pair[0], sample_pair[1]

    print("Original input sentence:", input_sentence)
    print("Original target sentence:", target_sentence)

    input_encoded = encoder_voc.encode(input_sentence)
    target_encoded = decoder_voc.encode(target_sentence)

    print("Encoded input:", input_encoded)
    print("Encoded target:", target_encoded)

    input_decoded = encoder_voc.decode(input_encoded)
    target_decoded = decoder_voc.decode(target_encoded)

    print("Decoded input:", input_decoded)
    print("Decoded target:", target_decoded)

    input_preserved = input_sentence == input_decoded
    target_preserved = target_sentence == target_decoded
    print("Input sentence preserved:", input_preserved)
    print("Target sentence preserved:", target_preserved)



### PREPROCESSING

import itertools

def indexesFromSentence(voc, sentence):
    if voc.tokenizer_type == "custom":
        return [voc.word2index.get(word, UNK_token) for word in sentence.split(' ')] + [EOS_token]
    else:
        return voc.encode(sentence)  # EOS is handled by the tokenizer's post-processor

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(voc, l, reverse):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(voc, l):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(encoder_voc, decoder_voc, pair_batch, reverse):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(encoder_voc, input_batch, reverse)
    output, mask, max_target_len = outputVar(decoder_voc, output_batch)
    return inp, lengths, output, mask, max_target_len


### MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # GRU with dropout
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        attn_energies = self.score(hidden, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        if self.method == 'dot':
            energy = torch.bmm(hidden.transpose(0, 1), encoder_outputs.transpose(0, 1).transpose(1, 2))
        elif self.method == 'general':
            energy = self.attn(encoder_outputs)
            energy = torch.bmm(hidden.transpose(0, 1), energy.transpose(0, 1).transpose(1, 2))
        elif self.method == 'concat':
            hidden_expanded = hidden.expand(-1, encoder_outputs.size(0), -1).transpose(0, 1)
            energy = self.attn(torch.cat((hidden_expanded, encoder_outputs), 2)).transpose(0, 1)
            energy = self.v.squeeze(0).unsqueeze(0).bmm(energy.transpose(1, 2))
        return energy.squeeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Add dropout layer for RNN output
        self.rnn_output_dropout = nn.Dropout(dropout)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        # Apply dropout to RNN output
        rnn_output = self.rnn_output_dropout(rnn_output)
        
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        
        # Return output and final hidden state
        return output, hidden, attn_weights
    

### Search Mechanism

import torch
import torch.nn.functional as F

def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=50):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]]).to(device)

            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, sentence.decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

def decode(decoder, decoder_hidden, encoder_outputs, voc, max_length=50, temperature=1.0, top_k=5):
    decoder_input = torch.LongTensor([[SOS_token]]).to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, encoder_outputs.size(0))

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        
        # Apply temperature
        decoder_output = decoder_output.div(temperature)
        
        # Top-k sampling
        top_k_logits, top_k_indices = decoder_output.topk(top_k)
        
        # Apply softmax to convert to probabilities
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Sample from the top-k distribution
        sampled_index = torch.multinomial(top_k_probs, 1)
        
        # Get the actual token index
        token_index = top_k_indices[0][sampled_index[0]]
        
        if token_index.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[token_index.item()])

        decoder_attentions[di] = decoder_attn.squeeze(0).squeeze(0).cpu().data

        decoder_input = token_index.unsqueeze(0)

    return decoded_words, decoder_attentions[:di + 1]


def decode_with_length_constraint(decoder, decoder_hidden, encoder_outputs, voc, output_length=7, temperature=1.0, top_k=5):
    decoder_input = torch.LongTensor([[SOS_token]]).to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(output_length, encoder_outputs.size(0))

    for di in range(output_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        
        # Apply temperature
        decoder_output = decoder_output.div(temperature)
        
        # Top-k sampling
        top_k_logits, top_k_indices = decoder_output.topk(top_k)
        
        # Apply softmax to convert to probabilities
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # If we haven't reached the output length - 1, exclude EOS token
        if di < output_length - 1:
            eos_indices = (top_k_indices == EOS_token).nonzero(as_tuple=True)
            if eos_indices[0].size(0) > 0:
                top_k_probs[0, eos_indices[1]] = 0
                if top_k_probs.sum() > 0:
                    top_k_probs = top_k_probs / top_k_probs.sum()  # Renormalize

        # Sample from the top-k distribution
        sampled_index = torch.multinomial(top_k_probs, 1)
        
        # Get the actual token index
        token_index = top_k_indices[0][sampled_index[0]]
        
        if token_index.item() == EOS_token and di == output_length - 1:
            decoded_words.append('<EOS>')
            break
        elif token_index.item() == EOS_token:
            continue
        else:
            decoded_words.append(voc.index2word[token_index.item()])

        decoder_attentions[di] = decoder_attn.squeeze(0).squeeze(0).cpu().data

        decoder_input = token_index.unsqueeze(0)

    # If we haven't reached the output length, pad with PAD
    while len(decoded_words) < output_length:
        decoded_words.append(voc.index2word[PAD_token])

    return decoded_words, decoder_attentions[:len(decoded_words)]



class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())
    

### EVALUATION

import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_and_visualize_attention(encoder, decoder, encoder_voc, decoder_voc, sentence, beam_size=1, max_length=MAX_LENGTH, temperature=1.0, top_k=5):
    print(f"Input: {sentence}")

    # Tokenize the input sentence based on the tokenizer type

    # Encode the input sentence using the encoder vocabulary
    indexes_batch = [encoder_voc.encode(sentence)]
    
    # Decode the encoded sentence back to words
    input_words = encoder_voc.decode(indexes_batch[0])
    
    # If the decoded result is a string, split it into a list of words
    # This is necessary because some tokenizers might return a string instead of a list
    if isinstance(input_words, str):
        input_words = input_words.split()
    
    # Note: At this point, input_words contains the tokenized input sentence
    # which will be used later for attention visualization
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1).to(device)

    # Perform the evaluation
    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    if beam_size == 1:
        decoded_words, attentions = decode(decoder, decoder_hidden, encoder_outputs, decoder_voc, max_length, temperature, top_k)
        output_sentence = ' '.join(decoded_words)
        print(f"Output: {output_sentence}")

        # Visualize attention
        fig, ax = plt.subplots(figsize=(10, 10))
        
        attention = attentions.squeeze(1).cpu().detach().numpy()
        
        sns.heatmap(attention, xticklabels=input_words, yticklabels=decoded_words, ax=ax, cmap='viridis')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.title("Attention Visualization")
        plt.tight_layout()
        plt.show()

        return output_sentence, attention

    else:
        beam_results = beam_decode(decoder, decoder_hidden, encoder_outputs, decoder_voc, beam_size, max_length)
        
        # Print top beam_size results
        for i, (decoded_words, score) in enumerate(beam_results[:beam_size]):
            output_sentence = ' '.join(decoded_words)
            print(f"Beam {i+1} (score: {score:.4f}): {output_sentence}")

        # We can't visualize attention for beam search as easily, so we'll skip that part
        return beam_results, None

# Example usage:
# result, attention = evaluate_and_visualize_attention(encoder, decoder, voc, "2 4370281 4 1806093 2 1977341 n82", beam_size=1)
# print(f"Translated sentence: {result}")

# For beam search:
# results, _ = evaluate_and_visualize_attention(encoder, decoder, voc, "2 4370281 4 1806093 2 1977341 n82", beam_size=3)
# for i, (sentence, score) in enumerate(results):
#     print(f"Beam {i+1} (score: {score:.4f}): {' '.join(sentence)}")


# Example usage:
# output_greedy = evaluate(encoder, decoder, encoder_voc, decoder_voc, "2 4370281 4 1806093 2 1977341 n82")
# print(f"Greedy decoding: {output_greedy}")

# output_beam = evaluate(encoder, decoder, encoder_voc, decoder_voc, "2 4370281 4 1806093 2 1977341 n82", beam_size=3)
# print(f"Beam search (beam_size=3): {output_beam}")

def evaluate_inlength_and_visualize_attention(encoder, decoder, encoder_voc, decoder_voc, sentence, output_length=7):
    print(f"Input: {sentence}")

    # Tokenize the input sentence based on the tokenizer type
    if encoder_voc.tokenizer_type == "custom":
        indexes_batch = [indexesFromSentence(encoder_voc, sentence)]
        input_words = sentence.split() + ['<EOS>']
    else:
        indexes_batch = [encoder_voc.encode(sentence)]
        input_words = encoder_voc.decode(indexes_batch[0]) + ['<EOS>']
    
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1).to(device)

    # Perform the evaluation
    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    decoded_words, attentions = decode_with_length_constraint(decoder, decoder_hidden, encoder_outputs, decoder_voc, output_length)
    output_sentence = ' '.join(decoded_words)
    print(f"Output: {output_sentence}")

    # Visualize attention
    fig, ax = plt.subplots(figsize=(10, 10))
    
    attention = attentions.cpu().detach().numpy()
    
    sns.heatmap(attention, xticklabels=input_words, yticklabels=decoded_words, ax=ax, cmap='viridis')
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.title("Attention Visualization")
    plt.tight_layout()
    plt.show()

    return output_sentence, attention


def evaluateinlength(encoder, decoder, encoder_voc, decoder_voc, sentence, output_length=7, temperature=1.0, top_k=5):
    # Tokenize the input sentence based on the tokenizer type
    if encoder_voc.tokenizer_type == "custom":
        indexes_batch = [indexesFromSentence(encoder_voc, sentence)]
    else:
        indexes_batch = [encoder_voc.encode(sentence)]
    
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1).to(device)

    # Perform the evaluation
    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    decoded_words, _ = decode_with_length_constraint(decoder, decoder_hidden, encoder_outputs, decoder_voc, output_length, temperature, top_k)
    output_sentence = ' '.join(decoded_words)
    return output_sentence

### External Context

import pandas as pd
import copy

class ExternalContext:
   def __init__(self, app_df, nodes_df, cloud_node_id):
       self.app_df = app_df
       self.nodes_df = nodes_df
       self.cloud_node_id = cloud_node_id
       self.cloud_allocations = set()  # Track modules allocated to cloud
       self.allocations = {}
       self.allocation_history = []
       self._module_count_cache = {}
       self.current_failures = {}  # Track failures for current placement attempt
       self.failure_reasons = {}   # Track reasons for failures
       self.module_failures = {}   # Track which modules failed on which nodes
       self.max_retries = 5
       self.prepare_data()
   def prepare_data(self):
       self.app_module_storage = self.app_df.set_index(['Application ID', 'Module ID'])['Storage'].to_dict()
       self.original_node_storage = self.nodes_df.set_index('id')['Storage'].to_dict()
       # Ensure cloud node has effectively infinite resources
       self.original_node_storage[self.cloud_node_id] = float('inf')
       self.remaining_node_storage = copy.deepcopy(self.original_node_storage)
   def record_node_failure(self, app_id, module_id, node_id, reason="Unknown"):
       """Record a node failure with reason for the current placement attempt"""
       if app_id not in self.current_failures:
           self.current_failures[app_id] = []
           self.failure_reasons[app_id] = []
           self.module_failures[app_id] = {}
       
       failure_entry = f"f{node_id}"
       self.current_failures[app_id].append(failure_entry)
       
       # Track which module failed on this node
       if module_id not in self.module_failures[app_id]:
           self.module_failures[app_id][module_id] = []
       self.module_failures[app_id][module_id].append(node_id)
       
       # Include module ID in the failure reason
       self.failure_reasons[app_id].append(f"{failure_entry}:m{module_id}:{reason}")
       print(f"Recording failure for app {app_id}, module {module_id}, node {node_id}: {reason}")
   def get_failure_string(self, app_id):
       """Get the accumulated failure string with reasons"""
       if app_id in self.current_failures:
           failures = " ".join(self.current_failures[app_id])
           reasons = " | ".join(self.failure_reasons[app_id])
           return f"{failures} ({reasons})"
       return ""
   def get_module_failures(self, app_id):
       """Get a summary of which modules failed on which nodes"""
       if app_id in self.module_failures:
           return {f"m{module_id}": [f"n{node_id}" for node_id in failed_nodes] 
                  for module_id, failed_nodes in self.module_failures[app_id].items()}
       return {}
   def reset_failures(self, app_id=None):
       """Reset failure tracking"""
       if app_id is None:
           self.current_failures = {}
           self.failure_reasons = {}
           self.module_failures = {}
       else:
           self.current_failures[app_id] = []
           self.failure_reasons[app_id] = []
           self.module_failures[app_id] = {}
   def get_module_storage(self, app_id, module_id):
       return self.app_module_storage.get((app_id, module_id), 0)
   def get_node_storage(self, node_id):
       return self.remaining_node_storage.get(node_id, 0)
   def is_valid_allocation(self, app_id, module_id, node_id):
       if module_id in self.allocations.get(app_id, {}):
           return False
       module_storage = self.get_module_storage(app_id, module_id)
       node_storage = self.get_node_storage(node_id)
       return module_storage <= node_storage
   def allocate_resource(self, app_id, module_id, node_id):
       if not self.is_valid_allocation(app_id, module_id, node_id):
           return False
           
       module_storage = self.get_module_storage(app_id, module_id)
       # Don't subtract from cloud resources (they're infinite)
       if node_id != self.cloud_node_id:
           self.remaining_node_storage[node_id] -= module_storage
       
       if app_id not in self.allocations:
           self.allocations[app_id] = {}
       self.allocations[app_id][module_id] = node_id
       
       self.log_allocation(app_id, module_id, node_id, True)
       return True
   def reset_app_allocation(self, app_id):
       """Reset allocations for a specific app"""
       if app_id in self.allocations:
           # Restore resources for this app's allocations
           for module_id, node_id in self.allocations[app_id].items():
               if node_id != self.cloud_node_id:
                   module_storage = self.get_module_storage(app_id, module_id)
                   self.remaining_node_storage[node_id] += module_storage
           
           # Remove the app's allocations
           del self.allocations[app_id]
           
           # Remove from cloud allocations
           self.cloud_allocations = {(a, m) for a, m in self.cloud_allocations if a != app_id}
   def reset_resources(self):
       self.remaining_node_storage = copy.deepcopy(self.original_node_storage)
       self.allocations = {}
       self.allocation_history = []
       self.cloud_allocations = set()
       self.current_failures = {}
       self.failure_reasons = {}
       self.module_failures = {}
   def get_num_modules(self, app_id):
       # Use caching for better performance
       if app_id not in self._module_count_cache:
           self._module_count_cache[app_id] = self.app_df[
               self.app_df['Application ID'] == app_id
           ]['Module ID'].nunique()
       return self._module_count_cache[app_id]
   def log_allocation(self, app_id, module_id, node_id, success):
       """Enhanced allocation logging"""
       self.allocation_history.append({
           'app_id': app_id,
           'module_id': module_id,
           'node_id': node_id,
           'is_cloud': node_id == self.cloud_node_id,
           'success': success,
           'remaining_storage': self.remaining_node_storage.copy(),
           'timestamp': datetime.now().isoformat()
       })



def display_allocation_logs(external_context, detailed=False):
   """
   Display allocation history in a formatted way.
   
   Args:
       external_context: ExternalContext instance
       detailed: If True, shows more detailed information including Storage
   """
   if not external_context.allocation_history:
       print("No allocations logged.")
       return
    # Convert to DataFrame for easier manipulation
   df = pd.DataFrame(external_context.allocation_history)
   
   # Basic statistics
   print("\n=== Allocation Summary ===")
   print(f"Total allocations: {len(df)}")
   print(f"Cloud allocations: {df['is_cloud'].sum()}")
   print(f"Edge allocations: {len(df) - df['is_cloud'].sum()}")
   print(f"Unique applications: {df['app_id'].nunique()}")
   
   # Per-application summary
   print("\n=== Per-Application Summary ===")
   app_summary = df.groupby('app_id').agg({
       'module_id': 'count',
       'is_cloud': 'sum'
   }).rename(columns={
       'module_id': 'Total Modules',
       'is_cloud': 'Cloud Placements'
   })
   app_summary['Edge Placements'] = app_summary['Total Modules'] - app_summary['Cloud Placements']
   print(app_summary)
   if detailed:
       print("\n=== Detailed Allocation Log ===")
       for idx, alloc in enumerate(external_context.allocation_history, 1):
           print(f"\nAllocation #{idx}:")
           print(f"App ID: {alloc['app_id']}")
           print(f"Module ID: {alloc['module_id']}")
           print(f"Node ID: {alloc['node_id']} {'(Cloud)' if alloc['is_cloud'] else '(Edge)'}")
           print(f"Success: {alloc['success']}")
           print("Remaining Storage after allocation:")
           for node_id, storage in alloc['remaining_storage'].items():
               if node_id != external_context.cloud_node_id:  # Skip cloud node
                   print(f"  Node {node_id}: {storage}")

def plot_allocation_distribution(external_context):
   """
   Create visualization of allocation distribution.
   """
   df = pd.DataFrame(external_context.allocation_history)
   
   # Create figure with subplots
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
   
   # Plot 1: Cloud vs Edge distribution
   placement_counts = df['is_cloud'].value_counts()
   ax1.pie([placement_counts.get(False, 0), placement_counts.get(True, 0)], 
           labels=['Edge', 'Cloud'],
           autopct='%1.1f%%',
           colors=['lightblue', 'lightcoral'])
   ax1.set_title('Cloud vs Edge Distribution')
   
   # Plot 2: Node allocation distribution (excluding cloud)
   edge_allocations = df[~df['is_cloud']]['node_id'].value_counts()
   edge_allocations.plot(kind='bar', ax=ax2)
   ax2.set_title('Edge Node Allocation Distribution')
   ax2.set_xlabel('Node ID')
   ax2.set_ylabel('Number of Allocations')
   
   plt.tight_layout()
   plt.show()


def decode_with_context_check(decoder, decoder_hidden, encoder_outputs, voc, external_context, app_id, 
                         max_length=50, temperature=1.0, top_k=5):
    """Modified decode function to handle and record failures"""
    # Input validation
    if app_id is None or not isinstance(app_id, (int, str)):
        raise ValueError("Invalid app_id provided")
    
    # Initialize variables
    decoder_input = torch.LongTensor([[SOS_token]]).to(device)
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, encoder_outputs.size(0))
    current_module_id = 0
    
    try:
        # Get expected number of modules for this application
        expected_modules = external_context.get_num_modules(app_id)
        if expected_modules <= 0:
            raise ValueError(f"Invalid number of modules ({expected_modules}) for app_id {app_id}")
         # Main decoding loop
        for di in range(max_length):
            # Break if we've placed all modules
            if current_module_id >= expected_modules:
                decoded_words.append('EOS')
                break
                
            # Get decoder output
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            # Store attention weights
            decoder_attentions[di] = decoder_attn.squeeze(0).squeeze(0).cpu().data
            
            # Apply temperature scaling
            decoder_output = decoder_output.div(temperature)
            
            # Get top-k candidates
            top_k_logits, top_k_indices = decoder_output.topk(top_k)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # Try to find a valid allocation
            allocation_found = False
            
            # Try each candidate node
            for i in range(top_k):
                token_index = top_k_indices[0][i]
                word = voc.index2word[token_index.item()]
                
                # Skip EOS token if we haven't placed all modules
                if word == 'EOS' and current_module_id < expected_modules - 1:
                    continue
                    
                # Process node number
                if word.startswith('n') and word[1:].isdigit():
                    node_id = int(word[1:])
                    
                    # Try to allocate
                    if external_context.is_valid_allocation(app_id, current_module_id, node_id):
                        if external_context.allocate_resource(app_id, current_module_id, node_id):
                            decoded_words.append(word)
                            decoder_input = token_index.unsqueeze(0).unsqueeze(0)
                            current_module_id += 1
                            allocation_found = True
                            break
                        else:
                            # Record failed allocation with reason
                            external_context.record_node_failure(app_id, current_module_id, node_id, 
                                reason="Resource allocation failed")
                    else:
                        # Record invalid allocation with reason
                        external_context.record_node_failure(app_id, current_module_id, node_id, 
                            reason="Invalid allocation")
            
            # If no valid allocation found among candidates
            if not allocation_found:
                failure_info = external_context.get_failure_string(app_id)
                print(f"Failed to find valid allocation. Failures: {failure_info}")
                return decoded_words, decoder_attentions[:len(decoded_words)], False
        
        # Verify all modules were placed
        if current_module_id < expected_modules:
            print(f"Warning: Only placed {current_module_id}/{expected_modules} modules for app {app_id}")
            return decoded_words, decoder_attentions[:len(decoded_words)], False
        
        return decoded_words, decoder_attentions[:len(decoded_words)], True
        
    except Exception as e:
        print(f"Error during decoding for app {app_id}: {e}")
        return decoded_words, decoder_attentions[:len(decoded_words)], False

def evaluate_greedy_with_external_context(encoder, decoder, encoder_voc, decoder_voc, sentence, external_context, app_id, max_length=MAX_LENGTH, temperature=1.0, top_k=5):
   """
   Evaluate the model with external context constraints.
   Returns the output sentence if successful, None if failed.
   """
   # Convert sentence to tensor
   indexes_batch = [encoder_voc.encode(sentence)]
   input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
   input_batch = input_batch.to(device)
    # Calculate input lengths
   input_lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Set to evaluation mode
   encoder.eval()
   decoder.eval()
   
   with torch.no_grad():
       try:
           # Run through encoder
           encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths)
           
           # Create starting decoder input
           decoder_hidden = encoder_hidden[:decoder.n_layers]
           
           # Reshape decoder_hidden if necessary
           decoder_hidden = decoder_hidden.view(decoder.n_layers, 1, -1)
           
           # Perform decoding with context check
           decoded_words, _, success = decode_with_context_check(
               decoder, decoder_hidden, encoder_outputs, decoder_voc, 
               external_context, app_id, max_length, temperature, top_k
           )
           
           if not success:
               print(f"Failed to place all modules for app {app_id}")
               return None
               
           # Format the output
           output_sentence = ' '.join(decoded_words)
           
           # Remove EOS token from output if present
           if 'EOS' in output_sentence:
               output_sentence = output_sentence[:output_sentence.index('EOS')]
           
           return output_sentence.strip()
           
       except Exception as e:
           print(f"Evaluation error for app {app_id}: {e}")
           return None



### TRAINING

def linear_decay_teacher_forcing(start_ratio, end_ratio, current_iteration, total_iterations):
    return max(end_ratio, start_ratio - (start_ratio - end_ratio) * (current_iteration / total_iterations))

def performance_based_teacher_forcing(start_ratio, end_ratio, current_loss, best_loss, patience=5):
    if current_loss < best_loss:
        return max(end_ratio, start_ratio * 0.9)  # Decrease by 10% if performance improves
    else:
        return min(start_ratio, start_ratio * 1.1)  # Increase by 10% if performance doesn't improve
    
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(inp.device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_embedding, decoder_embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio, device):
    
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for RNN packing should always be on CPU
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(model_name, encoder_voc, decoder_voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_embedding, decoder_embedding, 
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, 
               save_every, clip, corpus_name, loadFilename, start_teacher_forcing_ratio, end_teacher_forcing_ratio, 
               device, hidden_size, reverse=False, tf_strategy='linear'):
    
    print_loss = 0
    best_loss = float('inf')
    patience_counter = 0
    teacher_forcing_ratio = start_teacher_forcing_ratio

    # Load batches for each iteration
    training_batches = [batch2TrainData(encoder_voc, decoder_voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = load_checkpoint(loadFilename, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, voc)

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Update teacher forcing ratio
        if tf_strategy == 'linear':
            teacher_forcing_ratio = linear_decay_teacher_forcing(start_teacher_forcing_ratio, 
                                                                 end_teacher_forcing_ratio, 
                                                                 iteration, n_iteration)
        elif tf_strategy == 'curriculum':
            teacher_forcing_ratio = performance_based_teacher_forcing(teacher_forcing_ratio, 
                                                                  end_teacher_forcing_ratio, 
                                                                  print_loss, best_loss)

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_embedding, decoder_embedding, 
                     encoder_optimizer, decoder_optimizer, 
                     batch_size, clip, teacher_forcing_ratio, device)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Teacher forcing ratio: {:.2f}"
                  .format(iteration, iteration / n_iteration * 100, print_loss_avg, teacher_forcing_ratio))
            
            if print_loss_avg < best_loss:
                best_loss = print_loss_avg
                patience_counter = 0
            else:
                patience_counter += 1

            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'encoder_voc_dict': encoder_voc.__dict__,
                'decoder_voc_dict': decoder_voc.__dict__,
                'encoder_embedding': encoder_embedding.state_dict(),
                'decoder_embedding': decoder_embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
            
def load_checkpoint(loadFilename, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, voc):
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        embedding.load_state_dict(embedding_sd)
        return checkpoint['iteration']
    else:
        return 1

### initialize model
def initialize_model(hidden_size, encoder_n_layers, decoder_n_layers, dropout, batch_size, attn_score, learning_rate, tokenizer_type, encoder_voc=None, decoder_voc=None, encoder_vocab_size=None, decoder_vocab_size=None):
    # Initialize word embeddings
    # Determine encoder vocab size
    if tokenizer_type == "custom":
        encoder_vocab_size = encoder_voc.n_words
    elif tokenizer_type in ["bpe", "unigram"]:
        encoder_vocab_size = encoder_voc.vocab_size
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    # Decoder always uses custom tokenizer
    decoder_vocab_size = decoder_voc.n_words

    # Print vocabulary sizes for debugging
    print(f"Encoder vocabulary size: {encoder_vocab_size}")
    print(f"Decoder vocabulary size: {decoder_vocab_size}")


    # Initialize embeddings
    encoder_embedding = nn.Embedding(encoder_vocab_size, hidden_size)
    decoder_embedding = nn.Embedding(decoder_vocab_size, hidden_size)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, encoder_embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_score, decoder_embedding, hidden_size, decoder_vocab_size, decoder_n_layers, dropout)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Initialize optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # Set dropout layers to train mode
    encoder.train()
    decoder.train()

    # Print model information
    print(f'attention score: {attn_score}')
    print(f'tokenization type: {tokenizer_type}')
    print(f"Encoder vocabulary size: {encoder_vocab_size}")
    print(f"Decoder vocabulary size: {decoder_vocab_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Encoder layers: {encoder_n_layers}")
    print(f"Decoder layers: {decoder_n_layers}")
    print(f"Dropout: {dropout}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))}")
    print(f"Device: {device}")

    return encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_embedding, decoder_embedding


def load_model(loadFilename, encoder, decoder, encoder_voc, decoder_voc):
    # Load the saved data
    checkpoint = torch.load(loadFilename)
    
    # Load the model parameters
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    
    # Load the vocabularies
    encoder_voc.__dict__ = checkpoint['encoder_voc_dict']
    decoder_voc.__dict__ = checkpoint['decoder_voc_dict']
    
    return encoder, decoder, encoder_voc, decoder_voc

## JSON to DataFrame

import pandas as pd
import json
import networkx as nx

def json_to_dataframe(json_file_path):
    # Convert application JSON to DataFrame
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    rows = []
    for app in data:
        app_id = app['id']
        numberofmodule = app['numberofmodule']
        for module, message in zip(app['module'], app['message']):
            row = {
                'Application ID': app_id,
                'numberofmodule': numberofmodule,
                'Module ID': module['id'],
                'Module Name': module['name'],
                'Storage': module['Storage'],
                'Message ID': message['id'],
                'Message Name': message['name'],
                'Bytes': message['bytes'],
                'Instructions': message['instructions']
            }
            rows.append(row)
    return pd.DataFrame(rows)

def extract_nodes_to_dataframe(filepath):
    # Convert topology JSON to DataFrame
    with open(filepath, 'r') as file:
        data = json.load(file)
    nodes_data = data['nodes']
    return pd.DataFrame(nodes_data)


## PLACE USING SEQ2SEQ

import pandas as pd
import os

def place_seq2seq_externalcontext(csv_file_path, encoder, decoder, encoder_voc, decoder_voc, external_context, output_path=None):
    """
    Place applications using seq2seq model with external context and retry logic.
    Falls back to cloud placement after max retries.
    """
    df = pd.read_csv(csv_file_path)
    results = []
    
    # Reset external context before starting
    external_context.reset_resources()
    
    for index, row in df.iterrows():
        app_id = row['app_id']
        num_module = row['number_of_modules']
        base_input = row['modified_input']
        
        retry_count = 0
        current_placement = None
        placement_history = []
        last_attempted_placement = None
        used_cloud_fallback = False  # Initialize flag
        
        while retry_count < external_context.max_retries:
            failure_info = external_context.get_failure_string(app_id)
            current_input = f"{base_input} {failure_info}".strip()
            
            print(f"\nAttempt {retry_count + 1} for app {app_id}")
            print(f"Input with failures: {current_input}")
            
            translated = evaluate_greedy_with_external_context(
                encoder, decoder, encoder_voc, decoder_voc,
                current_input, app_id=app_id,
                external_context=external_context
            )
            
            if translated is not None:
                last_attempted_placement = translated
            
            placement_history.append({
                'attempt': retry_count + 1,
                'input': current_input,
                'output': translated,
                'failures': failure_info,
                'failure_details': external_context.failure_reasons.get(app_id, []),
                'module_failures': external_context.get_module_failures(app_id),
                'cloud_fallback': False,
                'original_placement': last_attempted_placement,
                'modules_to_cloud': []  # Empty list for non-cloud attempts
            })
            
            if translated is not None and translated != "FAILED":
                current_placement = translated
                used_cloud_fallback = False
                break
                
            retry_count += 1
            external_context.reset_app_allocation(app_id)
        
        # Cloud fallback logic
        if current_placement is None and last_attempted_placement is not None:
            nodes = last_attempted_placement.split()
            cloud_placement = []
            failed_modules = []  # Reset failed_modules for cloud fallback
            
            for i, node in enumerate(nodes):
                if node.startswith('n'):
                    if any(f"f{node[1:]}" in failure for failure in external_context.failure_reasons.get(app_id, [])):
                        cloud_placement.append(f"n{external_context.cloud_node_id}")
                        failed_modules.append(i)
                    else:
                        cloud_placement.append(node)
                else:
                    cloud_placement.append(node)
            
            current_placement = " ".join(cloud_placement)
            used_cloud_fallback = True
            
            placement_history.append({
                'attempt': retry_count + 1,
                'input': current_input,
                'output': current_placement,
                'failures': failure_info,
                'failure_details': external_context.failure_reasons.get(app_id, []),
                'module_failures': external_context.get_module_failures(app_id),
                'cloud_fallback': True,
                'original_placement': last_attempted_placement,
                'modules_to_cloud': failed_modules
            })
        
        results.append({
            'app_id': app_id,
            'number_of_modules': num_module,
            'final_input': current_input,
            'final_output': current_placement if current_placement else "FAILED",
            'attempts': retry_count + 1,
            'placement_history': placement_history,
            'used_cloud_fallback': used_cloud_fallback,
            'success_without_cloud': current_placement is not None and not used_cloud_fallback,
            'modules_in_cloud': failed_modules if used_cloud_fallback else []
        })
        
        external_context.reset_failures(app_id)
    
    # Create summary DataFrame
    results_df = pd.DataFrame([{
        'app_id': r['app_id'],
        'number_of_modules': r['number_of_modules'],
        'final_input': r['final_input'],
        'final_output': r['final_output'],
        'attempts': r['attempts'],
        'placement_success': r['final_output'] != "FAILED",
        'used_cloud_fallback': r['used_cloud_fallback'],
        'success_without_cloud': r['success_without_cloud'],
        'cloud_modules_count': len(r['modules_in_cloud']),
        'cloud_modules': r['modules_in_cloud']
    } for r in results])
    
    if output_path:
        with open(output_path.replace('.csv', '_detailed.json'), 'w') as f:
            json.dump(results, f, indent=2)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    return results_df, results


def evaluate_and_visualize_attentionV2(encoder, decoder, encoder_voc, decoder_voc, sentence, beam_size=1, max_length=MAX_LENGTH, temperature=1.0, top_k=5):
   print(f"Input: {sentence}")
    # Tokenize the input sentence based on the tokenizer type
   indexes_batch = [encoder_voc.encode(sentence)]
   
   # Decode the encoded sentence back to words
   input_words = encoder_voc.decode(indexes_batch[0])
   
   # If the decoded result is a string, split it into a list of words
   if isinstance(input_words, str):
       input_words = input_words.split()
   
   # Remove EOS from input_words if present
   if '<EOS>' in input_words:
       input_words.remove('<EOS>')
   
   lengths = [len(indexes) for indexes in indexes_batch]
   input_batch = torch.LongTensor(indexes_batch).transpose(0, 1).to(device)
    # Perform the evaluation
   encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)
   decoder_hidden = encoder_hidden[:decoder.n_layers]
   if beam_size == 1:
       decoded_words, attentions = decode(decoder, decoder_hidden, encoder_outputs, decoder_voc, max_length, temperature, top_k)
       
       # Remove EOS from decoded words if present
       if '<EOS>' in decoded_words:
           eos_idx = decoded_words.index('<EOS>')
           decoded_words = decoded_words[:eos_idx]
           attentions = attentions[:eos_idx]
           
       output_sentence = ' '.join(decoded_words)
       print(f"Output: {output_sentence}")
        # Visualize attention
       fig, ax = plt.subplots(figsize=(10, 10))
       
       attention = attentions.squeeze(1).cpu().detach().numpy()
       
       # Only use the actual words (no EOS) for visualization
       sns.heatmap(attention, xticklabels=input_words, yticklabels=decoded_words, ax=ax, cmap='viridis')
       
       ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
       ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
       
       plt.title("Attention Visualization")
       plt.tight_layout()
       plt.show()
       return output_sentence, attention