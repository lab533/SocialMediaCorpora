from __future__ import print_function
import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim import corpora
from collections import defaultdict
import sys




import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Embedding, Bidirectional, Merge
from keras.layers import LSTM, SimpleRNN, GRU



UNDEFINED = "_"
MAX_WORD_LENGTH = 20


def read_gikrya(path):
    """
    Reading format:
    row_index<TAB>form<TAB>lemma<TAB>POS<TAB>tag"""
    morpho_map = {"POS":{UNDEFINED: 0, 
                         0: UNDEFINED}}
    
    
    sentences = []
    vocab = {}    
    with open(path, 'r') as f:
        sentence = []
        for line in f:
            splits = line.strip().split('\t')            
            if len(splits) == 5:
                form, lemma, POS, tags = splits[1:]
                if POS not in  morpho_map["POS"]:
                    morpho_map["POS"][POS] = len(morpho_map["POS"]) // 2 
                    morpho_map["POS"][morpho_map["POS"][POS]] =  POS
                tags_list = [("POS", POS)]
                if tags != "_":
                    for tag_val in tags.split("|"):
                        tag, val = tag_val.split("=")
                        tags_list.append((tag, val))
                        if tag not in morpho_map:
                            morpho_map[tag] = {UNDEFINED: 0,
                                               0: UNDEFINED}
                        if val not in morpho_map[tag]:
                            morpho_map[tag][val] = len(morpho_map[tag]) // 2 
                            morpho_map[tag][morpho_map[tag][val]] = val
#                 else:
#                     tags_list.append(tags)
                if form not in vocab:
                    vocab[form] = form
                sentence.append((vocab[form], lemma, tags_list) )
                
                    
            elif len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
    return sentences, morpho_map       
 
    
def split_word(word, stemmer):
    flex = word[len(stemmer.stem(word)):]
    if len(flex):
        return word[:-len(flex)], flex
    return word, "empty"


def sentences_to_features(sentences, token2id, neighbors=3, undef_token="undefined_token"):
    arrays = [sentence_to_features(sent, token2id,  neighbors=neighbors,
                                  undef_token=undef_token) for sent in sentences]
    return np.vstack(arrays)


def sentence_to_features(sentence, token2id, 
                         neighbors=3, undef_token="undefined_token"):
    """
    Делает из предложения 
    матрицу id слов, где  строка соответствует словам предложения:
    в каждой строке состоит из neighbors id слов из левого контекста,
    потом id слова, затем neighbors id слов правого контекста
    0 - зарезерврован для паддинга, в словаре не должно быть слов с id 0
    """
    X = np.zeros(shape=(len(sentence), neighbors * 2 + 1), dtype=np.int)
    id_seq = np.zeros(shape=(len(sentence) + 2*neighbors,), dtype=np.int)
    for idx, token in enumerate(sentence):
        num = token2id.get(token, token2id[undef_token])
        assert num != 0
        id_seq[idx+neighbors] = num
    for idx in range(len(sentence)):
        X[idx, :] = id_seq[idx:idx + X.shape[1]]
    return X   
        
        
def build_vocab(sentences, min_freq=0, max_size=10000, undefined_id=1):
    """ 
    Строит словарь из слов встертившихся более min_freq раз,
    но размеров  не более max_size, в случае бОльшего количества токенов
    отбрасываются менее частотные токены, undefined_id - id первого токена в словаре,
    который будет называться "undefined_token"
    """
    offset = undefined_id
    token2id = {"undefined_token": offset}
    id2token = {offset: "undefined_token"}    
    
    counter = defaultdict(int)    
    for sentence in sentences:
        for token in sentence:
            counter[token] += 1
    sorted_tokens = [t_f[0]  for t_f in 
                     sorted([t_f for t_f in counter.items() if t_f[1] >= min_freq],
                           key=lambda tf: -tf[1])]                     
    
    for token in sorted_tokens[:max_size - len(token2id)]:
        offset += 1
        token2id[token] = offset
        id2token[offset] = token
    return token2id, id2token      
    
    
    
def simple_word2vec(word):
    pass


def build_morpho_vocab(morpho_map):
    morpho_сats = sorted([key for key in morpho_map.keys()])
    # чисто для удобства POS сделаем первым
    morpho_сats.insert(0, morpho_сats.pop(morpho_сats.index("POS"))) 
    abs_idx = 0
    tag2id = {}
    id2tag = {}
    for cat in morpho_сats:
        vals = [pair[0] for pair in sorted(list(morpho_map[cat].items()), 
                                           key=lambda p: p[1])]
        for val in vals:
            tag2id[(cat, val)] = abs_idx
            id2tag[abs_idx] = (cat, val)
            abs_idx += 1
    return tag2id, id2tag  


def tagsets_to_one_hot(tagsets, morpho_map, cat_order):    
    # при частых запусках не оптимально так:
    # cats = set([cat for cat, val in tag2id.keys()])
    y = [np.zeros(shape=(len(tagsets), len(morpho_map[cat]) // 2), dtype=np.int) 
         for cat in cat_order]
    
    for one_hot in y:
        one_hot[:, 0] = 1       
        
    for idx, tagset in enumerate(tagsets):                    
        for cat, tag in tagset:
            # не очень эффективно индекс искать постоянно
            
            cat_id = cat_order.index(cat)            
            y[cat_id][idx, 0] = 0
            y[cat_id][idx, morpho_map[cat][tag]] = 1            
    return y
        
    
def preproc_dataset(full_tag_sentences, stemmer):    
    sentences = []
    flexes = []
    token_tags = []
    
    for sent in full_tag_sentences:
        temp_sent = []
        temp_flexes = []
        for token_info in sent:
            token = token_info[0].lower()          
            splits = split_word(token, stemmer)
            temp_sent.append(splits[0])
            temp_flexes.append(splits[1])
            token_tags.append(token_info[2])  # надо бы переделать под стиль sentences или?          
        sentences.append(temp_sent)
        flexes.append(temp_flexes)    
    return sentences, flexes, token_tags
    

def make_train_dataset(path, stemmer, vocab_limit=20000, flex_vocab_limit=500, neighbors=3):
    full_tag_sentences, morpho_map = read_gikrya(path)
#     tag2id, id2tag = build_morpho_vocab(morpho_map)    
    sentences, flexes, token_tags = preproc_dataset(full_tag_sentences, stemmer)
    token2id, id2token = build_vocab(sentences, max_size=vocab_limit)
    flex2id, id2flex = build_vocab(flexes, max_size=flex_vocab_limit)
    X_stem = sentences_to_features(sentences, token2id, neighbors=neighbors)
    X_flex = sentences_to_features(flexes, flex2id, neighbors=neighbors)
    cat_order = sorted([key for key in morpho_map.keys()])
    y = tagsets_to_one_hot(token_tags, morpho_map, cat_order)
    return X_stem, X_flex, y, \
           morpho_map, cat_order, token2id, id2token, flex2id, id2flex, \
           sentences, flexes, token_tags, full_tag_sentences  
        
        
def make_test_dataset(path, stemmer,
                      morpho_map, cat_order,
                      token2id, id2token,
                      flex2id, id2flex, neighbors=3):
    full_tag_sentences, _ = read_gikrya(path)    
    sentences, flexes, token_tags = preproc_dataset(full_tag_sentences, stemmer)
    X_stem = sentences_to_features(sentences, token2id, neighbors=neighbors)
    X_flex = sentences_to_features(flexes, flex2id, neighbors=neighbors)
    y = tagsets_to_one_hot(token_tags, morpho_map, cat_order)
    return X_stem, X_flex, y, sentences, flexes, token_tags, full_tag_sentences 


def add_tags_to_sentences(full_tag_sentences, y, morpho_map, cat_order):
    new_full_tag_sents = []
    idx = 0
    for full_tag_sent in full_tag_sentences:
        new_full_tag = []   
        for token_info in full_tag_sent:
            tags = []
            for cat, oh_val in zip(cat_order, y):
                
                ntag = oh_val.shape[1]
                tags.append((cat,
                            [morpho_map[cat][i] for i in range(ntag) if oh_val[idx, i]==1][0]))
            new_full_tag.append((token_info[0],
                                '_',
                                tags))
            idx += 1
        new_full_tag_sents.append(new_full_tag)
    return new_full_tag_sents


def probs_to_one_hot(probs):
    one_hot = np.zeros_like(probs, dtype=np.int)
    for row in range(one_hot.shape[0]):
        one_hot[row, np.argmax(probs[row, :])] =1
    return one_hot


def many_probs_to_one_hot(probs):
    return [probs_to_one_hot(prob) for prob in probs]


def write_gikrya(path, full_tags):
    with open(path, 'w') as f:
        idx = 0
        for sentence in full_tags:
            for i, token_info in enumerate(sentence):
                f.write("{}\t{}\t{}\t{}\n".format(i+1,
                                                token_info[0],
                                                token_info[1],
                                                tagset2str(token_info[2])))
            f.write("\n")
            
                

                    
def tagset2str(tagset):
    POS = ""
    tags_list = []
    for tag, val in tagset:
        if  tag == "POS":
            POS = val
        else:
            if val != UNDEFINED:
                tags_list.append("{}={}".format(tag, val))
    tags = "_"
    if len(tags_list) > 0:
        tags = "|".join(tags_list)
    return "{}\t{}".format(POS, tags)
        

def main():
    if len(sys.argv) != 3:
        print("wrong params!\nrun format:\n{} train_path model_dir".format(sys.argv[0]))
        return None
        
    train_path = sys.argv[1]
    model_dir = sys.argv[2]
    
    stemmer = SnowballStemmer("russian")
    X_train, X_flex_train, y_train, \
               morpho_map, cat_order, token2id, id2token, flex2id, id2flex, \
               sentences, flexes, token_tags_train, full_tag_train = \
    make_train_dataset(gikrya_path, stemmer, 
                       vocab_limit=300000,
                       flex_vocab_limit=500,
                       neighbors=3)
    
    gikrya_test = "../morphoRuEval-2017/Baseline/source/gikrya_test.txt"
    X_test, X_flex_test, y_test, sentences, \
    flexes, token_tags, full_tag_test = make_test_dataset(gikrya_test, stemmer, 
                                                          morpho_map, cat_order,
                                                          token2id, id2token,
                                                          flex2id, id2flex,
                                                          neighbors=3)
    # Embedding dimensions.
    tok2vec_dim = 40
    flex2vec_dim = 10
    token_hidden = 128
    flex_hidden = 32
    
    root_in = Input(shape=(X.shape[1],))

    flex_in = Input(shape=(X_flex.shape[1], ))    
    root_embedding = Embedding(input_dim=len(token2id), output_dim=tok2vec_dim)(root_in)
    flex_embedding = Embedding(input_dim=len(flex2id), output_dim=flex2vec_dim)(flex_in)

    encoded_root = Bidirectional(GRU(token_hidden,
                                       dropout_U=0.1, 
                                       dropout_W=0.1))(root_embedding)

    encoded_flex = Bidirectional(GRU(flex_hidden,
                                       dropout_U=0.1, 
                                       dropout_W=0.1))(flex_embedding)

    merge_encoded = keras.layers.merge([encoded_root, encoded_flex], mode='concat')

    # prediction = Dense(output_dim=num_classes, activation='softmax')(merge_encoded)

    predictions = [Dense(output_dim=tag_y.shape[1], activation='softmax')(merge_encoded)
                  for tag_y in y_train]
    
    model = Model([root_in, flex_in], predictions)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
if __name__ == "__main__":
    main()
    
    
    
    