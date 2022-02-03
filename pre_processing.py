import pandas as pd
import os.path
import _pickle as cPickle
import numpy as np
import keras.utils
import time
from keras.callbacks import TensorBoard, CSVLogger
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Conv1D,GlobalMaxPool1D,Dropout,Bidirectional
from keras.layers.embeddings import Embedding
from tensorflow.keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model
from nltk.corpus import stopwords
import operator
from nltk.corpus import stopwords
import spacy
import nltk
#nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

frequent_speakers = {'barack-obama': 0, 'donald-trump': 1, 'hillary-clinton': 2,
                     'mitt-romney': 3, 'scott-walker': 4, 'john-mccain': 5,
                     'rick-perry': 6, 'chain-email': 7, 'marco-rubio': 8, 'viral-image': 13,
                     'rick-scott': 9, 'ted-cruz': 10, 'bernie-s': 11, 'newt-gingrich': 16,
                     'chris-christie': 12, 'facebook-posts': 13, 'blog-posting': 13,
                     'charlie-crist': 14, 'congressional': 15, 'republican': 15,
                     'national-committe': 15, 'democratic': 15}


def get_speaker_id(speaker):
    if isinstance(speaker, str):
        matched = [sp for sp in frequent_speakers if sp in speaker.lower()]
        if len(matched) > 0:
            return frequent_speakers[matched[0]]
        else:
            return len(set(frequent_speakers.values()))
    else:
        return len(set(frequent_speakers.values()))


frequent_jobs = {'senator': 0, 'president': 1, 'governor': 2,
                 'u.s. representative': 3, 'attorney': 4, 'congressman': 5,
                 'congresswoman': 5, 'social media posting': 6, 'lawyer': 4,
                 'businessman': 6, 'radio host': 8, 'host': 8,
                 'mayor': 7, 'assembly': 9, 'representative': 3,
                 'senate': 9, 'state representative': 3, 'milwaukee county executive': 10,
                 'u.s. house of representatives': 3, 'house representative': 3,
                 'house of representatives': 3, 'house member': 3}


def get_job_id(job):
    if isinstance(job, str):
        matched = [jb for jb in frequent_jobs if jb in job.lower()]
        if len(matched) > 0:
            return frequent_jobs[matched[0]]
        else:
            return len(set(frequent_jobs.values()))
    else:
        return len(set(frequent_jobs.values()))


frequent_parties = {'republican': 0, 'democrat': 1, 'none': 2, 'organization': 3, 'independent': 4}


def get_party_id(party):
    if isinstance(party, str):
        matched = [pt for pt in frequent_parties if pt in party.lower()]
        if len(matched) > 0:
            return frequent_parties[matched[0]]
        else:
            return len(set(frequent_parties.values()))
    else:
        return len(set(frequent_parties.values()))


other_states = ['wyoming', 'colorado', 'hawaii', 'tennessee', 'nevada', 'maine',
                'north dakota', 'mississippi', 'south dakota', 'oklahoma',
                'delaware', 'minnesota', 'north carolina', 'arkansas', 'indiana',
                'maryland', 'louisiana', 'idaho', 'iowa', 'west virginia',
                'michigan', 'kansas', 'utah', 'connecticut', 'montana', 'vermont',
                'pennsylvania', 'alaska', 'kentucky', 'nebraska', 'new hampshire',
                'missouri', 'south carolina', 'alabama', 'new mexico']

frequent_states = {'texas': 1, 'florida': 2, 'wisconsin': 3, 'new york': 4,
                   'illinois': 5, 'ohio': 6, 'georgia': 7, 'virginia': 8,
                   'rhode island': 9, 'oregon': 10, 'new jersey': 11,
                   'massachusetts': 12, 'arizona': 13, 'california': 14,
                   'washington': 15}


def get_state_id(state):
    if isinstance(state, str):
        if state.lower().rstrip() in frequent_states:
            return frequent_states[state.lower().rstrip()]
        elif state.lower().rstrip() in other_states:
            return 0
        else:
            if 'washington' in state.lower():
                return frequent_states['washington']
            else:
                return len(set(frequent_states.values()))+1
    else:
        return len(set(frequent_states.values()))+1


frequent_subjects = {'health': 0, 'tax': 1, 'immigration': 2, 'election': 3,
                     'education': 4, 'candidates-biography': 5, 'economy': 6,
                     'gun': 7, 'job': 8, 'federal-budget': 6, 'energy': 9,
                     'abortion': 10, 'foreign-policy': 6, 'state-budget': 6,
                     'crime': 11, 'gays-and-lesbians': 12, 'medicare': 0,
                     'terrorism': 11, 'finance': 6, 'criminal': 11,
                     'transportation': 13}


def get_subject_id(subject):
    if isinstance(subject, str):
        matched = [sbj for sbj in frequent_subjects if sbj in subject.lower()]
        if len(matched) > 0:
            return frequent_subjects[matched[0]]
        else:
            return len(set(frequent_subjects.values()))
    else:
        return len(set(frequent_subjects.values()))


frequent_venues = {'news release': 0, 'interview': 1, 'press release': 2,
                   'speech': 3, 'tv': 4, 'tweet': 5, 'campaign': 6,
                   'television': 4, 'debate': 7, 'news conference': 8,
                   'facebook': 5, 'press conference': 8, 'radio': 9,
                   'e-mail': 10, 'email': 10, 'mail': 10, 'social media': 5,
                   'twitter': 5, 'blog': 11, 'article': 11, 'comment': 12, 'web': 11}


def get_venue_id(venue):
    if isinstance(venue, str):
        matched = [ven for ven in frequent_venues if ven in venue.lower()]
        if len(matched) > 0:
            return frequent_venues[matched[0]]
        else:
            return len(set(frequent_venues.values()))
    else:
        return len(set(frequent_venues.values()))


num_party = 6
num_state = 17
num_venue = 14
num_job = 12
num_sub = 15
num_speaker = 18



def metadata_processing(party="", state="", venue="", job="", subject="", speaker=""):
    data= {'party_id' : [get_party_id(party)],
        'state_id' : [get_state_id(state)],
        'venue_id' : [get_venue_id(venue)],
        'job_id' : [get_job_id(job)],
        'subject_id' : [get_subject_id(subject)],
        'speaker_id' : [get_speaker_id(speaker)]}
    data = pd.DataFrame(data)
    party_test = keras.utils.np_utils.to_categorical(data['party_id'], num_classes=num_party)
    state_test = keras.utils.np_utils.to_categorical(data['state_id'], num_classes=num_state)
    venue_test = keras.utils.np_utils.to_categorical(data['venue_id'], num_classes=num_venue)
    job_test = keras.utils.np_utils.to_categorical(data['job_id'], num_classes=num_job)
    subject_test = keras.utils.np_utils.to_categorical(data['subject_id'], num_classes=num_sub)
    speaker_test = keras.utils.np_utils.to_categorical(data['speaker_id'], num_classes=num_speaker)
    meta = np.hstack((party_test, state_test, venue_test, job_test, subject_test, speaker_test))
    return meta


def load_statement_vocab_dict():
    vocabulary_dict = {}
    # print('Loading Vocabulary Dictionary...')
    vocabulary_dict = cPickle.load(open("vocabulary.p", "rb"))
    return vocabulary_dict


def statement_preprocessing(statement):
    vocabulary_dict = load_statement_vocab_dict()
    statement = [w for w in statement.split(' ') if w not in stopwords.words('english')]
    statement = ' '.join(statement)
    text = text_to_word_sequence(statement)
    val = [0] * 10
    val = [vocabulary_dict[t] for t in text if t in vocabulary_dict]
    return val


pos_tags = {'ADJ': 'adjective', 'ADP': 'adposition', 'ADV': 'adverb',
            'AUX': 'auxiliary verb', 'CONJ': 'coordinating conjunction',
            'DET': 'determiner', 'INTJ': 'interjection', 'NOUN': 'noun',
            'NUM': 'numeral', 'PART': 'particle', 'PRON': 'pronoun',
            'PROPN': 'proper noun', 'PUNCT': 'punctuation', 'X': 'other',
            'SCONJ': 'subord conjunction', 'SYM': 'symbol', 'VERB': 'verb'}


pos_dict = {'NOUN' : 0, 'VERB' : 1, 'ADP' : 2, 'PROPN' : 3, 'PUNCT' : 4,
            'DET' : 5, 'ADJ' : 6, 'NUM' : 7, 'ADV' : 8, 'PRON' : 9, 'X' : 9,
            'PART' : 9, 'SYM' : 9, 'INTJ' : 9 }


def get_pos(statement):
    doc = nlp(statement)
    taglist = []
    for token in doc:
        taglist.append(pos_dict.get(token.pos_, max(pos_dict.values())))
    return taglist


dep_dict = {'punct' : 0, 'prep' : 1, 'pobj' : 2, 'compound' : 3, 'det' : 4,
            'nsubj' : 5, 'ROOT' : 6, 'amod' : 7, 'dobj' : 8, 'aux' : 9,
            'advmod' : 10, 'nummod' : 10, 'ccomp' : 10, 'conj' : 10, 'cc' : 10,
            'advcl' : 10, 'poss' : 10, 'mark' : 10, 'quantmod' : 10, 'relcl' : 10,
            'attr' : 10, 'xcomp' : 10, 'npadvmod' : 10, 'nmod' : 10, 'auxpass' : 10,
            'acl' : 10, 'nsubjpass' : 10, 'pcomp' : 10, 'acomp' : 10, 'neg' : 10,
            'appos' : 10, 'prt' : 10, '' : 10, 'expl' : 10, 'dative' : 10,
            'agent' : 10, 'case' : 10, 'oprd' : 10, 'csubj' : 10, 'dep' : 10,
            'intj' : 10, 'predet' : 10, 'parataxis' : 10, 'preconj' : 10,
            'meta' : 10, 'csubjpass' : 10}


def get_dep_parse(statement):
    doc = nlp(statement)
    deplist = []
    for token in doc:
        deplist.append(dep_dict.get(token.dep_, max(dep_dict.values())))
    return deplist


def get_word_embeddings():
    embeddings = {}
    with open("glove.twitter.27B.100d.txt") as file_object:
        for line in file_object:
            word_embed = line.split()
            word = word_embed[0]
            embed = np.array(word_embed[1:], dtype="float32")
            embeddings[word.lower()] = embed

    EMBED_DIM = 100
    print(len(embeddings), " : Word Embeddings Found")
    print(len(embeddings[word]), " : Embedding Dimension")
    vocabulary_dict = load_statement_vocab_dict()
    num_words = len(vocabulary_dict) + 1
    embedding_matrix = np.zeros((num_words, EMBED_DIM))
    for word, i in vocabulary_dict.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embeddings_index = None

    pos_embeddings = np.identity(max(pos_dict.values()), dtype=int)
    dep_embeddings = np.identity(max(dep_dict.values()), dtype=int)
    return len(vocabulary_dict.keys()),embedding_matrix,embeddings_index, pos_embeddings, dep_embeddings


def stmt_processing(statement):
    word_id = [statement_preprocessing(statement)]
    pos_id = [get_pos(statement)]
    dep_id = [get_dep_parse(statement)]
    num_steps = 15
    word_id = sequence.pad_sequences(word_id, maxlen=num_steps, padding='post', truncating='post')
    pos_id = sequence.pad_sequences(pos_id, maxlen=num_steps, padding='post', truncating='post')
    dep_id = sequence.pad_sequences(dep_id, maxlen=num_steps, padding='post', truncating='post')
    return word_id, pos_id,dep_id




