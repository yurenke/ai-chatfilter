import os
from enum import Enum
import pandas as pd
import numpy as np
import re
import joblib
from scipy import sparse
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from opencc import OpenCC
import fasttext
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ads_path = os.path.join(BASE_DIR, 'ai/assets/vi_abb/abb_dict_special.xlsx')
abb_dict_special = pd.read_excel(ads_path)
adn_path = os.path.join(BASE_DIR, 'ai/assets/vi_abb/abb_dict_normal.xlsx')
abb_dict_normal = pd.read_excel(adn_path)

abb_model = joblib.load(os.path.join(BASE_DIR, 'ai/assets/vi_abb/abb_model.sav'))
label_enc = joblib.load(os.path.join(BASE_DIR, 'ai/assets/vi_abb/label_encoder.joblib'))
dict_vect = joblib.load(os.path.join(BASE_DIR, 'ai/assets/vi_abb/dict_vectorizer.joblib'))
tfidf_vect = joblib.load(os.path.join(BASE_DIR, 'ai/assets/vi_abb/tfidf_vectorizer.joblib'))
da_path = os.path.join(BASE_DIR, 'ai/assets/vi_abb/abbreviation_dictionary_vn.xlsx')
duplicate_abb = pd.read_excel(da_path, sheet_name='duplicate', header=None)
duplicate_abb = list(duplicate_abb[0])
cc = OpenCC('t2s')
fasttext_model = fasttext.load_model(os.path.join(BASE_DIR, 'ai/_models/fasttext_pretrained_model/lid.176.ftz'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()
alay_dict = pd.read_csv(os.path.join(BASE_DIR, 'ai/assets/indonesian/new_kamusalay.csv'), encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 
                                      1: 'replacement'})
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))

id_stopword_dict = pd.read_csv(os.path.join(BASE_DIR, 'ai/assets/indonesian/stopwordbahasa.csv'), header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})

class Language(Enum):
    ZH = "zh"
    ID = "id"
    TH = "th"
    VI = "vi"
    OTHER = "other"

# def remove_characters_emoji(text):
#     ch_emojis = [':)', '=)', ':(', ':v', '-_-', ':3', '<3', '@@', ':D', ':>', ':">', '=]', ':<', '^_^', '^^', ':-)', '><', '>.<', '~~', ':p', ':-p']
#     for e in ch_emojis:
#         text = text.replace(e, " ")
#     text = re.sub('  +', ' ', text).strip()
#     return text

def detect_language(text: str, threshold: float = 0.85) -> Language:
    try:
        labels, probs = fasttext_model.predict(text)
        if not labels or probs[0] < threshold:
            return Language.OTHER

        lang_code = labels[0].replace("__label__", "").lower()

        if lang_code.startswith("zh"):
            return Language.ZH
        elif lang_code == "id":
            return Language.ID
        elif lang_code == "th":
            return Language.TH
        elif lang_code == "vi":
            return Language.VI
        else:
            return Language.OTHER
    except Exception:
        return Language.OTHER

def replace_url(text):
    URL_PATTERN = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    text = re.sub(URL_PATTERN, ' URL ', text)
    text = re.sub('  +', ' ', text).strip()
    return text

def replace_repeated_character(text):
    def _replace_rep(m):
        c,cc = m.groups()
        return f'{c}'
    re_rep = re.compile(r'(\S)(\1{2,})')
    return re_rep.sub(_replace_rep, text)

def remove_special_character(text):
    text = re.sub(r'\d+', lambda m: " ", text)
    # text = re.sub(r'\b(\w+)\s+\1\b',' ', text) #remove duplicate number word
    text = re.sub("[~!@#$%^&*()_+{}“”|:\"<>?`´\-=[\]\;\\\/.,]", " ", text)
    text = re.sub('  +', ' ', text).strip()
    return text

def remove_special_character_1(text):  # remove dot and comma
    text = re.sub("[.,?!]", " ", text)
    text = re.sub('  +', ' ', text).strip()
    return text

def normalize_alay_for_id(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def remove_stopword_for_id(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def stemming_for_id(text):
    return stemmer.stem(text)

def abbreviation_kk(text):
    text = str(text)
    for t in text.split():
        if 'kk' in t:
            text = text.replace(t, ' ha ha ')
        else:
            if 'kaka' in t:
                text = text.replace(t, ' ha ha ')
            else:
                if 'kiki' in t:
                    text = text.replace(t, ' ha ha ')
                else:
                    if 'haha' in t:
                        text = text.replace(t, ' ha ha ')
                    else:
                        if 'hihi' in t:
                            text = text.replace(t, ' ha ha ')
    text = re.sub('  +', ' ', text).strip()
    return text

def annotations(dataset):
    pos = []
    max_len = 8000
    for i in range(dataset.shape[0]):
        n = len(dataset.at[i, 'cmt'])
        l = [0] * max_len
        s = int(dataset.at[i, 'start_index'])
        e = int(dataset.at[i, 'end_index'])
        for j in range(s, e):
            l[j] = 1
        pos.append(l)
    return pos

def abbreviation_normal(text):  # len word equal 1
    text = str(text)
    temp = ''
    for word in text.split():
        for i in range(abb_dict_normal.shape[0]):
            if str(abb_dict_normal.at[i, 'abbreviation']) == str(word):
                word = str(abb_dict_normal.at[i, 'meaning'])
        temp = temp + ' ' + word
    text = temp
    text = re.sub('  +', ' ', text).strip()
    return text

def abbreviation_special(text):  # including special character and number
    text = ' ' + str(text) + ' '
    for i in range(abb_dict_special.shape[0]):
        text = text.replace(' ' + abb_dict_special.at[i, 'abbreviation'] + ' ',
                            ' ' + abb_dict_special.at[i, 'meaning'] + ' ')
    text = re.sub('  +', ' ', text).strip()
    return text

def abbreviation_predict(t):
    text = str(t)
    max_len = 8000
    if len(t) > max_len:
        text = t[:max_len]

    cmt = ' ' + text + ' '
    for abb in duplicate_abb:
        start_index = 0
        count = 0
        while start_index > -1:  # start_index = -1 -> abb is not in cmt
            start_index = cmt.find(' ' + abb + ' ')  # find will return FIRST index abb in cmt
            if start_index > -1:
                end_index = start_index + len(abb)
                t = pd.DataFrame([[abb, start_index, end_index, text]],
                                 columns=['abb', 'start_index', 'end_index', 'cmt'], index=None)
                temp = annotations(t)
                X_pos = sparse.csr_matrix(np.asarray(temp))

                X_abb = dict_vect.transform(t[['abb']].to_dict('records'))
                # print(t['cmt'])
                X_text = tfidf_vect.transform([text])

                X = hstack((X_abb, X_pos, X_text))
                predict = abb_model.predict(X)
                origin = label_enc.inverse_transform(predict.argmax(axis=1))
                origin = ''.join(origin)
                text = text[:start_index + count * (len(origin) - len(abb))] + origin + text[end_index + count * (
                            len(origin) - len(abb)):]
                text = ''.join(text)
                count = count + 1
                for i in range(start_index + 1, end_index + 1):  # replace abb to space ' '
                    cmt = cmt[:i] + ' ' + cmt[i + 1:]
    return text

def preprocess_chat_text_for_ch(text):
    text = replace_url(text)
    # text = replace_repeated_character(text)
    text = remove_special_character(text)
    # text = cc.convert(text)
    return text

def preprocess_chat_text_for_thai_and_others(text):
    text = replace_url(text)
    text = replace_repeated_character(text)
    text = remove_special_character(text)
    return text

def preprocess_chat_text_for_vi(text):
    text = replace_url(text)
    text = remove_special_character_1(text)  # ##remove , . ? !
    text = abbreviation_kk(text)
    text = abbreviation_special(text)
    text = replace_repeated_character(text)
    text = remove_special_character(text)
    text = abbreviation_normal(text)
    text = abbreviation_predict(text)
    return text

def preprocess_chat_text_for_id(text):
    text = text.lower()
    text = replace_url(text)
    text = remove_special_character(text)
    text = normalize_alay_for_id(text)
    text = stemming_for_id(text)
    text = remove_stopword_for_id(text)
    return text