from bert_score.scorer import BERTScorer
from lime.lime_text import LimeTextExplainer
import shap
import tqdm
import numpy as np
from utils import NpEncoder
import json
from scipy.stats.mstats import gmean
from numpy import mean
import pandas as pd
from evaluation_utils import evaluate_sentence_level
import argparse

model_type = 'joeddav/xlm-roberta-large-xnli'
num_layers = 16
idf_sents = 0
scorer = BERTScorer(model_type=model_type, batch_size=64, num_layers=num_layers)

parser = argparse.ArgumentParser(description='Explain Lime')
parser.add_argument('--src','-s')
parser.add_argument('--tgt','-t')
parser.add_argument('--createsrc', action='store_true')


args = parser.parse_args()
SRC_LANG = 'et'
if args.src:
  SRC_LANG = args.src
TGT_LANG = 'en'
if args.tgt:
  TGT_LANG = args.tgt

#change to test21 if you want to use the test data
SPLIT = 'dev'
data_dir = f'data/{SPLIT}/{SRC_LANG}-{TGT_LANG}-{SPLIT}'
src = [s.strip() for s in open(f'{data_dir}/{SPLIT}.src').readlines()]
tgt = [s.strip() for s in open(f'{data_dir}/{SPLIT}.mt').readlines()]
wor = [list(map(int, s.strip().split())) for s in open(f'{data_dir}/{SPLIT}.tgt-tags').readlines()]
sen = [float(s.strip()) for s in open(f'{data_dir}/{SPLIT}.da').readlines()]
assert len(src) == len(tgt) == len(wor) == len(sen)
dataset = {'src': src, 'tgt': tgt, 'word_labels': wor, 'sent_labels': sen}

explainer = LimeTextExplainer(class_names=['score', 'score'], bow=False, split_expression = ' ')

class ExplainableWrapper:
   def predict_proba(self,translations):
    translations = [s[0] for s in translations]
    text_src = [self.src_sent] * len(translations)
    preds, _ , _= self.scorer.score(translations,text_src)
    return np.array(preds[2])

   def mask_model(self, mask, x):
    tokens = []
    for mm, tt in zip(mask,x):
      if mm: tokens.append(tt)
      else: tokens.append('[MASK]')
    trans_sent = ' '.join(tokens)
    sentence = pd.DataFrame([trans_sent])
    return sentence

   def build_feature(self, trans_sent):
    tokens = trans_sent.split()
    tdict = {}
    for i,tt in enumerate(tokens):
      tdict['{}_{}'.format(tt,i)] = tt
    df = pd.DataFrame(tdict, index=[0])
    return df

   def __init__(self, scorer):
    self.scorer = scorer
    self.explainer = shap.Explainer(self.predict_proba, self.mask_model)
    self.src_sent = None
    self.rows = 0

   def explain(self, src_sent, trans_sent, plot=False):
    self.src_sent = src_sent
    value = self.explainer(self.build_feature(trans_sent))
    if plot: shap.waterfall_plot(value[0])
    all_tokens = trans_sent.split()
    return [(token,sv) for token, sv in zip(all_tokens,value[0].values)]


def explain_instance(explainer, text_a, text_b):
    def predict_proba(texts):
        src = [text_a] * len(texts)
        scores, _,_= scorer.score(texts,src)
        preds = np.array(scores[2]) 
        results = np.vstack((preds, preds)).T
        return results

    #predictions, raw_outputs = model.predict([[text_a, text_b]])
    exp = explainer.explain_instance(text_b, predict_proba, num_features=len(text_b.split()), labels=(1, ),num_samples=100)
    return exp.as_map()

results = []
explain_model = ExplainableWrapper(scorer)
for i in tqdm.tqdm(range(1000)):
    exp = explain_model.explain(dataset['src'][i], dataset['tgt'][i])
    results.append(exp)

bertscores , _,_ = scorer.score(dataset['tgt'], dataset['src'])

#save results
with open(f"shap_results{SRC_LANG+TGT_LANG}_16.json", 'w') as f:
  json.dump(results, f,cls=NpEncoder)

# try:
#     sentence_scores = [ gmean([w[1] for w in s]) for s in results]
# except KeyError:
#     sentence_scores = [ gmean([w[1] for w in s]) for s in results]
# sentence_scores = np.nan_to_num(sentence_scores,0)
# for i in range(10):
#     cumscores =  np.add(bertscores*i/10,sentence_scores*i/10)
#     evaluate_sentence_level(sen,cumscores)
#     print(f"I: {i}: ")
#     evaluate_sentence_level(sen,sentence_scores)