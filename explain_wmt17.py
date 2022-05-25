from bert_score.scorer import BERTScorer
from lime.lime_text import LimeTextExplainer
import tqdm
import numpy as np
from utils import NpEncoder
import json
from scipy.stats.mstats import gmean
from numpy import mean
from evaluation_utils import evaluate_sentence_level
import argparse
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

model_type = 'joeddav/xlm-roberta-large-xnli'
num_layers = 16
idf_sents = 0
scorer = BERTScorer(model_type=model_type, batch_size=64, num_layers=num_layers)
SPLIT = 'test21'
data_dir = f'data/{SPLIT}/{SRC_LANG}-{TGT_LANG}-{SPLIT}'
src = [s.strip() for s in open(f'{data_dir}/{SPLIT}.src').readlines()]
tgt = [s.strip() for s in open(f'{data_dir}/{SPLIT}.mt').readlines()]
wor = [list(map(int, s.strip().split())) for s in open(f'{data_dir}/{SPLIT}.tgt-tags').readlines()]
sen = [float(s.strip()) for s in open(f'{data_dir}/{SPLIT}.da').readlines()]
assert len(src) == len(tgt) == len(wor) == len(sen)
dataset = {'src': src, 'tgt': tgt, 'word_labels': wor, 'sent_labels': sen}


explainer = LimeTextExplainer(class_names=['score', 'score'], bow=False, split_expression = ' ')


def explain_instance(explainer, text_a, text_b):
    def predict_proba(texts):
        src = [text_a] * len(texts)
        tgt = [text_b] * len(texts)
        # src scores for LIME
        if args.createsrc:
          scores, _,_= scorer.score(tgt,texts)
        else:# tgt scores for LIME
          scores, _,_= scorer.score(texts,tgt)
        preds = np.array(scores[2]) 
        results = np.vstack((preds, preds)).T
        return results

    #predictions, raw_outputs = model.predict([[text_a, text_b]])
    #src scores
    if args.createsrc:
      exp = explainer.explain_instance(text_a, predict_proba, num_features=len(text_a.split()), labels=(1, ),num_samples=100)
    else:#tgt scores
      exp = explainer.explain_instance(text_b, predict_proba, num_features=len(text_b.split()), labels=(1, ),num_samples=100)
    return exp.as_map()

results = []
for idx in tqdm.tqdm(range(len(dataset['src']))):
  expl = explain_instance(explainer, dataset['src'][idx], dataset['tgt'][idx])
  results.append(expl)

bertscores , _,_ = scorer.score(dataset['tgt'], dataset['src'])

#save lime word scores
with open(f"limeresults_{SRC_LANG+TGT_LANG}_16.json", 'w') as f:
  json.dump(results, f,cls=NpEncoder)