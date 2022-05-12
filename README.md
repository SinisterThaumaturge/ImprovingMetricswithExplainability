# Improving Metrics with Explainability

## Usage

Install the packages listed in [requirements.txt](requirements.txt) into a new virtual environment.

```python
conda create -n metrics python
conda activate metrics
pip install -r requirements.txt
```

run :

```python
LIME:
python explain_lime.py --src $src_lang --tgt $tgt_lang

SHAP:
python explain_shap.py --src $src_lang --tgt $tgt_lang
```

Creating data for WMT17 is still work in progress.

## Contents:

Here is a short overview of the repository contents:

- **data**:
  - dev: dev set for Eval4NLP
  - test21: test set for Eval4NLP
  - WMT17 data for WMT17
- explain_shap : use SHAP to create word scores
- explain_lime : use LIME to create word scores
- evaluation.ipynb : Notebook for combining aggregated word scores with sentence score of BERTScore

## References:

- Generalized Mean: https://en.wikipedia.org/wiki/Generalized_mean
- LIME: https://github.com/marcotcr/lime
- SHAP: https://shap.readthedocs.io/en/latest/
