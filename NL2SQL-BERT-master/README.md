# NL2SQL-BERT

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Content Enhanced BERT-based Text-to-SQL Generation https://arxiv.org/abs/1910.07179

# Requirements

python 3.6

records 0.5.3   

torch 1.1.0   

# Run

1, Data prepare:
Download all origin data( https://drive.google.com/file/d/1iJvsf38f16el58H4NPINQ7uzal5-V4v4 ) and put them at `data_and_model` directory.

Then run
`data_and_model/output_entity.py`

2, Train and eval:

`train.py`

# Trained model
https://drive.google.com/drive/folders/1AihPMAkwS9N6rW6jQTXODnGloD3VUMxn?usp=sharing


# Results on BERT-Base-Uncased without EG
| Dev <br />logical form <br />accuracy | Dev<br />execution<br/> accuracy | 
| ------------------------------------- | -------------------------------- |
| 83.9                      | 89.9                |
