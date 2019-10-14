# PyTorch implementation for NER with CoNLL 2003 using pre-trained BERT

This repository tries to replicate BERT's results on CoNLL 2003 NER task.

With `BERT-BASE-CASED`, the result is as follows on `eval` set:

```
           precision    recall  f1-score   support

      LOC       0.97      0.97      0.97      1837
     MISC       0.89      0.92      0.90       922
      PER       0.97      0.98      0.98      1836
      ORG       0.92      0.94      0.93      1341

micro avg       0.95      0.96      0.95      5936
macro avg       0.95      0.96      0.95      5936
```

To reproduce:
```
 python train.py --batch_size 32 --lr 3e-5 --n_epochs 5
```

Credits:

Key ideas to get this working are due to [this github issue/comment](https://github.com/huggingface/transformers/issues/64#issuecomment-443703063).