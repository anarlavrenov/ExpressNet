# ExpressNet

![image](https://github.com/anarlavrenov/ExpressNet/blob/master/logo.png)

ExpressNet is an autoregressive BiLSTM-based decoder-only model with Additive Attention Mechanism (Bahdanau Attention). 
Developed by [Anar Lavrenov, Head of AI at SPUNCH](https://www.linkedin.com/in/anar-lavrenov/).
For now ExpressNet is made for binary/multi classification tasks and there are plans for adding other tasks as well.
The main distinguishing feature of this model is high perfomance without text preprocessing. 

# Architecture
![image](https://github.com/anarlavrenov/ExpressNet/blob/master/model_scheme.png)


# Quick Start in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/anarlavrenov/ExpressNet/blob/master/usage.ipynb)

Clone this repository
```py
!git clone https://github.com/anarlavrenov/ExpressNet
%cd ExpressNet
```
Import model
```py
from ExpressNet.model import ExpressNet
```

Initialize model with your own hyperparameters
```py
model = ExpressNet(
    d_model=256,
    vocab_size=len(vocab),
    classification_type="multiclass",
    n_classes=4
).to(device)
```


# Validation
ExpressNet showed decent results on validatation on most of torchtext datasets.
General parameters were used:
1. No text preprocessing at all: no stopwords removal, no lemmatization etc.
2. `basic english` torch tokenizer everywhere
3. d_model: 256 everywhere

![image](https://github.com/anarlavrenov/ExpressNet/blob/master/benchmark.png)


# Usage Purposes

1. Primarily usage: playground for Machine Learning Researches and Data Scientists. You are very welcome to share your insights and recommendations.
2. Secondary usage: baseline for most of classification tasks without any text preprocessing. If you want to achieve instant high validation accuracy - you are welcome to use ExpressNet. 
