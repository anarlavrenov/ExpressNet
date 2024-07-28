# ExpressNet

ExpressNet is an autoregressive BiLSTM-based decoder-only model with Additive Attention Mechanism (Bahdanau Attention). 
Developed by [Anar Lavrenov, Head of AI at SPUNCH](https://www.linkedin.com/in/anar-lavrenov/).
For now this model is made for binary/multi classification tasks and there are plans for adding other tasks as well.

# Validation
ExpressNet showed decent results on validatation on most of torchtext datasets.
General parameters were used:
1. No text preprocessing at all: no stopwords removal, no lemmatization etc.
2. `basic english` torch tokenizer
3. d_model: 256 everywhere

![image](https://github.com/anarlavrenov/ExpressNet/blob/master/benchmark.png)


# Usage

1. Primararly usage: 
