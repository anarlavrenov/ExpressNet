# ExpressNet

ExpressNet is an autoregressive BiLSTM-based decoder-only model with Additive Attention Mechanism (Bahdanau Attention) . 
Developed by [Anar Lavrenov, Head of AI at SPUNCH](https://www.linkedin.com/in/anar-lavrenov/)
For now this model is made for binary/multi classification tasks and I have plans for other tasks implementation as well.

This model was tested on most torchtext.datasets as a benchmark.
General parameters were used:
1. No text preprocessing at all: no stopwords removal, no lemmatization etc.
2. `basic english` torch tokenizer
3. d_model: 256 everywhere

![image](https://github.com/anarlavrenov/ExpressNet/blob/master/benchmark.png)



