Dynamic Bernoulli Embeddings
============================

A PyTorch implementation of Dynamic Bernoulli Embeddings [1] for training word embeddings that smoothly evolve over time.

Introduction
------------

This is an easy to use, pip installable, PyTorch based implementation of Dynamic Bernoulli Embeddings [1]. The paper authors provide their implementation [here](https://github.com/mariru/dynamic_bernoulli_embeddings), but I found it a little difficult to use. For this reason as well as the opportunity to get some practice in PyTorch, I decided to implement my own version.

With this model, every term gets an embedding for every timestep -- `T * V` embeddings in all where `T` is the number of timesteps and `V` is the size of the vocabulary.

Check out [this Kaggle kernel](https://www.kaggle.com/llefebure/dynamic-bernoulli-embeddings) for an end to end application of this model on an interesting dataset.

Quick Start
-----------

Install with:

```
pip install git+git://github.com/llefebure/dynamic_bernoulli_embeddings.git
```

The train_model function expects a pandas data frame with at least two columns, `bow` and `time`. `bow` is just a list of words in the document, and `time` is expected to be an integer in `[0, T)` where `T` is the total number of timesteps. It also expects a dictionary mapping tokens to their index in `[0, V)` where `V` is the size of the vocabulary. Any token found in the dataset but not in the vocabulary is ignored.

```python
from dynamic_bernoulli_embeddings.training import train_model
model, loss_history = train_model(dataset, dictionary)
embeddings = model.get_embeddings()  # Will be of shape (T, V, k) -- k is the embedding dimension
```

Once you have the embeddings, you can use the analysis class to do some analysis.

```python
from dynamic_bernoulli_embeddings.analysis import DynamicEmbeddingAnalysis
emb = DynamicEmbeddingAnalysis(model.get_embeddings(), dictionary)
emb.absolute_drift()  # Terms that changed between the first and last timesteps
emb.neighborhood("climate", t)  # Find nearby terms for "climate" at time `t`
emb.change_points()  # Find (term, time) pairs with the largest shift from the previous timestep
```

References
----------
[1] [Dynamic Embeddings for Language Evolution](http://www.cs.columbia.edu/~blei/papers/RudolphBlei2018.pdf)
