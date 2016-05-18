# coheoka

Python coherence evaluation tool using Stanford's CoreNLP.

This repository is designed for entity-base coherence.

## Prerequisite

You must run a CoreNLP server on your own if you want to run any module in this repository.

You can download Stanford CoreNLP latest version (3.6.0) at [here](http://stanfordnlp.github.io/CoreNLP/download.html) and run a local server (requiring Java 1.8) by this way:

```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```

Then you can find a demo at [`localhost:9000`](http://localhost:9000/), which visualizes StanfordCoreNLP's sophisticated annotation for English documents.

Also, there is an online demo maintained by Stanford at [here](http://corenlp.run/).

If you need to annotate lots of documents, you **must** run a local server on your own. Otherwise you may want to set an environment variable `CORENLP_URL` to use other's server (e.g. `http://corenlp.run/` and don't forget the `http`).

Also, if you are using Windows, make sure you have installed any Python's scientific distribution such as [Anaconda](https://www.continuum.io/downloads) (If you want many scientific packages) or [Miniconda](http://conda.pydata.org/miniconda.html) (If you don't want to use too much disk space) which I strongly recommend.

## Install

The requirements are `nltk`, `numpy`, `pandas`, `requests`, `scipy` and `scikit-learn`.

If you have installed Anaconda or Miniconda just
```
conda create -n coheoka --file requirements.txt
activate coheoka  # Windows
```
or
```
source activate coheoka  # Linux
```

Check out [conda documentation](http://conda.pydata.org/docs/using/envs.html#create-an-environment) for more details.

## Reference
1. Barzilay, R., & Lapata, M. (2008).
    Modeling local coherence: An entity-based approach.
    Computational Linguistics, 34(1), 1-34.

2. Lapata, M., & Barzilay, R. (2005, July).
    Automatic evaluation of text coherence: Models and representations.
    In IJCAI (Vol. 5, pp. 1085-1090).

## Trivia

### What is the meaning of coheoka?

Coherence + Hyouka (means "evaluation" in Japanese. Kanji: 評価).

### Why it is so slow to download packages using conda?

Try using a mirror maintained by [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).
