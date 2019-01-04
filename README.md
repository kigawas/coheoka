# coheoka

Python coherence evaluation tool using Stanford's CoreNLP.

This repository is designed for entity-base coherence.

## Prerequisite

It is highly recommended to run a CoreNLP server on your own if you want to test coherence in this repository.

You can download Stanford CoreNLP latest version (3.9.2) at [here](http://stanfordnlp.github.io/CoreNLP/download.html) and run a local server (requiring Java 1.8+) by this way:

```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```

Then there comes a demo at [`localhost:9000`](http://localhost:9000/), which visualizes StanfordCoreNLP's sophisticated annotations for English documents.

Also, there is an online demo maintained by Stanford at [here](http://corenlp.run/).

If you need to annotate lots of documents, you **must** set up a local server on your own. Or if you just want to test a few documents without downloading the CoreNLP tool, you may set an environment variable `CORENLP_URL` to use an existing server (e.g. `http://corenlp.run/` and don't forget the **`http`**).

Also, if you are using Windows (actually, it is recommended to install pre-built binaries instead of building them by yourself whatever OS you choose), make sure you have installed any Python's scientific distribution such as [Anaconda](https://www.continuum.io/downloads) (if you want many scientific packages for future use) or [Miniconda](http://conda.pydata.org/miniconda.html) (if you don't want to spend too much disk space) which I strongly recommend.

## Install

The requirements are `nltk`, `numpy`, `pandas`, `requests`, `scipy` and `scikit-learn`.

If you have installed Anaconda or Miniconda just
```
conda create -n coheoka --file requirements.txt
```
and activate it by typing `activate coheoka` on Windows or `source activate coheoka` on Linux.

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

### Why so slow to download packages from conda?

Try using a mirror maintained by [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).
