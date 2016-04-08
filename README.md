# coheoka
Coherence evaluation using Stanford's CoreNLP tool.

This repository is designed for entity-base coherence.

## Install
You must run a CoreNLP server on your own if you want to run any module in this repository.

You can download Stanford CoreNLP latest version (3.6.0) at [here](http://stanfordnlp.github.io/CoreNLP/download.html) and run a local server (requiring Java 1.8) by this way:

```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```

Then you can find a demo at [`localhost:9000`](http://localhost:9000/), which visualizes StanfordCoreNLP's sophisticated annotation for English documents.

You can also find an online demo maintained by Stanford at [here](http://corenlp.run/).

Then it is necessary to install a CoreNLP's Python wrapper if you want to communicate with the server, or you can write a wrapper by yourself after reading CoreNLP's documentation. Also, make sure you have installed any Python's scientific distribution such as [Anaconda](https://www.continuum.io/downloads) which I strongly recommend. Trust me, you will love it.

This repository does **not** need to install a Python wrapper.

## Trivia
### What is the meaning of coheoka?
Coherence + Hyouka (means "evaluation" in Japanese. Kanji: 評価).
