## Fake News Detector

***
### Problem
***

The problem is not only hackers, going into accounts, and sending false information. The bigger problem here is what we call "Fake News". A fake are those news stories that are false: the story itself is fabricated, with no verifiable facts, sources, or quotes.

When someone (or something like a bot) impersonates someone or a reliable source to false spread information, that can also be considered as fake news. In most cases, the people creating this false information have an agenda, that can be political, economical or to change the behavior or thought about a topic.

There are countless sources of fake news nowadays, mostly coming from programmed bots, that can't get tired (they're machines hehe) and continue. to spread false information 24/7.

Serious studies in the past 5 years, have demonstrated big correlations between the spread of false information and elections, the popular opinion or feelings about different topics.

The problem is real and hard to solve because the bots are getting better are tricking us. Is not simple to detect when the information is true or not all the time, so we need better systems that help us understand the patterns of fake news to improve our social media, communication and to prevent confusion in the world.

### Purpose
***
This application provide a tool which can detect fake news according to the statement and some metadata of the given news. The accuracy of the prediction will higher than random prediction.

### Data
***

[Liar Dataset](https://paperswithcode.com/dataset/liar)

LIAR is a publicly available dataset for fake news detection. A decade-long of 12.8K manually labeled short statements were collected in various contexts from POLITIFACT.COM, which provides detailed analysis report and links to source documents for each case. This dataset can be used for fact-checking research as well. Notably, this new dataset is an order of magnitude larger than previously largest public fake news datasets of similar type. The LIAR dataset4 includes 12.8K human labeled short statements from POLITIFACT.COM’s API, and each statement is evaluated by a POLITIFACT.COM editor for its truthfulness.

### Model
***

Bi-LSTM with statement, dependency parse and metadata inputs.

### Get Started
***
#### Installation
***
1. Install the dependencies
```
pip install requirements.txt
```
2. Download the pre-trained model GloVe

Linux:
```
$ wget http://nlp.stanford.edu/data/glove.6B.zip
$ unzip glove*.zip
```

Download in link:
https://nlp.stanford.edu/data/glove.twitter.27B.zip


#### Graphical User Interface
***
```
python gui.py
```
Enter statement,and some information about the statement, click "Check", it will output the prediction.

#### Web Application
***
```
python web.py
```
#### Command Line Interface
***
For get help
```
python cli.py --help
```
Input Format:
```
python cli.py <statement> --subject <subject> --speaker <speaker> --job <job> --state <state> --party <party> --venue <venue>
```

Output Format:
```
[<prediction>, <correct probability>]
```

Example:
```
python cli.py  "Since 1968, more Americans have died from gunfire than died in all the wars of this countrys history." --subject guns --speaker mark-shields --job Columnist --state 'Washington, D.C.' --venue 'the PBS NewsHour'

['true', 0.88939434]
```

We advise you choose metadata from the list in [para_list.py](para_list.py) 