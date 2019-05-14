# cs247 Course Project

## Emerging Relation Detection from News in Heterogeneous Information Networks (HIN)

### Project Proposal

Proposal is [here](https://github.com/jiaowoshabi/cs247/blob/master/docs/submission_17474255.pdf)

### Datasets

1. [All the News](https://www.kaggle.com/snapcrack/all-the-news)
2. [BBC News](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip)

### Papers

1. [HEER:Heterogeneous Graph Embedding for Emerging Relation Detection from News](https://github.com/jiaowoshabi/cs247/blob/master/docs/Zhang%20et%20al.%20-%202016%20-%20Heer%20Heterogeneous%20graph%20embedding%20for%20emerging%20r.pdf)
2. [Mining heterogeneous information networks princip](https://github.com/jiaowoshabi/cs247/blob/master/docs/Sun%20and%20Han%20-%202012%20-%20Mining%20heterogeneous%20information%20networks%20princip.pdf)
...

### Project structure
    .
    ├── data                    # Datasets 
    │   └── bbc
    ├── docs                    # Documentation files, reports and reference papers
    ├── src                     # Source code files
    │   ├── reader.py
    │   └── tests               # Automated tests
    │        └── test_reader.py
    ├── requirements.txt        # Dependencies
    ├── Makefile
    ├── LICENSE
    └── README.md

### Development

**Note: this project uses Python (>= 3.6). Please make sure you have the updated Python**

**It is recommended to use [virtualenv](https://virtualenv.pypa.io/en/latest/) to set up Python environments.**

To begin with, install all the dependencies using the following command at project's top-level (e.g. `cs247/`).

```
make requirements
```

To run the unittest suites, use

```
make test
```

If you'd like to run any module (i.e. `.py`) directly, do `python3 -m path_to.module_name`. For example, to run `test_reader.py` under `src/tests` directory, 

```
python3 -m src.tests.test_reader
```

### To-dos

- [week 5&6](https://cs247project.slack.com/files/UHGR5HAP4/FJD2JBKTJ/Recap_of_the_kick-off_meeting_)