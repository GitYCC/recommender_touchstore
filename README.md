# Recommender Touchstore on MovieLens 20M Dataset

![image](https://img.shields.io/badge/python-3.6-blue.svg)

## MovieLens 20M Dataset

source: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

GroupLens Research has collected and made available rating data sets from the MovieLens web site (http://movielens.org). The data sets were collected over various periods of time, depending on the size of the set.

This dataset (ml-20m) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users between January 09, 1995 and March 31, 2015. This dataset was generated on March 31, 2015, and updated on October 17, 2016 to update links.csv and add genome-* files.

## Prepare

### virtual environment and dependencies

#### recommended method: pyenv + pipenv

use `pyenv` to local python version to this project,

```
$ pyenv install 3.6.5
$ pyenv local
```

use `pipenv` to set up dependencies,

```
$ pipenv --python 3.6.5
$ pipenv install -r requirements.txt
```

enter virtual environment

```
$ pipenv shell
```

if you want to exit virual environment,

```
$ deactivate
```

#### simple method
create virtual environment

```
$ python3.6 -m venv ./ENV
```

enter virtual environment

```
$ source ./ENV/bin/activate
```

if you want to exit virual environment,

```
$ deactivate
```

install dependencies under virtual environment

```
$ pip3.6 install -r requirements.txt
```

## Prepare Public Dataset

```
$ make prepare
```

Splitting raw data into public and private dataset is according to the analysis result in `./prepare/ml-20m_analysis.ipynb`.

