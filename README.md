# Recommender Touchstore on MovieLens 20M Dataset

![image](https://img.shields.io/badge/python-3.6-blue.svg)

## MovieLens 20M Dataset

source: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

GroupLens Research has collected and made available rating data sets from the MovieLens web site (http://movielens.org). The data sets were collected over various periods of time, depending on the size of the set.

This dataset (ml-20m) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users between January 09, 1995 and March 31, 2015. This dataset was generated on March 31, 2015, and updated on October 17, 2016 to update links.csv and add genome-* files.

## Prepare Environment

### virtual environment and dependencies

create virtual environment

```
$ python3.6 -m venv ./ENV
```

or

```
$ virtualenv --python=python3.6 ./ENV
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

## Prepare Public, Private, Question and Answer Data

```
$ make prepare
```

Splitting raw data into public and private dataset is according to the analysis result in `./prepare/ml-20m_analysis.ipynb`.

## Questions

Q1: Rating Problem: Design a system to predict an unknown rating when given `userId` and `movieId` (at `./src/data/test_q1.csv`). Evaluate results by RMSE.  
  
Q2: Ranking Problem: We defined that the rating > 3.0 as a favorite movie (at `./src/data/likes_pub.pkl`). Design a system to recommend a top-10 favorite movies for a person, a `movieId` list which `userId` did not see before (provide `userId` at `./src/data/test_q2.txt`). Evaluate results by MAP@10.  
  
Q3: Content-based Problem: We defined that the rating > 3.0 as a favorite movie (at `./src/data/likes_pub.pkl`) . Design a system to recommend a top-10 `userId` they may like a new movie (at `./src/data/test_q3.txt`). We will give you some information of that new movie (at `./src/data/ref_movie_feature.pkl`). Evaluate results by MAP@10.  

## Learning Algorithms

| Algorithm                        | For Question                               |
|----------------------------------|--------------------------------------------|
| Item Popularity Model            | Q1, Q2                                     |
| Item Cosine Similarity           | Q1 (mean centering), Q2, Q3                |
| Real Valued Matrix Factorization | Q1, Q2 (negative data), Q3 (negative data) |
