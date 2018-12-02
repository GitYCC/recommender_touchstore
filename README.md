# Recommender Touchstore on MovieLens 20M Dataset

![image](https://img.shields.io/badge/python-3.6-blue.svg)

## Prepare

### virtual environment and dependencies

#### method: pyenv + pipenv

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

