import sys
import os
import re
import subprocess
import signal
import logging
from tempfile import TemporaryDirectory
import pickle

import numpy as np
import pandas as pd

from .model import BaseModel

ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def execute(command, flush_filter=None, terminal_condition=None):
    """Execute OS command with Popen."""
    process = subprocess.Popen(command,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        nextline = process.stdout.readline().decode('utf8')
        if len(nextline) == 0 and process.poll() is not None:
            break

        if flush_filter:
            try:
                if not flush_filter(nextline):
                    nextline = ''
            except Exception:
                nextline = ''
        if terminal_condition:
            try:
                if terminal_condition(nextline):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    break
            except Exception:
                pass

        sys.stdout.write(nextline)
        sys.stdout.flush()

    output = process.communicate()[0]
    exit_code = process.returncode

    if (exit_code == 0):
        return output
    else:
        raise RuntimeError(output)


class LIBMFConnecter:

    @staticmethod
    def save_matrix(df, path, user_col='userId', item_col='movieId', rating_col='y'):
        # indexize
        user_index_col = user_col + '_index'
        item_index_col = item_col + '_index'

        df_user_index = df[[user_col]].drop_duplicates()
        df_user_index[user_index_col] = np.array(list(range(df_user_index.shape[0])))

        df_movie_index = df[[item_col]].drop_duplicates()
        df_movie_index[item_index_col] = np.array(list(range(df_movie_index.shape[0])))

        df = pd.merge(df, df_user_index, on=user_col).drop(columns=[user_col])
        df = pd.merge(df, df_movie_index, on=item_col).drop(columns=[item_col])
        df = df[[user_index_col, item_index_col, rating_col]]

        # save matrix for libmf
        with open(path, 'w') as fw:
            for named_tuple in df.itertuples():
                fw.write(
                    '{} {} {}\n'.format(
                        getattr(named_tuple, user_index_col),
                        getattr(named_tuple, item_index_col),
                        getattr(named_tuple, rating_col),
                    )
                )

        return df, df_user_index, df_movie_index

    @staticmethod
    def save_matrix_with_indexer(df, path, user_indexer, item_indexer,
                                 user_col='userId', item_col='movieId', rating_col='y'):
        # indexize
        user_index_col = user_col + '_index'
        item_index_col = item_col + '_index'

        df = pd.merge(df, user_indexer, on=user_col).drop(columns=[user_col])
        df = pd.merge(df, item_indexer, on=item_col).drop(columns=[item_col])
        df = df[[user_index_col, item_index_col, rating_col]]
        df = df.dropna()

        # save matrix for libmf
        with open(path, 'w') as fw:
            for named_tuple in df.itertuples():
                fw.write(
                    '{} {} {}\n'.format(
                        getattr(named_tuple, user_index_col),
                        getattr(named_tuple, item_index_col),
                        getattr(named_tuple, rating_col),
                    )
                )

        return df

    @staticmethod
    def load_matrix(path, df_user_index, df_item_index,
                    user_col='userId', item_col='movieId', rating_col='y'):
        list_ = list()
        with open(path, 'r') as fr:
            for line in fr.readlines():
                line = line.strip()
                content = line.split(' ')
                list_.append((int(content[0]), int(content[1]), float(content[2])))

        # back by indexer
        user_index_col = user_col + '_index'
        item_index_col = item_col + '_index'
        df = pd.DataFrame(list_, columns=[user_index_col, item_index_col, rating_col])
        df = pd.merge(df, df_user_index, on=user_index_col)
        df = pd.merge(df, df_item_index, on=item_index_col)
        df = df[[user_col, item_col, rating_col]]
        return df

    @staticmethod
    def load_model(path, df_user_index, df_item_index,
                   user_col='userId', item_col='movieId'):
        user_vectors = dict()
        item_vectors = dict()
        with open(path, 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                content = line.strip().split(' ')
                if len(content) == 2:
                    if content[0] == 'f':
                        pass
                    elif content[0] == 'm':
                        user_dim = int(content[1])
                    elif content[0] == 'n':
                        item_dim = int(content[1])
                    elif content[0] == 'k':
                        dim = int(content[1])
                    elif content[0] == 'b':
                        global_b = float(content[1])
                else:
                    is_nan_vector = True if content[1] == 'F' else False
                    if is_nan_vector:
                        vector = np.full([dim], None, dtype='float64')
                    else:
                        vector = np.array(list(map(float, content[2:])))

                    target = content[0][0]
                    index = int(content[0][1:])
                    if target == 'p':
                        user_vectors[index] = vector
                    elif target == 'q':
                        item_vectors[index] = vector

        # back by indexer
        user_index_col = user_col + '_index'
        item_index_col = item_col + '_index'

        df_user_vector = pd.DataFrame(user_vectors.items(), columns=[user_index_col, 'vector'])
        df_user_vector = pd.merge(df_user_vector, df_user_index, on=user_index_col)
        df_user_vector = df_user_vector[[user_col, 'vector']]

        df_item_vector = pd.DataFrame(item_vectors.items(), columns=[item_index_col, 'vector'])
        df_item_vector = pd.merge(df_item_vector, df_item_index, on=item_index_col)
        df_item_vector = df_item_vector[[item_col, 'vector']]

        return (df_user_vector, df_item_vector, user_dim, item_dim, dim, global_b)

    @staticmethod
    def train(method, dim, epoch, lr, pth_train, pth_model, pth_log,
              pth_valid=None, l1=0.0, l2=0.0):
        if sys.platform == 'linux' or sys.platform == 'darwin':
            libmf_trainer = os.path.join(ROOT_DIR, 'libmf', 'mf-train')
            if not os.path.exists(libmf_trainer):
                cmd = ('cd {}; make -e;'
                       .format(os.path.join(ROOT_DIR, 'libmf')))
                execute(cmd)
        elif sys.platform == 'win32' or sys.platform == 'cygwin':
            libmf_trainer = os.path.join(ROOT_DIR, 'libmf', 'windows', 'mf-train.exe')

        if method == 'RVMF':
            loss_function = 0
        elif method == 'BMF':
            loss_function = 5
        elif method == 'OCMF':
            loss_function = 10

        if pth_valid:
            cmd = ('{trainer} -f {loss_function} -l1 {l1} -l2 {l2} -k {dim} -t {epoch} -r {lr} '
                   '-p {pth_valid_mat} {pth_train_mat} {model} | tee {log}'
                   .format(trainer=libmf_trainer,
                           loss_function=loss_function,
                           l1=l1,
                           l2=l2,
                           dim=dim,
                           epoch=epoch,
                           lr=lr,
                           pth_valid_mat=pth_valid,
                           pth_train_mat=pth_train,
                           model=pth_model,
                           log=pth_log,)
                   )
        else:
            cmd = ('{trainer} -f {loss_function} -l1 {l1} -l2 {l2} -k {dim} -t {epoch} '
                   '-r {lr} {pth_train_mat} {model} | tee {log}'
                   .format(trainer=libmf_trainer,
                           loss_function=loss_function,
                           l1=l1,
                           l2=l2,
                           dim=dim,
                           epoch=epoch,
                           lr=lr,
                           pth_train_mat=pth_train,
                           model=pth_model,
                           log=pth_log,)
                   )

        def flush_filter_closure():
            flush_list = list(range(0, epoch, 50)) + [epoch-1]

            def func(line):
                epoch_index = int(line.strip().split(' ')[0])
                return epoch_index in flush_list
            return func

        def terminal_condition(line):
            return ('nan' in line.split(' ')) or ('-nan' in line.split(' '))

        execute(cmd, flush_filter=flush_filter_closure(), terminal_condition=terminal_condition)

        train_err = None
        valid_err = None
        overfit_rate = None

        if pth_valid:
            with open(pth_log, 'r') as fr:
                line = fr.readlines()[-1].strip()
                list_ = re.split(r' +', line)
                train_err = float(list_[1])
                valid_err = float(list_[2])
                overfit_rate = (valid_err - train_err) / train_err if train_err > 0.0 else np.nan
        else:
            with open(pth_log, 'r') as fr:
                line = fr.readlines()[-1].strip()
                list_ = re.split(r' +', line)
                train_err = float(list_[1])

        return (train_err, valid_err, overfit_rate)


class RealValuedMatrixFactorization(BaseModel):
    def __init__(self):
        self._df_user_vector = None
        self._df_movie_vector = None
        self._global_b = None

    def _has_vaildation(self, valid_user_movie_pair, valid_y):
        return valid_user_movie_pair is not None and valid_y is not None

    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None,
            valid_user_movie_pair=None, valid_y=None,
            valid_user_feature=None, valid_movie_feature=None,
            dim=10, epoch=1000, lr=0.1, l1=0.0, l2=0.0):
        with TemporaryDirectory() as temp_dir:
            train_matrix_path = os.path.join(temp_dir, 'train_matrix.txt')
            valid_matrix_path = os.path.join(temp_dir, 'valid_matrix.txt')
            model_path = os.path.join(temp_dir, 'model.txt')
            log_path = os.path.join(temp_dir, 'log.txt')

            # prepare matrix for training
            logger.info('prepare matrix for training')

            y = np.reshape(y, (y.shape[0], 1))
            content = np.hstack((user_movie_pair, y))
            df = pd.DataFrame(
                content,
                columns=['userId', 'movieId', 'y'],
            )
            df.userId = df.userId.astype('int32')
            df.movieId = df.movieId.astype('int32')

            indexed_df, user_indexer, movie_indexer = \
                LIBMFConnecter.save_matrix(df, train_matrix_path)

            if self._has_vaildation(valid_user_movie_pair, valid_y):
                # prepare matrix for validation
                logger.info('prepare matrix for validation')

                valid_y = np.reshape(valid_y, (valid_y.shape[0], 1))
                valid_content = np.hstack((valid_user_movie_pair, valid_y))
                valid_df = pd.DataFrame(
                    valid_content,
                    columns=['userId', 'movieId', 'y'],
                )
                valid_df.userId = valid_df.userId.astype('int32')
                valid_df.movieId = valid_df.movieId.astype('int32')

                LIBMFConnecter.save_matrix_with_indexer(
                    valid_df, valid_matrix_path, user_indexer, movie_indexer)

            # fit RVMF
            logger.info('fit RVMF with dim={}, epoch={}, lr={}, l1={}, l2={}'
                        .format(dim, epoch, lr, l1, l2))

            LIBMFConnecter.train(method='RVMF', dim=dim, epoch=epoch, lr=lr,
                                 pth_train=train_matrix_path, pth_model=model_path,
                                 pth_log=log_path, pth_valid=valid_matrix_path, l1=l1, l2=l2)
            df_user_vector, df_movie_vector, user_dim, item_dim, dim, global_b = \
                LIBMFConnecter.load_model(model_path, user_indexer, movie_indexer)

        self._df_user_vector = df_user_vector
        self._df_movie_vector = df_movie_vector
        self._global_b = global_b
        return self

    def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
        df = pd.DataFrame(
            user_movie_pair,
            columns=['userId', 'movieId'],
        )
        df = pd.merge(df, self._df_user_vector, how='left', on=['userId']) \
               .rename(columns={'vector': 'user_vector'})
        df = pd.merge(df, self._df_movie_vector, how='left', on=['movieId']) \
               .rename(columns={'vector': 'movie_vector'})
        df = df.reset_index(drop=True)

        df['rating'] = self._global_b

        where_nan_row = df.isnull().any(axis=1)
        df_full = df.loc[~where_nan_row, :]
        user_vectors = np.array(df_full['user_vector'].values.tolist())
        movie_vectors = np.array(df_full['movie_vector'].values.tolist())
        df.loc[~where_nan_row, 'rating'] = np.sum(user_vectors * movie_vectors, axis=1)

        df['rating'] = df['rating'].fillna(self._global_b)

        return df['rating'].values

    @classmethod
    def load(cls, local_dir):
        instance = RealValuedMatrixFactorization()
        instance._df_user_vector = pd.read_pickle(str(local_dir / 'df_user_vector.pkl'))
        instance._df_movie_vector = pd.read_pickle(str(local_dir / 'df_movie_vector.pkl'))
        with open(str(local_dir / 'global_b.pkl'), 'rb') as input_file:
            instance._global_b = pickle.load(input_file)
        return instance

    def save(self, local_dir):
        self._df_user_vector.to_pickle(str(local_dir / 'df_user_vector.pkl'))
        self._df_movie_vector.to_pickle(str(local_dir / 'df_movie_vector.pkl'))
        with open(str(local_dir / 'global_b.pkl'), 'wb') as output_file:
            pickle.dump(self._global_b, output_file)
