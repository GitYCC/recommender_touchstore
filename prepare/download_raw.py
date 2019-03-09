import os
import zipfile
import hashlib

from tqdm import tqdm
from six.moves.urllib.request import urlretrieve

WORKSPACE = os.path.dirname(__file__)
URL_MovieLens20M = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
ZIP_MovieLens20M = os.path.join(WORKSPACE, 'ml-20m.zip')
MD5_ZIP_MovieLens20M = 'cd245b17a1ae2cc31bb14903e1204af3'
progressbar = [None]


def _show_progress(count, block_size, total_size):
    if progressbar[0] is None:
        progressbar[0] = tqdm(total=total_size)

    downloaded = block_size * count
    if downloaded <= total_size:
        progressbar[0].update(block_size)
    else:
        progressbar[0].close()
        progressbar[0] = None


def maybe_download_movielens20m():
    if not os.path.exists(ZIP_MovieLens20M):
        url = URL_MovieLens20M
        print('download from {}'.format(url))
        filename, _ = urlretrieve(url, ZIP_MovieLens20M, _show_progress)


def check_raw_checksum():
    hash_md5 = hashlib.md5()
    with open(ZIP_MovieLens20M, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    if hash_md5.hexdigest() != MD5_ZIP_MovieLens20M:
        raise RuntimeError('checksum is not matching')


def unzip():
    with zipfile.ZipFile(ZIP_MovieLens20M, 'r') as zip_ref:
        zip_ref.extractall(WORKSPACE)


def main():
    maybe_download_movielens20m()
    check_raw_checksum()
    unzip()


if __name__ == '__main__':
    main()
