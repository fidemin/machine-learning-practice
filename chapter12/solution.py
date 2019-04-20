import codecs

import numpy as np
from scipy import stats
import requests
from requests.exceptions import MissingSchema

def read_data_with_primary_key(fname, delim):
    li = []

    count = 1
    with codecs.open(fname, 'r', encoding='latin-1') as f:
        for line in f:
            row = line.strip().split(delim)
            pkey = int(row[0])

            if count != pkey:
                print('errors at data_id')
            count += 1

            li.append(row[1:])

    print('rows in %s: %d' % (fname, len(li)))
    return li


if __name__ == "__main__":
    user_info_li = read_data_with_primary_key('./data/u.user', '|')
    movie_info_li = read_data_with_primary_key('./data/u.item', '|')

    # 무비렌즈 별점 정보 파일을 이용하여 유틸리티 행렬 만들기
    R = np.zeros((len(user_info_li), len(movie_info_li)), dtype=np.float64)

    for line in codecs.open('./data/u.data', 'r', encoding='latin-1'):
        user, movie, rating, date = line.strip().split('\t')
        user_idx = int(user) - 1
        movie_idx = int(movie) - 1
        # R[x, y] is same as R[(x, y)]
        R[user_idx, movie_idx] = float(rating)

    #print(R[0, 10])

