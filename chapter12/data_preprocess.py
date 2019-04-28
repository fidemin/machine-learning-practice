import codecs

import numpy as np
from scipy import stats
import requests
from requests.exceptions import MissingSchema

from common import read_data_with_primary_key

def data_preprocessing():
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

    return user_info_li, movie_info_li, R


user_info_li, movie_info_li, R = data_preprocessing()


if __name__ == "__main__":
    user_mean_li = []
    for i in range(0, R.shape[0]):
        user_rating = [x for x in R[i] if x > 0.0]
        user_mean_li.append(stats.describe(user_rating).mean)

    print("user mean stats:", stats.describe(user_mean_li))

    movie_mean_li = []

    R_T = R.T
    for i in range(0, R.shape[1]):
        movie_rating = [x for x in R_T[i] if x>0.0]
        movie_mean_li.append(stats.describe(movie_rating).mean)

    print("movie_mean_stats:", stats.describe(movie_mean_li))


