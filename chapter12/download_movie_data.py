import codecs
import json

import numpy as np
from scipy import stats
import requests
from requests.exceptions import MissingSchema

from urllib.parse import urlparse, unquote

API_KEY = '19be134c'


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
    movie_info_li = read_data_with_primary_key('./data/u.item', '|')

    movie_data_li = []
    for movie_info in movie_info_li[:100]:
        # url 정보로부터 이름과 연도 정보를 가져온다.
        url = movie_info[3]
        parsed = urlparse(url)
        movie_title_year = unquote(parsed.query).replace('+', ' ')
        title_year_lst = movie_title_year.split(' ')
        title = ' '.join(title_year_lst[:-1])
        year = title_year_lst[-1].strip(')').strip('(')
        print(title, int(year))

        # omdbapi를 사용해 영화 정보를 가져온다.
        # 에러가 있거나 영화 정보가 없으면 title, year 만 json에 저장한다.
        try:
            url = 'http://www.omdbapi.com/?t=%s&y=%s&plot=full&apikey=%s' % (title, year, API_KEY)
            movie_response = requests.get(url)

        except MissingSchema:
            movie_json = {'Title': title, 'Year': year}
        else:
            movie_json = json.loads(movie_response.text)

        if movie_json.get('Title', None) is None:
            movie_json = {'Title': title, 'Year': year}

        movie_data_li.append(movie_json)

    storing_data = json.dumps(movie_data_li)
    with open('./data/movie_info.json', 'w') as f:
        f.writelines(storing_data)

