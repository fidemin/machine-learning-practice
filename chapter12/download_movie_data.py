import json
import re

import numpy as np
from scipy import stats
import requests
from requests.exceptions import MissingSchema


from urllib.parse import urlparse, unquote

from common import read_data_with_primary_key

API_KEY = '19be134c'

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
        title = re.sub(r', The','', title)
        year = title_year_lst[-1].strip(')').strip('(')
        print(movie_title_year, title, int(year))

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
            print('----- No match data: %s -----' % title)
            movie_json = {'Title': title, 'Year': year}

        movie_data_li.append(movie_json)

    storing_data = json.dumps(movie_data_li)
    with open('./data/movie_info.json', 'w') as f:
        f.writelines(storing_data)

