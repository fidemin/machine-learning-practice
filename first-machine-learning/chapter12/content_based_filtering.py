import codecs
import json

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import read_data_with_primary_key


def similar_recommend_by_movie_id(X, movie_sim, movie_info_li, movielens_id):
    movie_idx = movielens_id - 1
    # 원본의 idx를 기억하기 위하여, enumerate를 쓴다. 
    # [(리스트 익덱스 0, 유사도0), ...] 형태의 데이터가 들어있다.
    similar_movies = sorted(list(enumerate(movie_sim[movie_idx])), key=lambda x:x[1], reverse=True)
    recommended = 1
    print("------ recommendation for movie %d ------" % movielens_id)

    for movie_info in similar_movies[1:6]:
        # 자기 자신은 제외
        movie_title = movie_info_li[movie_info[0]]
        print('rank %d recommendation: %s' % (recommended, movie_title[0]))
        recommended += 1


class LemmaTokenizer(object):
    def __init__(self):
        self.tokenizer = RegexpTokenizer('(?u)\w\w+')
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tokenizer.tokenize(doc)]


if __name__ == "__main__":
    movie_info_li = read_data_with_primary_key('./data/u.item', '|')

    movie_title_li = []
    movie_plot_li = []

    with open('./data/movie_info.json') as f:
        data = f.readline()
        json_data = json.loads(data)

        for row in json_data:
            movie_title = row.get('Title', '')
            movie_plot = row.get('Plot', '')

            movie_title_li.append(movie_title)
            movie_plot_li.append(movie_plot)

    vectorizer = TfidfVectorizer(min_df=3, tokenizer=LemmaTokenizer(), stop_words='english')
    X = vectorizer.fit_transform(movie_plot_li)
    #feature_names = vectorizer.get_feature_names()
    #print(feature_names)
    movie_sim = cosine_similarity(X)
    similar_recommend_by_movie_id(X, movie_sim, movie_info_li, 1)
