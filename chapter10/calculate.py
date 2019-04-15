import csv
import time
from collections import defaultdict

from scipy import stats

from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


def plot_data(product_per_user_li):
    plot_data_all = Counter(product_per_user_li)
    plot_data_x = list(plot_data_all.keys())
    plot_data_y = list(plot_data_all.values())
    plt.xlabel('# of unique products')
    plt.ylabel('# of users')
    plt.scatter(plot_data_x, plot_data_y, marker='o')

    plt.show()


def show_silhouette_score(user_product_vec_li):
    test_data = np.array(user_product_vec_li)

    for k in range(2, 9):
        km = KMeans(n_clusters=k).fit(test_data)
        print(len(km.labels_))

        print("score for %d clusters:%.3f" % (k, silhouette_score(test_data, km.labels_)))


def plot_elbow(user_product_vec_li):
    test_data = np.array(user_product_vec_li)
    n_clusters = []
    inertias = []

    for k in range(1, 8):
        km = KMeans(n_clusters=k).fit(test_data)
        n_clusters.append(k)
        inertias.append(km.inertia_)

    plt.xlabel("# of clusters")
    plt.ylabel("within ss")
    plt.plot(n_clusters, inertias, linestyle="-", marker='o')
    plt.show()


def analyze_cluster_keywords(labels, product_id_name_dic, user_product_dic, id_user_dic):
    print(Counter(labels))
    cluster_item = defaultdict(list)

    for i in range(len(labels)):
        for x in user_product_dic[id_user_dic[i]]:
            cluster_item[labels[i]].append(product_id_name_dic[x])

    for cluster_id, product_name in cluster_item.items():
        bigram = []
        product_name_keyword = (' ').join(product_name).replace(' OF ', ' ').split()

        for i in range(0, len(product_name_keyword) -1):
            bigram.append(' '.join(product_name_keyword[i:i+2]))


        print('cluster_id:', cluster_id)
        print(Counter(bigram).most_common(20))


def analyze_clusters_product_count(labels, user_product_dic, id_user_dic):
    product_len_dic = defaultdict(list)

    for i in range(0, len(labels)):
        product_len_dic[labels[i]].append(len(user_product_dic[id_user_dic[i]]))

    for k, v in product_len_dic.items():
        print('cluster:', k)
        print(stats.describe(v))


with open('./Online_Retail_Large.csv') as f:
    f.readline().strip()

    reader = csv.reader(f, quotechar='"')
    # dict of user_id : set(product_id)
    user_product_dic = defaultdict(set) 

    # dict of product_id : set(user_id)
    product_user_dic = defaultdict(set)

    product_id_name_dic = {}

    counter = 0
    for row in reader:
        user_id = row[6]
        product_id = row[1]
        product_name = row[2]
        if len(user_id) == 0:
            continue

        country = row[7]
        if country != 'United Kingdom':
            continue

        try:
            invoice_year = time.strptime(row[4], '%m/%d/%Y %H:%M').tm_year
        except ValueError as e:
            print(e)
            continue

        if invoice_year != 2011:
            continue

        user_product_dic[user_id].add(product_id)
        product_user_dic[product_id].add(user_id)
        product_id_name_dic[product_id] = product_name


    product_per_user_li = [len(x) for x in user_product_dic.values()]

    print('# of users:', len(user_product_dic))
    print('# of products:', len(product_user_dic))

    print(stats.describe(product_per_user_li))

    #plot_data(product_per_user_li)

    min_product_user_li = [k for k, v, in user_product_dic.items() if len(v)==1]
    max_product_user_li = [k for k, v in user_product_dic.items() if len(v) > 600]

    print("# of users purchased one product: %d" % (len(min_product_user_li)))
    print("# of users purchased more than 600 product: %d" % (len(max_product_user_li)))

    user_product_dic = {k:v for k, v in user_product_dic.items() if len(v) > 1 and len(v) <= 600}
    print("# of left users: %d" % len(user_product_dic))

    # product의 id 에 0부터 시작하는 새로운 id를 부여한다.
    product_id_dic = {}
    id_ = 0

    for product_set in user_product_dic.values():
        for x in product_set:
            if x not in product_id_dic:
                product_id_dic.setdefault(x, id_)
                id_ += 1


    #print(product_id_dic)
    print("# of left items: %d" % (len(product_id_dic)))

    id_user_dic = {}
    user_product_vec_li = []
    num_of_products = len(product_id_dic)

    user_id = 0
    for user_code, product_set in user_product_dic.items():
        id_user_dic[user_id] = user_code

        vec = [0] * num_of_products
        for product in product_set:
            pid = product_id_dic[product]
            vec[pid] = 1

        user_id += 1

        user_product_vec_li.append(vec)


    train_data = user_product_vec_li[:2500]
    test_data = user_product_vec_li[2500:]
    print("# of train data: %d, # of test_data: %d" % (len(train_data), len(test_data)))

    km_predict = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=20).fit(train_data)
    km_predict_result = km_predict.predict(test_data)

    #show_silhouette_score(user_product_vec_li)

    #plot_elbow(user_product_vec_li)

    test_data = np.array(user_product_vec_li)
    km = KMeans(n_clusters=2, n_init=10, max_iter=20).fit(test_data)
    analyze_cluster_keywords(km.labels_, product_id_name_dic, user_product_dic, id_user_dic)
    analyze_clusters_product_count(km.labels_, user_product_dic, id_user_dic)
