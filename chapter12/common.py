import codecs

def read_data_with_primary_key(fname, delim):
    li = []

    count = 1
    with codecs.open(fname, 'r', encoding='latin-1') as f:
        for line in f:
            row = line.strip().split(delim)
            # 데이터의 첫번째 값은 primary key index이다.
            pkey = int(row[0])

            # pkey는 1로 시작한다.
            if count != pkey:
                print('errors at data_id')
            count += 1

            li.append(row[1:])

    print('rows in %s: %d' % (fname, len(li)))
    return li
