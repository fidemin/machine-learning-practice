import codecs

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
