import json

with open('./data/movie_info.json') as f:
    data = f.readline()
    json_data = json.loads(data)

    for row in json_data:
        print(row.get('Plot', None))
