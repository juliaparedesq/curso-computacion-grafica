import json

pallete4 = {}
pallete4['transparent'] = [0.5, 0.5, 0.5]
pallete4['pallete'] = [ [1, 0, 0], [1, 1, 0], [0, 0.6667, 0.8941], [1, 1, 1], [0, 0, 0] ]

with open('pallete4.json', 'w') as file:
    json.dump(pallete4, file)