import json

pallete2 = {}
pallete2['transparent'] = [0.5, 0.5, 0.5]
pallete2['pallete'] = [ [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 1, 0.5], [1, 0.5, 0.5], [0.2, 0.5, 0.1], [0.66, 0.2, 0.2], [1, 0.6, 0.6], [0.05, 0.3, 0.7] ]

with open('pallete2.json', 'w') as file:
    json.dump(pallete2, file)