import json

pallete3 = {}
pallete3['transparent'] = [0.2, 0.5, 0.1]
pallete3['pallete'] = [ [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 1, 0.5], [1, 0.5, 0.5], [0.5, 0.5, 0.5], [0.66, 0.2, 0.2], [1, 0.6, 0.6], [0.05, 0.3, 0.7] ]

with open('pallete3.json', 'w') as file:
    json.dump(pallete3, file)