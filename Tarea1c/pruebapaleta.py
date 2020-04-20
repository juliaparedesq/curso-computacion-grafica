import json

pallete1 = {}
pallete1['transparent'] = [0.5, 0.5, 0.5]
pallete1['pallete'] = [ [1, 0, 0], [1, 1, 1], [0, 0, 0] ]

with open('pallete1.json', 'w') as file:
    json.dump(pallete1, file)