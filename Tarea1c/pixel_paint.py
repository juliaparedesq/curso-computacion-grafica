import json
import glfw
import OpenGL

with open('pallete1.json') as file:
    paleta = json.load(file)
    colortransparente=paleta['transparent']
    otroscolores=paleta['pallete']

N=16 #numero de pixeles

