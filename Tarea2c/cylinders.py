import glfw
from OpenGL.GL import *
import numpy as np
import sys

import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as light_s

#INDEXION COLOR TRIANGLE
def createColorTriangleIndexation(start_index, a, b, c, color):
    # Defining locations and colors for each vertex of the shape
    v1 = np.array([a_v - b_v for a_v, b_v in zip(a, b)])
    v2 = np.array([b_v - c_v for b_v, c_v in zip(b, c)])
    v1xv2 = np.cross(v1, v2)
    vertices = [
    # positions colors
        a[0], a[1], a[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2]
    ]
    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        start_index, start_index+1, start_index+2,
        start_index+2, start_index+3, start_index
        ]
    return (vertices, indices)

#INDEXION TEXTURE TRIANGLE
def createTextureTriangleIndexation(start_index, a, b, c, t, dt):
    print(a)
    # Defining locations and colors for each vertex of the shape
    v1 = np.array([a_v - b_v for a_v, b_v in zip(a, b)])
    v2 = np.array([b_v - c_v for b_v, c_v in zip(b, c)])
    v1xv2 = np.cross(v1, v2)
    vertices = [
    # positions colors
        a[0], a[1], a[2], 0.5, 0.5, v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2], 0.5*np.cos(t)+0.5, 0.5*np.sin(t)+0.5, v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2], 0.5*np.cos(dt)+0.5, 0.5*np.sin(dt)+0.5 , v1xv2[0], v1xv2[1], v1xv2[2]
    ]
    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        start_index, start_index+1, start_index+2,
        start_index+2, start_index+3, start_index
        ]
    return (vertices, indices)

#INDEXION COLOR QUAD
def createColorQuadIndexation(start_index, a, b, c, d, color):
    # Defining locations and colors for each vertex of the shape
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)
    vertices = [
    # positions colors
        a[0], a[1], a[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2],
        d[0], d[1], d[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2]
    ]
    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        start_index, start_index+1, start_index+2,
        start_index+2, start_index+3, start_index
        ]
    return (vertices, indices)


#INDEXION TEXTURE QUAD
def createTextureQuadIndexation(start_index, a, b, c, d, inicio, final):
    # Defining locations and colors for each vertex of the shape
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)
    vertices = [
    # positions colors
        a[0], a[1], a[2], inicio, 1, v1xv2[0], v1xv2[1], v1xv2[2],  #1
        b[0], b[1], b[2], final, 1, v1xv2[0], v1xv2[1], v1xv2[2],   #1   les puse +1.1 (en el eje y) para así poder elegir como se vea el borde, con gl_repeat o clamp to edge
        c[0], c[1], c[2], final, 0, v1xv2[0], v1xv2[1], v1xv2[2],   #0
        d[0], d[1], d[2], inicio, 0, v1xv2[0], v1xv2[1], v1xv2[2]   #0
    ]
    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        start_index, start_index+1, start_index+2,
        start_index+2, start_index+3, start_index
        ]
    return (vertices, indices)


#CYLINDER TEXTURE
def generateTextureCylinder(latitudes, image, R = 1.0, z_bottom=0.0, z_top = 1.0):
    vertices = []
    indices = []
    # Angle step
    dtheta = 2 * np.pi / latitudes
    theta = 0
    start_index = 0
    lat=1/latitudes
    lat0=0
    # We generate a rectangle for every latitude, lados del cilindro
    for _ in range(latitudes):
        a= np.array([R*np.cos(theta), R*np.sin(theta), z_bottom])
        b = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_bottom])
        c = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_top])
        d = np.array([R * np.cos(theta), R * np.sin(theta), z_top])
        vert, ind = createTextureQuadIndexation(start_index, a, b, c, d, lat0, lat)
        vertices += vert
        indices += ind
        lat0 = lat
        lat= 1/latitudes + lat
        start_index += 4
        theta = theta + dtheta

    """#tapa del cilindro
    dtheta = 2 * np.pi / latitudes
    theta = 0

    for i in range(latitudes):
        e = np.array([R * np.cos(theta), R * np.sin(theta), z_top])
        f = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_top])
        g= np.array([0, 0, z_top])
        vv, ii = createTextureTriangleIndexation(start_index, g, e, f, theta, theta + dtheta)
        vertices += vv
        indices += ii
        start_index += 3
        theta = theta + dtheta"""

    return bs.Shape(vertices, indices, image)


#CYLINDER COLOR
def generateCylinder(latitudes, color, R = 1.0, z_bottom=0.0, z_top = 1.0):
    vertices = []
    indices = []
    # Angle step
    dtheta = 2 * np.pi / latitudes
    theta = 0
    start_index = 0
    # We generate a rectangle for every latitude, lados del cilindro
    for _ in range(latitudes):
        a= np.array([R*np.cos(theta), R*np.sin(theta), z_bottom])
        b = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_bottom])
        c = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_top])
        d = np.array([R * np.cos(theta), R * np.sin(theta), z_top])
        vert, ind = createColorQuadIndexation(start_index, a, b, c, d, color)
        vertices += vert
        indices += ind
        start_index += 4
        theta = theta + dtheta

    #tapa del cilindro
    dtheta = 2 * np.pi / latitudes
    theta = 0
    for i in range(latitudes):
        e = np.array([R * np.cos(theta), R * np.sin(theta), z_top])
        f = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_top])
        g= np.array([0, 0, z_top])
        vv, ii = createColorTriangleIndexation(start_index, g, f, e, color)
        vertices += vv
        indices += ii
        start_index += 3
        theta = theta + dtheta

    return bs.Shape(vertices, indices)

generateTextureCylinder(20, 'tallo.jpg', 1.0, 0.0, 5.0)