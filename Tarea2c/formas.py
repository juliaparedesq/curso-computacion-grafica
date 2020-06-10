import basic_shapes as bs
import numpy as np

def createTextureTriangleIndexation(start_index, a, b, c, t, dt):
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



def generateTextureCerro(latitudes, image, R = 1.0, z_bottom=0.0, z_top = 1.0):
    vertices = []
    indices = []
    # Angle step
    dtheta = 2 * np.pi / latitudes
    theta = 0
    start_index = 0
    lat=1/latitudes
    lat0=0

    #tapa del cilindro
    dtheta = 2 * np.pi / latitudes
    theta = 0

    for i in range(latitudes):
        e = np.array([R * np.cos(theta), R * np.sin(theta), z_bottom])
        f = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_bottom])
        g= np.array([0, 0, z_top])
        vv, ii = createTextureTriangleIndexation(start_index, g, e, f, theta, theta + dtheta)
        vertices += vv
        indices += ii
        start_index += 3
        theta = theta + dtheta

    return bs.Shape(vertices, indices, image)