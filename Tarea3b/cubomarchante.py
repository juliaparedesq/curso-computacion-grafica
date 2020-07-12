from scipy.sparse import csr_matrix
import numpy as np
import basic_shapes as bs
import easy_shaders as es

def createColorCube(i, j, k, X, Y, Z):
    l_x = X[i]
    r_x = X[i+1]
    b_y = Y[j]
    f_y = Y[j+1]
    b_z = Z[0]
    t_z = Z[1]
    c = np.random.rand
    #   positions    colors
    vertices = [
        # Z+: number 1
        l_x, b_y, t_z, c(), c(), c(),
        r_x, b_y, t_z, c(), c(), c(),
        r_x, f_y, t_z, c(), c(), c(),
        l_x, f_y, t_z, c(), c(), c(),
        # Z-: number 6
        l_x, b_y, b_z, 0, 0, 0,
        r_x, b_y, b_z, 1, 1, 1,
        r_x, f_y, b_z, 0, 0, 0,
        l_x, f_y, b_z, 1, 1, 1,
        # X+: number 5
        r_x, b_y, b_z, 0, 0, 0,
        r_x, f_y, b_z, 1, 1, 1,
        r_x, f_y, t_z, 0, 0, 0,
        r_x, b_y, t_z, 1, 1, 1,
        # X-: number 2
        l_x, b_y, b_z, 0, 0, 0,
        l_x, f_y, b_z, 1, 1, 1,
        l_x, f_y, t_z, 0, 0, 0,
        l_x, b_y, t_z, 1, 1, 1,
        # Y+: number 4
        l_x, f_y, b_z, 0, 0, 0,
        r_x, f_y, b_z, 1, 1, 1,
        r_x, f_y, t_z, 0, 0, 0,
        l_x, f_y, t_z, 1, 1, 1,
        # Y-: number 3
        l_x, b_y, b_z, 0, 0, 0,
        r_x, b_y, b_z, 1, 1, 1,
        r_x, b_y, t_z, 0, 0, 0,
        l_x, b_y, t_z, 1, 1, 1,
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        4, 5, 1, 1, 0, 4,
        6, 7, 3, 3, 2, 6,
        5, 6, 2, 2, 1, 5,
        7, 4, 0, 0, 3, 7]

    return bs.Shape(vertices, indices)

def my_marching_cube(suelo, isosurface):
    sh=suelo.shape[0]-1, suelo.shape[1]-1
    voxels = np.zeros(shape=sh, dtype=bool)
    for i in range(1, suelo.shape[0]-1):
        for j in range(1, suelo.shape[1]-1):
            v_min = suelo[i-1:i+2, j-1:j+1].min()
            v_max= suelo[i-1:i+2, j-1:j+1].max()

            if v_min <= isosurface and isosurface<= v_max:
                voxels[i,j]= True

            else:
                voxels[i,j] = False
    return voxels

def merge(destinationShape, strideSize, sourceShape):
    # current vertices are an offset for indices refering to vertices of the new shape
    offset = len(destinationShape.vertices)
    destinationShape.vertices += sourceShape.vertices
    destinationShape.indices += [(offset / strideSize) + index for index in sourceShape.indices]



# coding=utf-8
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys

import transformations as tr
import easy_shaders as es
import basic_shapes as bs

PROJECTION_ORTHOGRAPHIC = 0
PROJECTION_FRUSTUM = 1
PROJECTION_PERSPECTIVE = 2


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.projection = PROJECTION_ORTHOGRAPHIC


# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return

    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_1:
        print('Orthographic projection')
        controller.projection = PROJECTION_ORTHOGRAPHIC

    elif key == glfw.KEY_2:
        print('Frustum projection')
        controller.projection = PROJECTION_FRUSTUM

    elif key == glfw.KEY_3:
        print('Perspective projection')
        controller.projection = PROJECTION_PERSPECTIVE

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Projections Demo", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program
    pipeline = es.SimpleModelViewProjectionShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(pipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.15, 0.15, 0.15, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))

    # Load potentials and grid
    isosurface = bs.Shape([], [])
    # Now let's draw voxels!
    u = np.zeros((8, 8))
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if i == j:
                u[1, 1] = 2
    k = 1
    my = my_marching_cube(u, 2)  # hago matriz de voxeles con isosuperficie 2
    # my = np.zeros((7, 7), dtype=bool)
    # my[2, 2] = True
    # print(my)

    for i in range(my.shape[0]):
        for j in range(my.shape[1]):
            # print(X[i,j,k])
            if my[i, j]:  # si es True
                # print(i, j)
                temp_shape = createColorCube(i, j, 5, range(u.shape[0]), range(u.shape[1]), [k, k + 1])
                merge(destinationShape=isosurface, strideSize=6, sourceShape=temp_shape)

    gpu_surface = es.toGPUShape(isosurface)
    isosurface = bs.Shape([], [])
    # Now let's draw voxels!
    suelo=np.load('solution.npy')

    k = 0
    my = my_marching_cube(suelo, 22)  # hago matriz de voxeles con isosuperficie 22
    #my = np.zeros((7, 7), dtype=bool)
    #my[2, 2] = True
    print(csr_matrix(my))

    for i in range(my.shape[0]):
        for j in range(my.shape[1]):
            # print(X[i,j,k])
            if my[i, j]:  # si es True
                # print(i, j)
                temp_shape = createColorCube(i, j, 5, range(suelo.shape[0]), range(suelo.shape[1]), [k, k + 1])
                merge(destinationShape=isosurface, strideSize=6, sourceShape=temp_shape)

    gpu_surface = es.toGPUShape(isosurface)

    t0 = glfw.get_time()
    camera_theta = np.pi / 4

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()



        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta -= 2 * dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta += 2 * dt

        # Setting up the view transform

        camX = 10 * np.sin(camera_theta)+1
        camY = 10 * np.cos(camera_theta)+1

        viewPos = np.array([0, 0, 0])

        view = tr.lookAt(
            viewPos,
            np.array([camX, camY, 3]),
            np.array([0, 0, 1])
        )

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)

        # Setting up the projection transform

        if controller.projection == PROJECTION_ORTHOGRAPHIC:
            projection = tr.ortho(-8, 8, -8, 8, 0.1, 100)

        elif controller.projection == PROJECTION_FRUSTUM:
            projection = tr.frustum(-5, 5, -5, 5, 9, 100)

        elif controller.projection == PROJECTION_PERSPECTIVE:
            projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        else:
            raise Exception()

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Drawing shapes with different model transformations
        transform=np.matmul(tr.translate(-100, -30,0) ,tr.uniformScale(1))
        #glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.translate(5, 0, 0))

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, transform)

        pipeline.drawShape(gpu_surface)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        pipeline.drawShape(gpuAxis, GL_LINES)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()