# coding=utf-8
"""
Daniel Calderon, CC3501, 2019-2
plotting a 2d function as a surface
"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys

import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import ex_triangle_mesh
from datetime import datetime
import Auxiliar7


MIN_VALUE = 12
MAX_VALUE = 15

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True


# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Isoline", None, None)

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
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))

    # TODO!: Revisar performance variando N
    # como la implementación es recursiva, podemos superrar el limite establecido por python,
    # podemos modificar este limite dentro de ciertos rangos conservadores.
    print("Recursion Limit: ", sys.getrecursionlimit())
    sys.setrecursionlimit(3000)
    print("Recursion Limit: ", sys.getrecursionlimit())
    print()
    paraboloid_mesh, mesh_triangles, mesh_vertices = Auxiliar7.create_paraboloid_mesh(100)

    triangle_mesh_with_value = Auxiliar7.find_triangle_with_range(paraboloid_mesh, mesh_vertices, MIN_VALUE, MAX_VALUE)
    #triangle_mesh_with_value = Auxiliar7.find_triangle_with_value(paraboloid_mesh, mesh_vertices, MAX_VALUE)

    init_time = datetime.now()
    filtered_triangles = Auxiliar7.filter_triangles_one_by_one_range(paraboloid_mesh, mesh_vertices, MIN_VALUE ,MAX_VALUE)
    #filtered_triangles = Auxiliar7.filter_triangles_one_by_one_value(paraboloid_mesh, mesh_vertices, MAX_VALUE)
    print("Cantidad de triangulos totales: {}".format(mesh_triangles.__len__()))
    print("Tiempo de demora en filtrado (todos): {}".format(datetime.now() - init_time))
    print()

    init_time = datetime.now()
    #filtered_triangles, cantidad = Auxiliar7.mesh_filter_triangles_range(paraboloid_mesh, mesh_vertices, MIN_VALUE, MAX_VALUE, triangle_mesh_with_value, [triangle_mesh_with_value.data])
    #filtered_triangles, cantidad = Auxiliar7.mesh_filter_triangles_value(paraboloid_mesh, mesh_vertices, MAX_VALUE, triangle_mesh_with_value, [triangle_mesh_with_value.data])
    #print("Cantidad de triangulos visitados: {}".format(cantidad))
    #print("Tiempo de demora en filtrado (vecinos): {}".format(datetime.now() - init_time))

    filtered_mesh = Auxiliar7.create_mesh(filtered_triangles)

    cpuSurface = Auxiliar7.draw_mesh(filtered_mesh, mesh_vertices)
    gpuSurface = es.toGPUShape(cpuSurface)

    t0 = glfw.get_time()
    camera_theta = np.pi/4

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
            camera_theta += 2* dt

        # Setting up the view transform

        camX = 10 * np.sin(camera_theta)
        camY = 10 * np.cos(camera_theta)

        viewPos = np.array([camX, camY, 10])

        view = tr.lookAt(
            viewPos,
            np.array([0,0,5]),
            np.array([0,0,1])
        )

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)

        # Setting up the projection transform
        projection = tr.perspective(60, float(width)/float(height), 0.1, 100)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        pipeline.drawShape(gpuAxis, GL_LINES)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Drawing shapes with different model transformations
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.uniformScale(0.5))
        pipeline.drawShape(gpuSurface)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()