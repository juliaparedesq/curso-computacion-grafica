# coding=utf-8
"""
Daniel Calderon, CC3501, 2019-2
Drawing 3D cars via scene graph
"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import hotel_viewer as hv
import transformations as tr
import basic_shapes as bs
import scene_graph as sg
import easy_shaders as es


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True


# we will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    #if action != glfw.PRESS:
        #return
    
    global controller

    if key== glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        sys.exit()



def createCar(r, g, b):

    gpuBlackCube = es.toGPUShape(bs.createColorCube(0, 0, 0))
    gpuColorCube = es.toGPUShape(bs.createColorCube(r, g, b))
    gpuPinkCube = es.toGPUShape(bs.createColorCube(1, 0.9, 0.9))
    gpuLightblueCube = es.toGPUShape(bs.createColorCube(0.87, 0.87, 1))

    # Creating a single wheel
    wheel = sg.SceneGraphNode("wheel")
    wheel.transform = tr.scale(0.2, 0.8, 0.2)
    wheel.childs += [gpuBlackCube]

    wheelrotation = sg.SceneGraphNode("wheelrotation")
    wheelrotation.childs += [wheel]

    # Differentiating between front and rear wheel
    frontwheel = sg.SceneGraphNode("frontwheel")
    frontwheel.transform = tr.translate(0.3, 0, -0.12)
    frontwheel.childs += [wheelrotation]

    rearwheel = sg.SceneGraphNode("frontwheel")
    rearwheel.transform = tr.translate(-0.3, 0, -0.12)
    rearwheel.childs += [wheelrotation]

    # Creating the car chasis with 2 parts
    chasisunderneath = sg.SceneGraphNode("chasisunderneath")
    chasisunderneath.transform = tr.scale(1, 0.7, 0.2)
    chasisunderneath.childs += [gpuColorCube]

    chasisontop = sg.SceneGraphNode("chasisontop")
    chasisontop.transform = np.matmul(tr.translate(-0.11, 0, 0.2), tr.scale(0.76, 0.7, 0.17))
    chasisontop.childs += [gpuPinkCube]

    chasis = sg.SceneGraphNode("chasis")
    chasis.childs += [chasisunderneath, chasisontop]

    # Creating rear-view mirror
    mirror = sg.SceneGraphNode("mirror")
    mirror.transform = np.matmul(tr.translate(0.218, 0, 0.16), tr.scale(0.08, 0.8, 0.08))
    mirror.childs += [gpuColorCube]

    # Creating windscreen
    windscreen = sg.SceneGraphNode("windscreen")
    windscreen.transform = np.matmul(tr.translate(-0.11, 0, 0.2), tr.scale(0.761, 0.5, 0.14))
    windscreen.childs += [gpuLightblueCube]

    # Creating only one row of windows
    rowwindows = sg.SceneGraphNode("rowwindows")
    rowwindows.transform = tr.scale(0.15, 0.701, 0.14)
    rowwindows.childs += [gpuLightblueCube]

    # each row of windows
    firstrow = sg.SceneGraphNode("firstrow")
    firstrow.transform = tr.translate(0.05, 0, 0.21)
    firstrow.childs += [rowwindows]

    secondrow = sg.SceneGraphNode("secondrow")
    secondrow.transform = tr.translate(-0.3, 0, 0.21)
    secondrow.childs += [rowwindows]

    # All pieces together
    car = sg.SceneGraphNode("car")
    car.childs += [frontwheel]
    car.childs += [rearwheel]
    car.childs += [windscreen]
    car.childs += [firstrow]
    car.childs += [secondrow]
    car.childs += [mirror]
    car.childs += [chasis]

    return car

dt=0.01
theta=0
i=0
j=0
if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "3D cars via scene graph", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    
    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))
    redCarNode = createCar(1,0,0)
    blueCarNode = createCar(0,0,1)
    whiteCarNode = createCar(1,1,1)
    hotel=hv.createhotel()

    whiteCarNode.transform = tr.rotationZ(np.pi / 5)

    blueCarNode.transform = np.matmul(tr.rotationZ(-np.pi/5), tr.translate(3.0,0,0.5))

    # Using the same view and projection matrices in the whole application
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)



    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()
        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            theta += dt
        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            theta -= dt
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            i = i + dt * np.cos(theta)
            j = j + dt * np.sin(theta)
        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            i = i - dt * np.cos(theta)
            j = j - dt * np.sin(theta)
        view = tr.lookAt(
            np.array([i, j, 0.2]),  # donde estoy
            np.array([i + 0.5 * np.cos(theta), j + 0.5 * np.sin(theta), 0.2]),  # hacia donde miro
            np.array([0, 0, 1])
        )
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if controller.showAxis:
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawShape(gpuAxis, GL_LINES)

        # Moving the red car and rotating its wheels
        redCarNode.transform = tr.translate(3 * np.sin( glfw.get_time() ),0,0.5)
        redWheelRotationNode = sg.findNode(redCarNode, "wheelrotation")
        redWheelRotationNode.transform = tr.rotationY(-10 * glfw.get_time())

        # Uncomment to print the red car position on every iteration
        #print(sg.findPosition(redCarNode, "car"))

        # Drawing the Car
        #sg.drawSceneGraphNode(redCarNode, mvpPipeline, "model")
        #sg.drawSceneGraphNode(blueCarNode, mvpPipeline, "model")
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(hotel, mvpPipeline, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    
    glfw.terminate()