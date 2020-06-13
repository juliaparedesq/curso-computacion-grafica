import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys

import transformations as tr
import basic_shapes as bs
import scene_graph as sg
import easy_shaders as es
import lighting_shaders as ls



class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.mousePos = (300,300)
controller = Controller()

def cursor_pos_callback(window, x, y):  # da la posición del mouse en pantalla.
    global controller
    controller.mousePos = (x,y)

def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return

    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        sys.exit()

    else:
        print('Unknown key')

def createbird():
    #gpus
    gpupico = es.toGPUShape(bs.createColorNormalsCube(1, 1, 0))
    gpucabeza = es.toGPUShape(bs.createColorNormalsCube(0.5, 0.5, 0.5))
    gpucuerpo = es.toGPUShape(bs.createColorNormalsCube(0.6, 0.6, 0.6))
    gpuala = es.toGPUShape(bs.createColorNormalsCube(0.7, 0.7, 0.7))
    gpucola = es.toGPUShape(bs.createColorNormalsCube(0.5, 0.5, 0.5))

    #creating
    pico = sg.SceneGraphNode('pico')
    pico.transform = np.matmul(tr.translate(0.65,0,-0.1), tr.scale(0.3,0.1,0.2))
    pico.childs += [gpupico]
    cabeza = sg.SceneGraphNode('cabeza')
    cabeza.transform = np.matmul(tr.translate(0.25,0,0), tr.scale(0.5,0.5,0.5))
    cabeza.childs += [gpucabeza]

    cuello = sg.SceneGraphNode('cuello')
    cuello.childs += [cabeza, pico]

    cuerpo = sg.SceneGraphNode('cuerpo')
    cuerpo.transform = np.matmul(tr.translate(-0.5,0,0), tr.scale(1,0.7,0.6))
    cuerpo.childs += [gpucuerpo]


    colatransform = sg.SceneGraphNode('colatransform')
    colatransform.transform = np.matmul(tr.translate(-1.1,0,0.0), tr.scale(0.25,0.25,0.15))
    colatransform.childs += [gpucola]

    cola = sg.SceneGraphNode('cola')
    cola.childs += [colatransform]

    #alas tamaño
    alas = sg.SceneGraphNode('alas')
    alas.transform = tr.scale(0.4,0.6, 0.18)  #0.18
    alas.childs += [gpuala]

    ala1der = sg.SceneGraphNode('ala1der')
    ala1der.transform =  tr.translate(-0.5,0.65,0.0)
    ala1der.childs += [alas]
    ala2der =sg.SceneGraphNode('ala2der')
    ala2der.transform =  tr.translate(-0.5, 1.25,0.0)
    ala2der.childs += [alas]

    ala1izq = sg.SceneGraphNode('ala1izq')
    ala1izq.transform = tr.translate(-0.5,-0.65,0.0)
    ala1izq.childs += [alas]
    ala2izq =sg.SceneGraphNode('ala2izq')
    ala2izq.transform =  tr.translate(-0.5, -1.25,0.0)
    ala2izq.childs += [alas]

    #All pieces together
    bird = sg.SceneGraphNode('bird')
    bird.childs += [cuello]
    bird.childs += [cuerpo]
    bird.childs += [ala1izq]
    bird.childs += [ala2izq]
    bird.childs += [ala1der]
    bird.childs += [ala2der]
    bird.childs += [cola]

    return bird

def aleteo(x, birdnode):
    y = 0.6
    z = 0.18
    #############  ALAS ARRIBA
    alpha = np.arctan(z / y)
    th = -np.pi * x / 2100 + np.pi / 7
    dz0 = 0.3 * np.sin(th)
    d1 = 0.3 - 0.3 * np.cos(th)
    d2 = y / 2 - np.sqrt(y ** 2 + z ** 2) * np.cos(th + alpha) / 2
    d22 = 2 * d2
    ###########  CUELLO
    phi = -np.pi * x / (300 * 30) + np.pi / 30
    ### CABEZA
    zcab = 0.5
    mov = zcab * np.sin(phi) / 2

    ########### ALAS ABAJO
    alpha_2 = np.arctan(z / y)
    th_2 = -np.pi * x / (20 * 300) + np.pi / 20
    dz0_2 = 0.3 * np.sin(th_2)
    dz1_2 = y * np.sin(th_2)
    d1_2 = 0.3 - 0.3 * np.cos(th_2)
    d2_2 = y / 2 - np.sqrt(y ** 2 + z ** 2) * np.cos(th_2 + alpha_2) / 2
    d22_2 = 2 * d2_2

    ########## COLA
    beta = -np.pi * x / (30 * 300) + np.pi / 30  #30
    zcola = 0.15
    movcola = zcola * np.sin(beta) / 2

    if x < 300:

        ala1izqNode = sg.findNode(birdnode, "ala1izq")
        ala1izqNode.transform = np.matmul(tr.translate(-0.5, -0.65 + d1, dz0), tr.rotationX(-th))

        ala2izqNode = sg.findNode(birdnode, "ala2izq")
        ala2izqNode.transform = np.matmul(tr.translate(-0.5, -1.25 + (d1 + d22), dz0), tr.rotationX(th))

        ala1derNode = sg.findNode(birdnode, "ala1der")
        ala1derNode.transform = np.matmul(tr.translate(-0.5, 0.65 - (d1), dz0), tr.rotationX(th))

        ala2derNode = sg.findNode(birdnode, "ala2der")
        ala2derNode.transform = np.matmul(tr.translate(-0.5, 1.25 - (d1 + d22), dz0), tr.rotationX(-th))

        cuelloNode = sg.findNode(birdnode, "cuello")
        cuelloNode.transform = np.matmul(tr.translate(-mov, 0, 0), tr.rotationY(phi))

        colaNode = sg.findNode(birdnode, "colatransform")
        ss=np.matmul(tr.scale(0.25,0.25,0.15), tr.rotationY(beta))
        colaNode.transform = np.matmul(tr.translate(movcola-1.1,0,0.0),ss)

    elif x > 300:
        ala1izqNode = sg.findNode(birdnode, "ala1izq")
        ala1izqNode.transform = np.matmul(tr.translate(-0.5, -0.65 + d1_2, dz0_2), tr.rotationX(-th_2))

        ala2izqNode = sg.findNode(birdnode, "ala2izq")
        ala2izqNode.transform = np.matmul(tr.translate(-0.5, -1.25, dz1_2 + dz0_2), tr.rotationX(-th_2))

        ala1derNode = sg.findNode(birdnode, "ala1der")
        ala1derNode.transform = np.matmul(tr.translate(-0.5, 0.65 -d1_2, dz0_2), tr.rotationX(th_2))

        ala2derNode = sg.findNode(birdnode, "ala2der")
        ala2derNode.transform = np.matmul(tr.translate(-0.5, 1.25, dz1_2 + dz0_2), tr.rotationX(th_2))

        cuelloNode =sg.findNode(birdnode, "cuello")
        cuelloNode.transform = np.matmul(tr.translate( mov , 0 , 0  ), tr.rotationY(phi))

        colaNode =sg.findNode(birdnode, "colatransform")
        ss=np.matmul(tr.scale(0.25,0.25,0.15), tr.rotationY(beta))
        colaNode.transform = np.matmul(tr.translate(movcola-1.1,0,0.0),ss)
    return


if __name__ == "__main__":

    if not glfw.init():
        sys.exit()
    width = 600
    height = 600

    window = glfw.create_window(width, height, "3D cars via scene graph", None, None)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)
    glfw.set_key_callback(window, on_key)


    phong = ls.SimplePhongShaderProgram()
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    glUseProgram(phong.shaderProgram)

    glClearColor(0.85, 0.85, 0.85, 1.0)
    glEnable(GL_DEPTH_TEST)

    gpuAxis = es.toGPUShape(bs.createAxis(7))
    birdNode = createbird()


    viewPos = np.array([-2,-4,2])
    view = tr.lookAt(
        viewPos,
        np.array([0, 0, 0]),
        np.array([0, 0, 1])
    )

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if controller.showAxis:
            glUseProgram(mvpPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            projection = tr.perspective(45, float(width) / float(height), 0.1, 100)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
            mvpPipeline.drawShape(gpuAxis, GL_LINES)


        x = controller.mousePos[1]
        aleteo(x, birdNode)

        glUseProgram(phong.shaderProgram)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        # Object is barely visible at only ambient. Diffuse behavior is slightly red. Sparkles are white
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Ka"), 0.4, 0.4,
                    0.4)  # 1 es muy brillante/blanco todo, 0 osuro/negro
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Kd"), 0.9, 0.5,
                    0.5)  # 0,0,0 todo es color sombra 1,1,1 las caras no sombras son muy fluor
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Ks"), 1, 1,
                    1)  # 1 luz de destello blanca, 0 no hay.

        # TO DO: Explore different parameter combinations to understand their effect!

        glUniform3f(glGetUniformLocation(phong.shaderProgram, "lightPosition"), 3, 3, 3)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "viewPosition"), viewPos[0], viewPos[1],
                    viewPos[2])
        glUniform1ui(glGetUniformLocation(phong.shaderProgram, "shininess"), 100)

        glUniform1f(glGetUniformLocation(phong.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(phong.shaderProgram, "linearAttenuation"),
                    0.3)  # 1 colores más oscuros, 0 colores más reales
        glUniform1f(glGetUniformLocation(phong.shaderProgram, "quadraticAttenuation"), 0.01)  # 1 no destello

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "view"), 1, GL_TRUE, view)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, tr.uniformScale(0.1))
        #birdNode.transform = tr.rotationZ(-np.pi/5)

        sg.drawSceneGraphNode(birdNode, phong, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    glfw.terminate()