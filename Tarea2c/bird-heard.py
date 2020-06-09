import csv
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import bird
import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as ls
#import ex_curves

"""PARA UN FUTURO CUANDO PONGA SYS"""
###path = sys.argv[1] with open(path, newline='') as File:

with open('path.csv', newline='') as File:
    reader = csv.reader(File)
    for row in reader:
        print(row)

class Controller:
    def __init__(self):
        self.mousePos = (300,300)
controller = Controller()

def cursor_pos_callback(window, x, y):  # da la posición del mouse en pantalla.
    global controller
    controller.mousePos = (x,y)


def createDice(image_filename):
    # Defining locations and texture coordinates for each vertex of the shape
    vertices = [
        #   positions         tex coords   normals
        # Z+: number 1
        -0.5, -0.5, 0.5, 0, 1 / 3, 0, 0, 1,
        0.5, -0.5, 0.5, 1 / 2, 1 / 3, 0, 0, 1,
        0.5, 0.5, 0.5, 1 / 2, 0, 0, 0, 1,
        -0.5, 0.5, 0.5, 0, 0, 0, 0, 1,

        # Z-: number 6
        -0.5, -0.5, -0.5, 1 / 2, 1, 0, 0, -1,
        0.5, -0.5, -0.5, 1, 1, 0, 0, -1,
        0.5, 0.5, -0.5, 1, 2 / 3, 0, 0, -1,
        -0.5, 0.5, -0.5, 1 / 2, 2 / 3, 0, 0, -1,

        # X+: number 5
        0.5, -0.5, -0.5, 0, 1, 1, 0, 0,
        0.5, 0.5, -0.5, 1 / 2, 1, 1, 0, 0,
        0.5, 0.5, 0.5, 1 / 2, 2 / 3, 1, 0, 0,
        0.5, -0.5, 0.5, 0, 2 / 3, 1, 0, 0,

        # X-: number 2
        -0.5, -0.5, -0.5, 1 / 2, 1 / 3, -1, 0, 0,
        -0.5, 0.5, -0.5, 1, 1 / 3, -1, 0, 0,
        -0.5, 0.5, 0.5, 1, 0, -1, 0, 0,
        -0.5, -0.5, 0.5, 1 / 2, 0, -1, 0, 0,

        # Y+: number 4
        -0.5, 0.5, -0.5, 1 / 2, 2 / 3, 0, 1, 0,
        0.5, 0.5, -0.5, 1, 2 / 3, 0, 1, 0,
        0.5, 0.5, 0.5, 1, 1 / 3, 0, 1, 0,
        -0.5, 0.5, 0.5, 1 / 2, 1 / 3, 0, 1, 0,

        # Y-: number 3
        -0.5, -0.5, -0.5, 0, 2 / 3, 0, -1, 0,
        0.5, -0.5, -0.5, 1 / 2, 2 / 3, 0, -1, 0,
        0.5, -0.5, 0.5, 1 / 2, 1 / 3, 0, -1, 0,
        -0.5, -0.5, 0.5, 0, 1 / 3, 0, -1, 0
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 2, 2, 3, 0,  # Z+
        7, 6, 5, 5, 4, 7,  # Z-
        8, 9, 10, 10, 11, 8,  # X+
        15, 14, 13, 13, 12, 15,  # X-
        19, 18, 17, 17, 16, 19,  # Y+
        20, 21, 22, 22, 23, 20]  # Y-

    return bs.Shape(vertices, indices, image_filename)



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


    #phong = ls.SimplePhongShaderProgram()
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    phong = ls.SimpleTexturePhongShaderProgram()
    glUseProgram(phong.shaderProgram)

    glClearColor(0.85, 0.85, 0.85, 1.0)
    glEnable(GL_DEPTH_TEST)

    gpuAxis = es.toGPUShape(bs.createAxis(7))
    #birdNode = createbird()
    gpubackground = es.toGPUShape(createDice('cielo.jpg'), GL_REPEAT, GL_LINEAR)

    t0 = glfw.get_time()
    camera_theta = 0
    cama= 0


    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Setting up the view transform
        x= controller.mousePos[0]
        y = controller.mousePos[1]

        thetaa = np.pi * x / (900) - np.pi / 3
        R= 10 * np.sqrt(2)
        phii = -np.pi * y / (600) + np.pi / 2

        X = R * np.cos(thetaa) * np.sin(phii)
        Y = R * np.sin(thetaa) * np.sin(phii)
        Z = R * np.cos(phii)

        viewPos = np.array([0.5, 0.5, 0.5])
        at= np.array([3+X, 3+Y, 2+ Z])
        eye=viewPos
        a=eye[0]
        b=eye[1]
        c=eye[2]

        up=np.array([Y*c-Z*b, Z*a-X*c, X*b-Y*a])
        view = tr.lookAt(
            eye , #viewPos posicion de la camara  EYE
            at,  #donde mira la camara  AT
            up)  #vector que sale de la cabeza del 'camarografo'   UP

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        glUseProgram(mvpPipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        projection = tr.perspective(45, float(width) / float(height), 0.1, 100)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        mvpPipeline.drawShape(gpuAxis, GL_LINES)


        x = controller.mousePos[1]
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
        phi = -np.pi * x / 3000 + np.pi / 10
        ### CABEZA
        zcab = 0.5
        mov = zcab * np.sin(phi)/2

        ########### ALAS ABAJO
        alpha_2 = np.arctan(z / y)
        th_2 = -np.pi * x / (20*300) + np.pi / 20
        dz0_2 = 0.3 * np.sin(th_2)
        dz1_2 = y * np.sin(th_2)
        d1_2 = 0.3 - 0.3 * np.cos(th_2)
        d2_2 = y / 2 - np.sqrt(y ** 2 + z ** 2) * np.cos(th_2 + alpha_2) / 2
        d22_2 = 2 * d2_2

        ########## COLA
        beta = -np.pi * x / (30*300) + np.pi / 30
        zcola = 0.15
        movcola = zcola * np.sin(beta)/2

        """if controller.mousePos[1]<300:

            ala1izqNode = sg.findNode(birdNode, "ala1izq")
            ala1izqNode.transform = np.matmul(tr.translate(-0.5, -0.65 + d1, dz0), tr.rotationX(-th))

            ala2izqNode = sg.findNode(birdNode, "ala2izq")
            ala2izqNode.transform = np.matmul(tr.translate(-0.5, -1.25 + (d1 + d22), dz0), tr.rotationX(th))

            ala1derNode = sg.findNode(birdNode, "ala1der")
            ala1derNode.transform = np.matmul(tr.translate(-0.5, 0.65 - (d1), dz0), tr.rotationX(th))

            ala2derNode = sg.findNode(birdNode, "ala2der")
            ala2derNode.transform = np.matmul(tr.translate(-0.5, 1.25 - (d1 + d22), dz0) , tr.rotationX(-th))

            cuelloNode =sg.findNode(birdNode, "cuello")
            cuelloNode.transform = np.matmul(tr.translate( -mov , 0 , 0  ),tr.rotationY(phi))

            colaNode =sg.findNode(birdNode, "cola")
            colaNode.transform = np.matmul(tr.translate(- movcola,0,0.0),tr.rotationY(beta))

        elif controller.mousePos[1]>300:
            ala1izqNode = sg.findNode(birdNode, "ala1izq")
            ala1izqNode.transform = np.matmul(tr.translate(-0.5, -0.65 + d1_2, dz0_2), tr.rotationX(-th_2))

            ala2izqNode = sg.findNode(birdNode, "ala2izq")
            ala2izqNode.transform = np.matmul(tr.translate(-0.5, -1.25, dz1_2 + dz0_2), tr.rotationX(-th_2))

            ala1derNode = sg.findNode(birdNode, "ala1der")
            ala1derNode.transform = np.matmul(tr.translate(-0.5, 0.65 -d1_2, dz0_2), tr.rotationX(th_2))

            ala2derNode = sg.findNode(birdNode, "ala2der")
            ala2derNode.transform = np.matmul(tr.translate(-0.5, 1.25, dz1_2 + dz0_2), tr.rotationX(th_2))

            cuelloNode =sg.findNode(birdNode, "cuello")
            cuelloNode.transform = np.matmul(tr.translate( mov , 0 , 0  ), tr.rotationY(phi))

            colaNode =sg.findNode(birdNode, "cola")
            colaNode.transform = np.matmul(tr.translate(movcola,0,0.0), tr.rotationY(beta))"""


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
        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.uniformScale(18), tr.translate(1,1,1)))

        phong.drawShape(gpubackground)
        #sg.drawSceneGraphNode(birdNode, phong, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    glfw.terminate()