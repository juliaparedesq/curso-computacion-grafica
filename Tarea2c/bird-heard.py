import csv
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import bird
import cylinders as cyl
import scene_graph as sg
import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as ls
import formas as f
import ex_curves as curvas

"""PARA UN FUTURO CUANDO PONGA SYS"""
###path = sys.argv[1] with open(path, newline='') as File:

with open('path.csv', newline='') as File:
    reader = csv.reader(File)
    puntos5=np.ndarray(shape=(0,3), dtype=float)
    for row in reader:
        puntos5=np.append(puntos5, np.array([[int(row[0]), int(row[1]), int(row[2])]]), 0 )

#print(np.array([[puntos5[0]]]).T, puntos5[1].T, puntos5[2].T, puntos5[3].T, puntos5[4].T)
p1 = np.array([puntos5[0]]).T
p2 = np.array([puntos5[1]]).T
p3 = np.array([puntos5[2]]).T
p4 = np.array([puntos5[3]]).T
p5 = np.array([puntos5[4]]).T

curva= curvas.CR(p1, p2, p3, p4, p5)
print(curva[0][0])


class Controller:
    def __init__(self):
        self.mousePos = (300,300)
controller = Controller()

def cursor_pos_callback(window, x, y):  # da la posición del mouse en pantalla.
    global controller
    controller.mousePos = (x,y)



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


    phongnotexture = ls.SimplePhongShaderProgram()
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    mvpTexture = es.SimpleTextureModelViewProjectionShaderProgram()
    phong = ls.SimpleTexturePhongShaderProgram()
    glUseProgram(phong.shaderProgram)

    glClearColor(0.85, 0.85, 0.85, 1.0)
    glEnable(GL_DEPTH_TEST)

    gpuAxis = es.toGPUShape(bs.createAxis(7))
    birdNode = bird.createbird()
    gpuCerro = es.toGPUShape(f.generateTextureCerro(20, "cerro3.jpg", 1.0, 0, 2.0), GL_REPEAT, GL_NEAREST)
    gpuCielo = es.toGPUShape(cyl.generateTextureCylinder(150, "cielo.jpg", 18.0, -2, 25.0), GL_REPEAT, GL_NEAREST)
    gpuSueloCube = es.toGPUShape(bs.createTextureNormalsCube('pasto.jpg'), GL_REPEAT, GL_NEAREST)

    t0 = glfw.get_time()
    camera_theta = 0
    cama= 0


    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Setting up the view transform
        L=8
        R=L-0.2
        H=7
        x= controller.mousePos[0]
        y = controller.mousePos[1]

        thetaa = -np.pi * x / (3600) +  np.pi / 3  #mov horizontal
        phii = np.pi * y / (2400) + 3 * np.pi / 8

        X = R * np.cos(thetaa) * np.sin(phii) +0.1
        Y = R * np.sin(thetaa) * np.sin(phii) +0.1
        Z = R * np.cos(phii) + H

        viewPos = np.array([-L, -L, H])
        at= np.array([X, Y, Z])
        eye=viewPos
        a=eye[0]
        b=eye[1]
        c=eye[2]

        #ESTE SIRVE PARA LA TAREA, POR AHORA PONDRE UNO FIJO PA VER EL DIBUJO GRAL
        view = tr.lookAt(
            eye , #viewPos posicion de la camara  EYE
            at, #np.array([8,8,L/2]),  #at,  #donde mira la camara  AT
            np.array([0,0,1])    #up   #vector que sale de la cabeza del 'camarografo'   UP
            )
        """view = tr.lookAt(
            np.array([10,10,10]),  # viewPos posicion de la camara  EYE
            np.array([0,0,0]),  # np.array([8,8,L/2]),  #at,  #donde mira la camara  AT
            np.array([0, 0, 1])  # up   #vector que sale de la cabeza del 'camarografo'   UP
        )"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        glUseProgram(mvpTexture.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(mvpTexture.shaderProgram, "view"), 1, GL_TRUE, view)
        projection = tr.perspective(45, float(width) / float(height), 0.1, 100)
        glUniformMatrix4fv(glGetUniformLocation(mvpTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(mvpTexture.shaderProgram, "model"), 1, GL_TRUE, tr.scale(30, 30, 30))
        #mvpTexture.drawShape(gpuSuelo)



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
        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.uniformScale(40), tr.translate(0, 0, -0.5)))

        phong.drawShape(gpuSueloCube)
        """glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.uniformScale(2*L),  tr.translate(0.5,0.5,0.5)))

        phong.drawShape(gpubackground)""" ##ESTO ES LO DEL CUADRADO CIELO JIJI

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(18*np.cos(np.pi*3/8), 18*np.sin(np.pi*3/8), 0), tr.uniformScale(1.8)))
        phong.drawShape(gpuCerro)
        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(18*np.cos(np.pi/4), 18*np.sin(np.pi/4), 0), tr.uniformScale(1.5)))
        phong.drawShape(gpuCerro)
        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(18*np.cos(np.pi/14), 18*np.sin(np.pi/14), 0), tr.uniformScale(2.3)))
        phong.drawShape(gpuCerro)
        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(18*np.cos(7*np.pi/16), 18*np.sin(7*np.pi/16), 0), tr.uniformScale(1)))
        phong.drawShape(gpuCerro)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.uniformScale(1), tr.translate(0, 0, 0)))
        phong.drawShape(gpuCielo)


        glUseProgram(phongnotexture.shaderProgram)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        # Object is barely visible at only ambient. Diffuse behavior is slightly red. Sparkles are white
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Ka"), 0.4, 0.4,
                    0.4)  # 1 es muy brillante/blanco todo, 0 osuro/negro
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Kd"), 0.9, 0.5,
                    0.5)  # 0,0,0 todo es color sombra 1,1,1 las caras no sombras son muy fluor
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Ks"), 1, 1,
                    1)  # 1 luz de destello blanca, 0 no hay.

        # TO DO: Explore different parameter combinations to understand their effect!

        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "lightPosition"), 3, 3, 3)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "viewPosition"), viewPos[0], viewPos[1],
                    viewPos[2])
        glUniform1ui(glGetUniformLocation(phongnotexture.shaderProgram, "shininess"), 100)

        glUniform1f(glGetUniformLocation(phongnotexture.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(phongnotexture.shaderProgram, "linearAttenuation"),
                    0.3)  # 1 colores más oscuros, 0 colores más reales
        glUniform1f(glGetUniformLocation(phongnotexture.shaderProgram, "quadraticAttenuation"), 0.01)  # 1 no destello

        glUniformMatrix4fv(glGetUniformLocation(phongnotexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongnotexture.shaderProgram, "view"), 1, GL_TRUE, view)


        i=int(2*glfw.get_time())
        print(i)
        x=curva[i][0]
        y=curva[i][1]
        z=curva[i][2]
        trans=tr.matmul([tr.translate(x, y, z), tr.uniformScale(0.5)])
        glUniformMatrix4fv(glGetUniformLocation(phongnotexture.shaderProgram, "model"), 1, GL_TRUE, trans)
        sg.drawSceneGraphNode(birdNode, phongnotexture, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    glfw.terminate()