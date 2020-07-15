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

path = sys.argv[1]

with open(path, newline='') as File:
    reader = csv.reader(File)
    puntos5=np.ndarray(shape=(0,3), dtype=float)
    for row in reader:
        puntos5=np.append(puntos5, np.array([[float(row[0]), float(row[1]), float(row[2])]]), 0 )

#print(np.array([[puntos5[0]]]).T, puntos5[1].T, puntos5[2].T, puntos5[3].T, puntos5[4].T)
p1 = np.array([puntos5[0]]).T
p2 = np.array([puntos5[1]]).T
p3 = np.array([puntos5[2]]).T
p4 = np.array([puntos5[3]]).T
p5 = np.array([puntos5[4]]).T
curva= curvas.CR(p1, p2, p3, p4, p5)


class Controller:
    def __init__(self):
        self.mousePos = (300,300)
controller = Controller()

def cursor_pos_callback(window, x, y):  # da la posición del mouse en pantalla.
    global controller
    controller.mousePos = (x,y)

def create5aves(): #c es la curva

    scaledBird = sg.SceneGraphNode("scaledBird")
    scaledBird.transform = tr.uniformScale(0.6)
    scaledBird.childs += [bird.createbird()] # Re-using the previous function

    birds = sg.SceneGraphNode("birds")
    t= "translatedBird"
    for i in range(5):
        newNode = sg.SceneGraphNode(t + str(i))
        newNode.transform = tr.translate(1, 1, -10)  ##para que no se vean en la pos 0,0
        newNode.childs += [scaledBird]
        birds.childs += [newNode]

    return birds

if __name__ == "__main__":

    if not glfw.init():
        sys.exit()
    width = 600
    height = 600

    window = glfw.create_window(width, height, "bird-herd", None, None)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)


    phongnotexture = ls.SimplePhongShaderProgram()
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    phong = ls.SimpleTexturePhongShaderProgram()
    glUseProgram(phong.shaderProgram)

    glClearColor(0.85, 0.85, 0.85, 1.0)
    glEnable(GL_DEPTH_TEST)

    gpuAxis = es.toGPUShape(bs.createAxis(7))
    birdNode = bird.createbird()
    gpuCerro2 = es.toGPUShape(f.generateTextureCerro(20, "cerro2.jpg", 1.0, 0, 2.0), GL_REPEAT, GL_NEAREST)
    gpuCerro = es.toGPUShape(f.generateTextureCerro(20, "cerro3.jpg", 1.0, 0, 2.0), GL_REPEAT, GL_NEAREST)
    gpuCielo = es.toGPUShape(cyl.generateTextureCylinder(150, "cielo.jpg", 15.0, -2, 25.0), GL_REPEAT, GL_NEAREST)
    gpuSueloCube = es.toGPUShape(bs.createTextureNormalsCube('pasto.jpg'), GL_REPEAT, GL_NEAREST)
    birds = create5aves()
    t0 = glfw.get_time()
    camera_theta = 0
    cama= 0
    i=1



    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Setting up the view transform
        L=8
        R=L-0.2
        H=7
        x= controller.mousePos[0]
        y = controller.mousePos[1]

        thetaa = -  np.pi * x / (600) +  3* np.pi / 4  #mov horizontal
        phii = np.pi * y / (2400) + 3 * np.pi / 8

        X = R * np.cos(thetaa) * np.sin(phii) +0.1
        Y = R * np.sin(thetaa) * np.sin(phii) +0.1
        Z = R * np.cos(phii) + H

        viewPos = np.array([-L, -L, H])
        at= np.array([X-2, Y, Z])
        eye=viewPos
        eye  = np.array([-3, -8, H])
        a=eye[0]
        b=eye[1]
        c=eye[2]

        #ESTE SIRVE PARA LA TAREA, POR AHORA PONDRE UNO FIJO PA VER EL DIBUJO GRAL
        view = tr.lookAt(
            eye , #viewPos posicion de la camara  EYE
            at, #np.array([8,8,L/2]),  #at,  #donde mira la camara  AT
            np.array([0,0,1])    #up   #vector que sale de la cabeza del 'camarografo'   UP
            )

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        projection = tr.perspective(45, float(width) / float(height), 0.1, 100)
        """glUseProgram(mvpPipeline.shaderProgram)  #AXIS!!
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        projection = tr.perspective(45, float(width) / float(height), 0.1, 100)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        mvpPipeline.drawShape(gpuAxis, GL_LINES)"""


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


        glUseProgram(phong.shaderProgram)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Ls"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Ka"), 0.4, 0.4, 0.4)  # 1 es muy brillante/blanco todo, 0 osuro/negro
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Kd"), 0.9, 0.5, 0.5)  # 0,0,0 todo es color sombra 1,1,1 las caras no sombras son muy fluor
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "Ks"), 1, 1, 1)  # 1 luz de destello blanca, 0 no hay.
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "lightPosition"), 3, 3, 3)
        glUniform3f(glGetUniformLocation(phong.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(phong.shaderProgram, "shininess"), 100)
        glUniform1f(glGetUniformLocation(phong.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(phong.shaderProgram, "linearAttenuation"), 0.3)  # 1 colores más oscuros, 0 colores más reales
        glUniform1f(glGetUniformLocation(phong.shaderProgram, "quadraticAttenuation"), 0.01)  # 1 no destello
        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "view"), 1, GL_TRUE, view)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.uniformScale(40), tr.translate(0, 0, -0.5)))
        phong.drawShape(gpuSueloCube)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(15*np.cos(np.pi*3/8), 15*np.sin(np.pi*3/8), 0), tr.uniformScale(1.8)))
        phong.drawShape(gpuCerro2)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(15*np.cos(np.pi/4), 15*np.sin(np.pi/4), 0), tr.uniformScale(1.5)))
        phong.drawShape(gpuCerro)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(15 * np.cos(np.pi *7/ 12), 15 * np.sin(np.pi*7 / 12), 0), tr.uniformScale(2)))
        phong.drawShape(gpuCerro)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(15*np.cos(np.pi/14), 15*np.sin(np.pi/14), 0), tr.uniformScale(2.3)))
        phong.drawShape(gpuCerro2)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.translate(15*np.cos(7*np.pi/16), 15*np.sin(7*np.pi/16), 0), tr.uniformScale(1)))
        phong.drawShape(gpuCerro)

        glUniformMatrix4fv(glGetUniformLocation(phong.shaderProgram, "model"), 1, GL_TRUE, np.matmul(tr.uniformScale(1), tr.translate(0, 0, 0)))
        phong.drawShape(gpuCielo)


        glUseProgram(phongnotexture.shaderProgram)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Ls"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Ka"), 0.4, 0.4, 0.4)  # 1 es muy brillante/blanco todo, 0 osuro/negro
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Kd"), 0.9, 0.5, 0.5)  # 0,0,0 todo es color sombra 1,1,1 las caras no sombras son muy fluor
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "Ks"), 1, 1, 1)  # 1 luz de destello blanca, 0 no hay.
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "lightPosition"), 3, 3, 3)
        glUniform3f(glGetUniformLocation(phongnotexture.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(phongnotexture.shaderProgram, "shininess"), 100)
        glUniform1f(glGetUniformLocation(phongnotexture.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(phongnotexture.shaderProgram, "linearAttenuation"), 0.3)  # 1 colores más oscuros, 0 colores más reales
        glUniform1f(glGetUniformLocation(phongnotexture.shaderProgram, "quadraticAttenuation"), 0.01)  # 1 no destello
        glUniformMatrix4fv(glGetUniformLocation(phongnotexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongnotexture.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(phongnotexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        if int(i) == 157:
            i=1


        for k in range(5):
            birdk = sg.findNode(birds, "translatedBird" +str(k))
            try:
                dy = (curva[int(i) + 7 * k - 1][1] - curva[int(i) + 7 * k + 1][1])
                dx = (curva[int(i) + 7 * k + 1][0] - curva[int(i) + 7 * k - 1][0])
                dy1=dy
                dx1=dx
            except:
                dy=dy1
                dx=dx1

            ang = np.arctan(dy / dx)
            
            if curva[0][1]<0:
                ang=180-ang

            try:
                x=curva[int(i) + 7 * k][0]
                y=curva[int(i) + 7 * k][1]
                z=curva[int(i) + 7 * k][2]
                dxx=x
                dyy=y
                dzz=z
            except:
                i=1
            if ang <= -1:
                birdk.transform = np.matmul(
                    tr.translate(curva[int(i) + 7 * k][0], curva[int(i) + 7 * k][1], curva[int(i) + 7 * k][2]),
                    tr.rotationZ(ang))
            else:
                birdk.transform = np.matmul(
                    tr.translate(curva[int(i) + 7 * k][0], curva[int(i) + 7 * k][1], curva[int(i) + 7 * k][2]),
                    tr.rotationZ(-ang))

        avess = sg.findNode(birds, "scaledBird")
        bird.aleteo(300*np.sin(10*glfw.get_time()), avess.childs[0])

        i+=0.25/2
        sg.drawSceneGraphNode(birds, phongnotexture, "model")

        glfw.swap_buffers(window)

    glfw.terminate()