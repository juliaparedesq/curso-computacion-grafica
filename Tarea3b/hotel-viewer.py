import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import json
from scipy.sparse import csr_matrix
import transformations as tr
import basic_shapes as bs
import scene_graph as sg
import easy_shaders as es
import cylinders as cyl
from PIL import Image

Hotel=sys.argv[1]
with open(Hotel) as file:
    hotel= json.load(file)
    filename=hotel['filename']
    window_loss =hotel['window_loss']
    ambient_temperature=hotel['ambient_temperature']
    heater_power=hotel['heater_power']
    P = hotel['P']
    L = hotel['L']
    D = hotel['D']
    W = hotel['W']
    E = hotel['E']
    H1 = hotel['H1']
    H2 = hotel['H2']
    windows = hotel['windows']


""""#PRUEBA:
filename = "solution.npy"
window_loss = 0.01
ambient_temperature = 20
heater_power = 6
P = 1
L = 4
D = 5
W = 0.1
E = 1
H1 = 9.4
H2 = 2
windows = [0,0,1,0,1]"""


# Problem setup
HH = P+D+2*W
WW = 5*L+ 6*W
h = 0.1
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.curvasdenivel = False
        self.flechas = False
        self.showAxis = True

# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return

    global controller

    if key == glfw.KEY_0:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_SPACE:
        controller.curvasdenivel = not controller.curvasdenivel

    elif key == glfw.KEY_X:
        controller.flechas = not controller.flechas

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)


X, Y= np.mgrid[0:206:1j, 0:62:1j]

suelo=np.load(filename)
blanco=[1,1,1]
rojo=[1,0,0]
naranjo =[250/255, 146/255, 42/255]
amarillo=[252/255, 232/255, 40/255]
x= (suelo.max()- suelo.min())/3
print('blanco: ', blanco, ' temperatura: ', suelo.min())
print('amarillo: ', amarillo, ' temperatura: ', suelo.min() + x)
print('naranjo: ', naranjo, ' temperatura: ', suelo.min() +2* x)
print('rojo: ', rojo, ' temperatura: ', suelo.max())


def degrade(valor):

    if suelo.min()<= valor <= suelo.min()+ x:
        #blanco                 amarillo
        x1=suelo.min()
        r=((amarillo[0]-blanco[0])/x)*(valor-x1)+ blanco[0]
        g=((amarillo[1]-blanco[1])/x)*(valor-x1)+ blanco[1]
        b=((amarillo[2]-blanco[2])/x)*(valor-x1)+ blanco[2]

    elif suelo.min()+ x < valor <= suelo.min() +2*x:
        #amarillo                   naranjo
        x1=suelo.min() + x
        r=((naranjo[0]-amarillo[0])/x)*(valor-x1)+ amarillo[0]
        g=((naranjo[1]-amarillo[1])/x)*(valor-x1)+ amarillo[1]
        b=((naranjo[2]-amarillo[2])/x)*(valor-x1)+ amarillo[2]

    elif suelo.min()+2*x < valor <= suelo.max():
        #naranjo                    rojo
        x1=suelo.min() + 2*x
        r=((rojo[0]-naranjo[0])/x)*(valor-x1)+ naranjo[0]
        g=((rojo[1]-naranjo[1])/x)*(valor-x1)+ naranjo[1]
        b=((rojo[2]-naranjo[2])/x)*(valor-x1)+ naranjo[2]
    return [r,g,b]

def createFlecha(i,j,ang, z,h, r):
    nov=np.pi/2
    #ancho flecha
    af=r*0.07
    #ancho punta
    ap=2*r*0.1
    a,b=0.7 * r*np.cos(ang), 0.7 * r*np.sin(ang)
    z=z*h
    vertices = [(af * np.cos(ang+ nov) + i)*h, (af * np.sin(ang+nov) + j)*h, z, 0.5, 0.5, 0.5,  #e
                 (af * np.cos(ang - nov) + i)*h,( af * np.sin(ang - nov) + j)*h, z, 0.5, 0.5, 0.5,  #f
                  (a + af * np.cos(ang-nov) + i)*h, (b + af * np.sin(ang -nov) + j)*h, z, 0.5, 0.5, 0.5,  #h
                   (a + af * np.cos(ang + nov) + i)*h, (b + af * np.sin(ang + nov) + j)*h, z, 0.5, 0.5, 0.5,  # g
                    (a + ap * np.cos(ang+nov) + i)*h, (b + ap * np.sin(ang+nov) + j)*h, z, 0.5, 0.5, 0.5,  #c
                     (a + ap * np.cos(ang - nov) + i)*h, (b + ap * np.sin(ang - nov) + j)*h, z, 0.5, 0.5, 0.5,  # d
                      (r*np.cos(ang) +i)*h, (r*np.sin(ang) +j)*h , z, 0.5,0.5,0.5
                ]
    indices = [
        0,1,2,2,3,0,
        4,5,6
        ]

    return bs.Shape(vertices, indices)



def createColorCube(i, j, X, Y, Z):
    l_x = X[i]*h
    r_x = X[i+1]*h
    b_y = Y[j]*h
    f_y = Y[j+1]*h
    b_z = Z[0]
    t_z = Z[0] + h/2
    #   positions    colors
    vertices = [
        # Z+: number 1
        l_x, b_y, t_z, 0, 0, 0,
        r_x, b_y, t_z, 0, 0, 0,
        r_x, f_y, t_z, 0, 0, 0,
        l_x, f_y, t_z, 0, 0, 0,
        # Z-: number 6
        l_x, b_y, b_z, 0, 0, 0,
        r_x, b_y, b_z, 0, 0, 0,
        r_x, f_y, b_z, 0, 0, 0,
        l_x, f_y, b_z, 0, 0, 0,
        # X+: number 5
        r_x, b_y, b_z, 0, 0, 0,
        r_x, f_y, b_z, 0, 0, 0,
        r_x, f_y, t_z, 0, 0, 0,
        r_x, b_y, t_z, 0, 0, 0,
        # X-: number 2
        l_x, b_y, b_z, 0, 0, 0,
        l_x, f_y, b_z, 0, 0, 0,
        l_x, f_y, t_z, 0, 0, 0,
        l_x, b_y, t_z, 0, 0, 0,
        # Y+: number 4
        l_x, f_y, b_z, 0, 0, 0,
        r_x, f_y, b_z, 0, 0, 0,
        r_x, f_y, t_z, 0, 0, 0,
        l_x, f_y, t_z, 0, 0, 0,
        # Y-: number 3
        l_x, b_y, b_z, 0, 0, 0,
        r_x, b_y, b_z, 0, 0, 0,
        r_x, b_y, t_z, 0, 0, 0,
        l_x, b_y, t_z, 0, 0, 0
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

def my_marching_cube(suelo, lista_isosurfaces):
    sh=suelo.shape[0]-1, suelo.shape[1]-1
    voxels = np.zeros(shape=sh, dtype=float)
    for i in range(1, suelo.shape[0]-1):
        for j in range(1, suelo.shape[1]-1):
            v_min = suelo[i-1:i+2, j-1:j+1].min()
            v_max= suelo[i-1:i+2, j-1:j+1].max()
            for k in range(len(lista_isosurfaces)):
                if v_min <= lista_isosurfaces[k] and lista_isosurfaces[k]<= v_max:
                    voxels[i,j]= k+1 #se guarda su indice +1 para evitar el 0

    return voxels

def merge(destinationShape, strideSize, sourceShape):
    # current vertices are an offset for indices refering to vertices of the new shape
    offset = len(destinationShape.vertices)
    destinationShape.vertices += sourceShape.vertices
    destinationShape.indices += [(offset / strideSize) + index for index in sourceShape.indices]

def funcioncurvas(): #165
    #10 CURVAS       #166
    min, max=suelo.min(), suelo.max()    #167
    l=np.linspace(min, max, 12, dtype=int)    #168
    l=np.delete(l, 0)     #169
    l=np.delete(l, len(l)-1)    #170
    isosurface = bs.Shape([], [])       #171
    kk=np.linspace(0.5, 4.5, 10 )       #172
    my = my_marching_cube(suelo, l)         #173 # hago matriz de voxeles con isosuperficie 22
    my=csr_matrix(my)       #174
    for i in range(my.shape[0]):       #175
        for j in range(my.shape[1]):       #176
            if my[i, j]:  # si es True       #178
                k = kk[int(my[i, j]) - 1]       #180
                temp_shape = createColorCube(i, j, range(suelo.shape[0]), range(suelo.shape[1]), [round(k,2), k + 1])
                merge(destinationShape=isosurface, strideSize=6, sourceShape=temp_shape)

    gpu_surface = es.toGPUShape(isosurface)
    return gpu_surface

yg=np.load('yg.npy')
xg=np.load('xg.npy')

def new():
    nm=np.zeros(shape=xg.shape)
    for i in range(xg.shape[0]):
        for j in range(xg.shape[1]):
            y=yg[i,j] #b
            x=xg[i,j] #a
            if x!=0 or y!=0:
                if x!= 0:
                    if y>=0 and x>0:
                        ang=np.arctan(y/x)
                        nm[i, j] = ang
                    elif y>0 and x<0:
                        ang= np.pi + np.arctan(y/x)
                        nm[i,j]=ang
                    elif x>0 and y<=0:
                        ang=np.arctan(y/x)
                        nm[i, j] = ang
                    elif x<0 and y<0:
                        ang=np.arctan(y/x)+ np.pi
                        nm[i, j] = ang
                elif x==0:
                    if y>0:
                        ang = np.pi/2
                        nm[i, j] = ang
                    elif y<0:
                        ang = - np.pi/2
                        nm[i, j] = ang

    return nm
kk=new()
#print(csr_matrix(kk[100:120, 0:15]))
#print(csr_matrix(xg[100:120, 0:15]))
#print(csr_matrix(yg[100:120, 0:15]))


def fngradiente():
    isosurface = bs.Shape([], [])
    my = new()
    for i in range(my.shape[0]):
        for j in range(my.shape[1]):
            y = yg[i, j]  # b
            x = xg[i, j]  # a
            if (x != 0 or y != 0):  # si es True
                temp_shape = createFlecha(i,j, my[i,j], 0.2, h, 0.8)
                merge(destinationShape=isosurface, strideSize=6, sourceShape=temp_shape)

    gpu_surface = es.toGPUShape(isosurface)

    return gpu_surface

def createhotel():

    gpuPared = es.toGPUShape(bs.createColorCube(51/255, 102/255, 153/255))
    gpuVentana =  es.toGPUShape(bs.createColorCube(133/255, 232/255, 255/255))
    gpuTecho = es.toGPUShape(bs.createColorCube(1,250/255, 239/255))
    gpuPared2 =es.toGPUShape(bs.createColorCube(100/255, 150/255, 153/255))
    gpu_suelo = es.toGPUShape(bs.createSuelo(suelo.shape[0], suelo.shape[1],degrade, suelo))
    gpucalefactor = es.toGPUShape(bs.createColorCube(167/255, 111/255, 55/255))



    # Creating a single wheel
    alto=5
    horizontal = sg.SceneGraphNode("horizontal")
    horizontal.transform = np.matmul(tr.translate((L-E)/2 +W, W/2, 0), tr.scale(L-E, W, alto))
    horizontal.childs += [gpuPared]

    vertical = sg.SceneGraphNode('vertical')
    vertical.transform = np.matmul(tr.translate(W/2, (D+W)/2, 0), tr.scale(W, D+W, alto))
    vertical.childs +=[gpuPared]

    techo = sg.SceneGraphNode('techo')
    techo.transform = np.matmul(tr.translate(WW/2, (HH) / 2, alto-0.1), tr.scale(WW, HH, 0.1))
    techo.childs += [gpuTecho]

    piso = sg.SceneGraphNode('piso')
    piso.transform = tr.scale(h, h,1)
    piso.childs += [gpu_suelo]

    formaL = sg.SceneGraphNode('L')
    formaL.childs +=[horizontal, vertical]

    Ls = sg.SceneGraphNode("Ls")
    t = "translatedL"
    for i in range(5):
        newNode = sg.SceneGraphNode(t + str(i))
        newNode.transform = tr.translate(i*(L+W), W+P, alto/2)  ##para que no se vean en la pos 0,0
        newNode.childs += [formaL]
        Ls.childs += [newNode]

    pasi =sg.SceneGraphNode('pasilloizquierdo')
    pasi.transform = np.matmul(tr.translate(W/2, (P+W)/2, alto/2), tr.scale(W, P+W, alto))
    pasi.childs+= [gpuPared2]

    pasdownizq =sg.SceneGraphNode('pasilloabajoizq')
    pasdownizq.transform = np.matmul(tr.translate((W+H1)/2, W/2, alto/2), tr.scale(W+H1, W, alto))
    pasdownizq.childs += [gpuPared2]

    pasdownder = sg.SceneGraphNode('pasilloabajoder')
    pasdownder.transform = np.matmul(tr.translate(H1+H2+W +(WW - (H1+H2+W))/2, W / 2, alto / 2),
                                     tr.scale(WW - (H1+H2+W), W, alto))
    pasdownder.childs += [gpuPared2]

    calefactor=sg.SceneGraphNode('cale')
    calefactor.transform = np.matmul(tr.translate(H1+W+ (H2)/2, W/2, alto/2), tr.scale(H2, W, alto) )
    calefactor.childs += [gpucalefactor]

    pasd = sg.SceneGraphNode('pasilloderecho')
    pasd.transform = np.matmul(tr.translate((W)/2 + 5*(L+W), (HH)/2 , alto/2), tr.scale(W, HH, alto))
    pasd.childs += [gpuPared2]

    vent =sg.SceneGraphNode('ventana')
    vent.transform =np.matmul(tr.translate(L/2 + W, HH - W/2, alto/2 ), tr.scale(L, W, alto))
    vent.childs +=[gpuVentana]
    ventanas = sg.SceneGraphNode("ventanas")
    t = "translatedVent"
    for i in range(5):
        newNode = sg.SceneGraphNode(t + str(i))
        newNode.transform = tr.translate(i * (L + W), 0, 0)  ##para que no se vean en la pos 0,0
        newNode.childs += [vent]
        ventanas.childs += [newNode]


    # All pieces together
    hotel = sg.SceneGraphNode("hotel")
    hotel.childs +=[ piso, pasd, ventanas, pasdownizq, pasdownder, calefactor, pasi , Ls, techo]

    return hotel


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Hotel-viewer", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)
    hotelNode = createhotel()
    flefle = fngradiente()
    gpu_surface = funcioncurvas()
    gpuAxis = es.toGPUShape(bs.createAxis(H1+W))
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()


    # Assembling the shader program



    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.15, 0.15, 0.15, 1.0)
    glEnable(GL_DEPTH_TEST)




    t0 = glfw.get_time()
    camera_phi = 0
    cameraX = L / 2
    cameraY = W + P / 2
    velCamera = 1
    velGiro = 2

    t0 = glfw.get_time()
    camera_theta = np.pi / 4

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1


        # Setting up the view transform

        visionX = np.cos(camera_phi)
        visionY = np.sin(camera_phi)
        visionZ = 3

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_phi += velGiro * dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_phi -= velGiro * dt

        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            cameraX += velCamera * dt * visionX
            cameraY += velCamera * dt * visionY

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            cameraX -= velCamera * dt * visionX
            cameraY -= velCamera * dt * visionY

        viewPos = np.array([cameraX, cameraY, 1])

        view = tr.lookAt(
            viewPos,
            np.array([cameraX + visionX, cameraY + visionY, 1]),
            np.array([0, 0, 1])
        )

        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)


        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)



        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        if controller.curvasdenivel:
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawShape(gpu_surface)
        if controller.flechas:
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawShape(flefle)

        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.uniformScale(1))

        sg.drawSceneGraphNode(hotelNode, mvpPipeline, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)


    glfw.terminate()
