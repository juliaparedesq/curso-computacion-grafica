import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
from scipy.sparse import csr_matrix
import transformations as tr
import basic_shapes as bs
import scene_graph as sg
import easy_shaders as es
import cylinders as cyl
#from curvas import *
from PIL import Image

#PRUEBA:
filename = "solution.npy"
window_loss = 0.01
ambient_temperature = 20
heater_power = 3
P = 1
L = 4
D = 5
W = 0.1
E = 1
H1 = 9.4
H2 = 2
windows = [0,1,1,1,0]
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

    elif key == glfw.KEY_RIGHT_CONTROL:
        controller.flechas = not controller.flechas

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

X, Y= np.mgrid[0:206:1j, 0:62:1j]

suelo=np.load(filename)
blanco=[1,1,1]
rojo=[1,0,0]
naranjo =[250/255, 146/255, 42/255]
amarillo=[252/255, 232/255, 40/255]
def degrade(valor):
    x= (suelo.max()- suelo.min())/3
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
def createColorCube(i, j, X, Y, vox):
    l_x = X[i]*h
    r_x = X[i+1]*h
    b_y = Y[j]*h
    f_y = Y[j+1]*h
    b_z = vox[i,j]*h
    t_z = vox[i,j]*h + h
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

def my_marching_cube(suelo, lista_isosurfaces):
    sh=suelo.shape[0]-1, suelo.shape[1]-1
    voxels = np.zeros(shape=sh, dtype=float)
    for i in range(1, suelo.shape[0]-1):
        for j in range(1, suelo.shape[1]-1):
            v_min = suelo[i-1:i+2, j-1:j+1].min()
            v_max= suelo[i-1:i+2, j-1:j+1].max()
            for k in lista_isosurfaces:
                if v_min <= k and k<= v_max:
                    voxels[i,j]= suelo[i,j]
                else:
                    voxels[i,j] = 0
    return voxels

def merge(destinationShape, strideSize, sourceShape):
    # current vertices are an offset for indices refering to vertices of the new shape
    offset = len(destinationShape.vertices)
    destinationShape.vertices += sourceShape.vertices
    destinationShape.indices += [(offset / strideSize) + index for index in sourceShape.indices]

def funcioncurvas():
    #10 CURVAS 
    min, max=suelo.min(), suelo.max()
    l=np.linspace(min, max, 12)
    l=np.delete(l, 0)
    l=np.delete(l, len(l)-1)
    l[0]=22

    isosurface = bs.Shape([], [])
    # Now let's draw voxels!

    k = [2,4,6,8,10, 12, 14, 16, 18, 20]
    my = my_marching_cube(suelo, l)  # hago matriz de voxeles con isosuperficie 22
    # my = np.zeros((7, 7), dtype=bool)
    # my[2, 2] = True
    # print(my)
    my=csr_matrix(my)

    for i in range(my.shape[0]):
        for j in range(my.shape[1]):
            # print(X[i,j,k])
            if my[i, j]:  # si es True
                # print(i, j)
                temp_shape = createColorCube(i, j, range(suelo.shape[0]), range(suelo.shape[1]), my)
                merge(destinationShape=isosurface, strideSize=6, sourceShape=temp_shape)

    gpu_surface = es.toGPUShape(isosurface)

    return gpu_surface

def createhotel():

    gpuPared = es.toGPUShape(bs.createColorCube(51/255, 102/255, 153/255))
    gpuVentana =  es.toGPUShape(bs.createColorCube(133/255, 232/255, 255/255))
    gpuPared2 =es.toGPUShape(bs.createColorCube(100/255, 150/255, 153/255))
    gpu_suelo = es.toGPUShape(bs.createSuelo(suelo.shape[0], suelo.shape[1],degrade, suelo))


    # Creating a single wheel
    horizontal = sg.SceneGraphNode("horizontal")
    horizontal.transform = np.matmul(tr.translate((L-E)/2 +W, W/2, 0), tr.scale(L-E, W, 10))
    horizontal.childs += [gpuPared2]

    vertical = sg.SceneGraphNode('vertical')
    vertical.transform = np.matmul(tr.translate(W/2, (D+W)/2, 0), tr.scale(W, D+W, 10))
    vertical.childs +=[gpuPared]

    lll = sg.SceneGraphNode('lll')
    lll.transform = tr.scale(h, h,1)
    lll.childs += [gpu_suelo]

    formaL = sg.SceneGraphNode('L')
    formaL.childs +=[horizontal, vertical]

    Ls = sg.SceneGraphNode("Ls")
    t = "translatedL"
    for i in range(5):
        newNode = sg.SceneGraphNode(t + str(i))
        newNode.transform = tr.translate(i*(L+W), W+P, 5)  ##para que no se vean en la pos 0,0
        newNode.childs += [formaL]
        Ls.childs += [newNode]

    pasi =sg.SceneGraphNode('pasilloizquierdo')
    pasi.transform = np.matmul(tr.translate(W/2, (P+W)/2, 5), tr.scale(W, P+W, 10))
    pasi.childs+= [gpuPared2]

    pasdown =sg.SceneGraphNode('pasilloabajo')
    pasdown.transform = np.matmul(tr.translate((5*L+4*W)/2 + W, W/2, 5), tr.scale(5*L+4*W, W, 10))
    pasdown.childs += [gpuPared2]

    pasd = sg.SceneGraphNode('pasilloderecho')
    pasd.transform = np.matmul(tr.translate((W)/2 + 5*(L+W), (D+P+W)/2 + W, 5), tr.scale(W, D+P+W, 10))
    pasd.childs += [gpuPared2]



    vent =sg.SceneGraphNode('ventana')
    vent.transform =np.matmul(tr.translate(L/2 + W, D+P+2*W, 5 ), tr.scale(L, 0.1, 10))
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
    hotel.childs +=[ pasd ,pasdown, pasi , Ls, lll, ventanas]

    return hotel


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
    hotelNode = createhotel()

    # Assembling the shader program

    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.15, 0.15, 0.15, 1.0)
    gpuAxis = es.toGPUShape(bs.createAxis(15))
    gpu_surface=funcioncurvas()


    t0 = glfw.get_time()
    camera_phi = 0
    cameraX = 0
    cameraY = 0
    velCamera = 1
    velGiro = 2

    """isosurface = bs.Shape([], [])
    # Now let's draw voxels!

    k = 2
    oo=[22]
    my = my_marching_cube(suelo, oo)  # hago matriz de voxeles con isosuperficie 22
    # my = np.zeros((7, 7), dtype=bool)
    # my[2, 2] = True
    print(csr_matrix(my))

    for i in range(my.shape[0]):
        for j in range(my.shape[1]):
            # print(X[i,j,k])
            if my[i, j]:  # si es True
                # print(i, j)
                temp_shape = createColorCube(i, j, 5, range(suelo.shape[0]), range(suelo.shape[1]), [k, k + 1])
                merge(destinationShape=isosurface, strideSize=6, sourceShape=temp_shape)

    gpu_surface = es.toGPUShape(isosurface)"""
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
        visionZ = 0.06

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

        viewPos = np.array([cameraX, cameraY, 0.06])

        view = tr.lookAt(
            viewPos,
            np.array([cameraX + visionX, cameraY + visionY, visionZ]),
            np.array([0, 0, 1])
        )
        """view = tr.lookAt(
            np.array([10, 4,20]),
            np.array([10,4,0]),
            np.array([0,1,0])
        )
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta -= 2 * dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta += 2 * dt

        # Setting up the view transform

        camX = 15 * np.sin(camera_theta)
        camY = 15 * np.cos(camera_theta)

        viewPos = np.array([camX, camY, 6])

        view = tr.lookAt(
            viewPos,
            np.array([3, 2, 2]),
            np.array([0, 0, 1])
        )"""

        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)


        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        #if controller.curvasdenivel:
        #funcioncurvas()



        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.translate(0, 0, 0))

        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.uniformScale(1))

        sg.drawSceneGraphNode(hotelNode, mvpPipeline, "model")
        transform = np.matmul(tr.translate(0, 0, 0), tr.uniformScale(1))
        # glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.translate(5, 0, 0))

        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        mvpPipeline.drawShape(gpu_surface)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        mvpPipeline.drawShape(gpuAxis, GL_LINES)

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    glfw.terminate()
