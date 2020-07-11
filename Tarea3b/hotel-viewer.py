import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import transformations as tr
import basic_shapes as bs
import scene_graph as sg
import easy_shaders as es
import cylinders as cyl
#from curvas import *
from PIL import Image

#PRUEBA:
filename = "solution.py"
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
def fast_marching_cube(X, Y, Z, scal_field, isosurface_value):
    dims = X.shape[0]-1, Y.shape[1]-1
    voxels = np.zeros(shape=dims, dtype=bool)
    for i in range(1, X.shape[0]-1):
        for j in range(1, Y.shape[1]-1):
                    voxels[i,j] = True
    return voxels

suelo=np.load('suelo.npy')
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

"""isosurface = bs.Shape([], [])
# Now let's draw voxels!
for i in range(X.shape[0] - 1):
    for j in range(X.shape[1] - 1):
        for k in range(X.shape[2] - 1):
            # print(X[i,j,k])
            if load_voxels[i, j, k]: #si es True
                temp_shape = createColorCube(i, j, k, X, Y, Z)
                merge(destinationShape=isosurface, strideSize=6, sourceShape=temp_shape)"""



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
        newNode.transform = tr.translate(i*(L+W), W+P, 0)  ##para que no se vean en la pos 0,0
        newNode.childs += [formaL]
        Ls.childs += [newNode]

    pasi =sg.SceneGraphNode('pasilloizquierdo')
    pasi.transform = np.matmul(tr.translate(W/2, (P+W)/2, 0), tr.scale(W, P+W, 10))
    pasi.childs+= [gpuPared2]

    pasdown =sg.SceneGraphNode('pasilloabajo')
    pasdown.transform = np.matmul(tr.translate((5*L+4*W)/2 + W, W/2, 0), tr.scale(5*L+4*W, W, 10))
    pasdown.childs += [gpuPared2]

    pasd = sg.SceneGraphNode('pasilloderecho')
    pasd.transform = np.matmul(tr.translate((W)/2 + 5*(L+W), (D+P+W)/2 + W, 0), tr.scale(W, D+P+W, 10))
    pasd.childs += [gpuPared2]



    vent =sg.SceneGraphNode('ventana')
    vent.transform =np.matmul(tr.translate(L/2 + W, D+P+2*W,0 ), tr.scale(L, 0.1, 10))
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
    hotel.childs +=[ventanas, pasd ,pasdown, pasi , Ls, lll]

    #suelo= np.load('suelo.npy')


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
    pipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    #gpu_suelo = es.toGPUShape(cyl.createSuelo())

    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.15, 0.15, 0.15, 1.0)


    t0 = glfw.get_time()
    camera_phi = 0
    cameraX = 0
    cameraY = 0
    velCamera = 1
    velGiro = 2



    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        # Texture Shader
        glUseProgram(pipeline.shaderProgram)

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
        view = tr.lookAt(
            np.array([10, 4,20]),
            np.array([10,4,0]),
            np.array([0,1,0])
        )

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)


        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)


        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Drawing shapes with different model transformations

        #glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.scale(5, 5, 5))
        #pipeline.drawShape(gpuGround)

        # Color shader
        glUseProgram(mvpPipeline.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

        sg.drawSceneGraphNode(hotelNode, mvpPipeline, "model")
        #mvpPipeline.drawShape(gpu_suelo)

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    glfw.terminate()
print(degrade(suelo.max()))