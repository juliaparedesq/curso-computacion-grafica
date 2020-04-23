import json
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import transformations as tr

import OpenGL.GL.shaders

import numpy as np

import sys
N=int(sys.argv[1])
paletaelegida=sys.argv[2]
nombreguardado=sys.argv[3]

with open(paletaelegida) as file:
    paleta = json.load(file)
    colortransparente=paleta['transparent']  #[0.5, 0.5, 0.5]
    otroscolores=paleta['pallete'] #[[1, 1, 1], [0.4, 1, 1], [0,0,0]]
ncolores=len(otroscolores)+1

vertex_shader = """
#version 130
in vec3 position;
in vec3 color;
out vec3 fragColor;
uniform mat4 transform; // Parametro de matriz de transformacion
void main()
{
fragColor = color;
gl_Position = transform * vec4(position, 1.0f); // Se modifica la␣
,→posicion usando la matriz
}
"""

# We will use 32 bits data, so an integer has 4 bytes
# 1 byte = 8 bits
INT_BYTES = 4


# A class to store the application control
class Controller:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.rotate = False
        self.fillPolygon = True
        self.mousePos = (0, 0)
        self.leftClickOn = False
        self.rightClickOn = False

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.rotate = False
        self.fillPolygon = True
        self.mousePos = (0, 0)
        self.leftClickOn = False
        self.rightClickOn = False


controller = Controller()


def on_key(window, key, scancode, action, mods):
    global controller
    # Keep pressed buttons
    if action == glfw.REPEAT or action == glfw.PRESS:
        controller.x += 0
    if action != glfw.PRESS:
        return
    elif key == glfw.KEY_1:
        controller.fillPolygon = not controller.fillPolygon
    elif key == glfw.KEY_ESCAPE:
        sys.exit()

    else:
        print('Unknown key')


def cursor_pos_callback(window, x, y):  # da la posición del mouse en pantalla valores entre -1 y 1
    global controller
    controller.mousePos = (x,y)


def mouse_button_callback(window, button, action, mods):  # define las acciones que ocurren al presionar el mouse
    global controller
    if (action == glfw.PRESS or action == glfw.REPEAT):
        if (button == glfw.MOUSE_BUTTON_1):
            controller.leftClickOn = True
        if (button == glfw.MOUSE_BUTTON_2):
            controller.rightClickOn = True

    elif (action == glfw.RELEASE):
        if (button == glfw.MOUSE_BUTTON_1):
            controller.leftClickOn = False
        if (button == glfw.MOUSE_BUTTON_2):
            controller.rightClickOn = False


class GPUShape:
    def __init__(self):
        self.vao = 0
        self.vbo = 0
        self.ebo = 0
        self.texture = 0
        self.size = 0


def drawShape(shaderProgram, shape, transform):
    # Binding the proper buffers
    glBindVertexArray(shape.vao)
    glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)
    # updating the new transform attribute
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "transform"), 1, GL_TRUE, transform)
    # Describing how the data is stored in the VBO
    position = glGetAttribLocation(shaderProgram, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)
    color = glGetAttribLocation(shaderProgram, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    # This line tells the active shader program to render the active element buffer with the given size
    glDrawElements(GL_TRIANGLES, shape.size, GL_UNSIGNED_INT, None)


def createQuad(color):
    # Here the new shape will be stored
    gpuShape = GPUShape()
    # Defining locations and colors for each vertex of the shape
    vertexData = np.array([
        # positions colors
        -0.5, -0.5, 0.0, color[0], color[1], color[2],
        0.5, -0.5, 0.0, color[0], color[1], color[2],
        0.5, 0.5, 0.0, color[0], color[1], color[2],
        -0.5, 0.5, 0.0, color[0], color[1], color[2]
        # It is important to use 32 bits data
    ], dtype=np.float32)
    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
         2, 3, 0], dtype=np.uint32)
    gpuShape.size = len(indices)
    # VAO, VBO and EBO and for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)
    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)
    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)
    return gpuShape

def createallquads(c=ncolores): #c=cantidadrequerida, siempre será igual a ncolores
    listanombrescolores = ['c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14',
                           'c15', 'c16', 'c17', 'c18', 'c19', 'c20']
    di={'c1': createQuad(colortransparente)}
    for i in range(c-1):
        di[listanombrescolores[i]]=createQuad(otroscolores[i])
    return di

def createallcolors(c=ncolores):
    lista=np.array([createQuad(colortransparente)],GPUShape)
    for i in range(c-1):
        lista=np.append(lista,createQuad(otroscolores[i]))
    return lista


def funcionrango(n, final=0.6, separaciontotal=1.6):  # retorna lista con posiciones del centro de los pixeles
    inicio = -(1 - (separaciontotal / (2 * n))) #centro del primer cuadrado
    distancia = (separaciontotal / n)
    return np.arange(inicio, final, distancia)


def main():
    if not glfw.init():
        sys.exit()
    width = 1000
    height = 1000

    window = glfw.create_window(width, height, "Using Transformations", None, None)
    if not window:
        glfw.terminate()
        sys.exit()
    glfw.make_context_current(window)
    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.make_context_current(window)
    # Defining shaders for our pipeline
    vertex_shader = """
    #version 130
    in vec3 position;
    in vec3 color;
    out vec3 fragColor;
    uniform mat4 transform;
    void main()
    {
    fragColor = color;
    gl_Position = transform * vec4(position, 1.0f);
    }
    """
    fragment_shader = """
    #version 130
    in vec3 fragColor;
    out vec4 outColor;
    void main()
    {
    outColor = vec4(fragColor, 1.0f);
    }"""

    # Assembling the shader program (pipeline) with both shaders
    shaderProgram = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    # Telling OpenGL to use our shader program
    glUseProgram(shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)
    # Creating shapes on GPU memory


    gpus=createallcolors()
    colorbase=gpus[0]

    matrizcolores = np.zeros((N, N), GPUShape)
    for i in range(N):
        for j in range(N):
            matrizcolores[i][j] = gpus[0]
    vectorpaleta =np.array(np.arange(ncolores),GPUShape)
    for i in vectorpaleta:
        vectorpaleta[i]=gpus[i]

    """matrizpaleta = np.zeros((2,10),GPUShape)
    for i in range(ncolores):
        if i < 10:
            matrizpaleta[0][i] = white
        else:
            matrizpaleta[1][i-10]= white"""


    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT)
        # Create transform matrix
        transall = tr.uniformScale(1.6 / (N*1.05))
        rango = funcionrango(N,0.6,1.6)
        for i in range(N):
            for j in range(N):
                trans = tr.matmul([tr.translate(rango[i], -rango[j], 0), transall])
                drawShape(shaderProgram, matrizcolores[i][j], trans)

        tran = tr.uniformScale(1.6 / (10 * 1.03))
        rangopaly=funcionrango(10)
        rangopalx=funcionrango(10,1,2)

        for i in range(ncolores):
            if i <10:
                tran2 = tr.matmul([tr.translate(rangopalx[8], -rangopaly[i], 0), tran])
                drawShape(shaderProgram, vectorpaleta[i], tran2)
            else:
                tran2 = tr.matmul([tr.translate(rangopalx[9], -rangopaly[i-10], 0), tran])
                drawShape(shaderProgram, vectorpaleta[i], tran2)


        if controller.leftClickOn:
            print(controller.mousePos)
            a=int(controller.mousePos[0] *N/ 800)
            b=int(controller.mousePos[1] *N/ 800)
            if a<N and b<N:
                matrizcolores[a][b] = colorbase
            elif controller.mousePos[0]>811 and controller.mousePos[0]<887:
                c = int(controller.mousePos[1] * 10 / 800)
                colorbase=vectorpaleta[c]
            elif controller.mousePos[0]>911 and controller.mousePos[0]<988:
                d = int(controller.mousePos[1] * 10 / 800)
                colorbase= vectorpaleta[d+10]
        if controller.fillPolygon:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glfw.swap_buffers(window)
    glfw.terminate()

print(funcionrango(N))
print(funcionrango(10,1,2))

controller.reset()
main()
