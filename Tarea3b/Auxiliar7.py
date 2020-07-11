# coding=utf-8
"""
Nelson Marambio, CC3501, 2020-1
"""

"""
Pregunta 1: Revisar triangle_mesh.py (sin el builder) y ex_triangle_mesh.py.
"""

"""
Pregunta 2: En el archivo ex_triangle_mesh.py añada un vértice 6 ubicado espacialmente al sur del vértice 4.
Añada dos triángulos que conecten este nuevo vértice con los vértices 3-4 y 4-5. Procure enlazar correctamente su malla.
"""

import triangle_mesh as tm

# Creating all vertices.
"""
This is the order for the vertices and triangles
0-------1-------2
| \  1  | 2   / |
|   \   |   /   |
| 0   \ | /   3 |
3-------4-------5
  \   4 |  5  /
    \   |   /
      \ | /
        6
"""

vertices = [
    (0, 200, 0),
    (100, 250, 0),
    (200, 230, 50),
    (0, 100, 0),
    (110, 110, 100),
    (190, 90, 0),
    (110, 90 ,100) # Vertice 6
]

# Triangles are created from left to right
triangles = [
    tm.Triangle(3, 4, 0),
    tm.Triangle(0, 4, 1),
    tm.Triangle(1, 4, 2),
    tm.Triangle(2, 4, 5),
    tm.Triangle(6, 3, 4), # Nuevos triangulos
    tm.Triangle(6, 4, 5)  # Nuevos triangulos
]

# We use just indices to the triangle list
# Yes, we could use the triangle themselves as node objects.
meshes = [
    tm.TriangleFaceMesh(0),
    tm.TriangleFaceMesh(1),
    tm.TriangleFaceMesh(2),
    tm.TriangleFaceMesh(3),
    tm.TriangleFaceMesh(4), # Nuevas caras
    tm.TriangleFaceMesh(5)  # Nuevas caras
]

# Making connections
meshes[0].bc = meshes[1]
meshes[1].ab = meshes[0]
meshes[1].bc = meshes[2]
meshes[2].ab = meshes[1]
meshes[2].bc = meshes[3]
meshes[3].ab = meshes[2]

# Nuevas conexiones
meshes[0].ab = meshes[4]
meshes[4].bc = meshes[0]
meshes[3].bc = meshes[5]
meshes[5].bc = meshes[3]


"""
Pregunta 3: Crear una malla de triangulos del paraboloide a traves del meshBuilder 
"""

import ex_2dplotter
import numpy as np
import triangle_mesh as tm
import basic_shapes as bs


def create_paraboloid_mesh(N):
    """Creates a paraboloid mesh from ex_2dplotter file
    """
    ## Creacion del paraboloide

    ### Creamos la funcion del paraboloide
    simpleParaboloid = lambda x, y: ex_2dplotter.paraboloid(x, y, 3.0, 3.0)

    ### Generamos un numpy array con N valores entre 10 y -10
    xs = np.linspace(-10, 10, N)
    ys = np.linspace(-10, 10, N)

    ### Creamos el paraboloide
    paraboloid = ex_2dplotter.generateMesh(xs, ys, simpleParaboloid, [1, 0, 0])

    ## Creamos los vertices del paraboloide, pq este en su paraboloide.vertices tiene vertices
    # y el rgb ---> guardamos solo los 3 'primeros' cada 6, que son los vertices y no
    # los colores
    mesh_vertices = []

    for i in range(0, len(paraboloid.vertices) - 1, 6):
        mesh_vertices.append((paraboloid.vertices[i], paraboloid.vertices[i + 1], paraboloid.vertices[i + 2]))

    ## Creamos los triangulos con los indices de parabolid.indices, en mesh_triangles GUARDO TRIANGULOS, no indices como en paraboloid.indices
    mesh_triangles = []

    for i in range(0, len(paraboloid.indices) - 1, 3):
        mesh_triangles.append(
            tm.Triangle(paraboloid.indices[i], paraboloid.indices[i + 1], paraboloid.indices[i + 2]))

    ## Creamos la malla con un meshBuilder
    mesh_builder = tm.TriangleFaceMeshBuilder()

    for triangle in mesh_triangles:
        mesh_builder.addTriangle(triangle)
    #retorno malla de triangulos, lista de triangulos y lista de vertices
    return mesh_builder, mesh_triangles, mesh_vertices


"""
Pregunta 4: Obtener todos los triangulos que se encuentren en una curva de nivel. 

Tenemos dos opciones: 1) Recorrer triangulo por triangulo viendo con cuales nos quedamos.
                      2) Ir recorriendo los triangulos a traves de las conexiones, y solo avanzar si el triangulo cumple con el valor que estamos buscando.
"""


## Primera opcion:

def filter_triangles_one_by_one_range(mesh, vertices, min_value, max_value):
    """Filters triangles with the Z axis value

    :param mesh: a meshBuilder object.
    :param vertices: a list of vertexes
    :param min_value: min value in Z axis
    :param max_value: max value in Z axis
    :return: a list of triangles
    """
    new_triangles = []
    # Recorremos cada triangulo
    for triangle_mesh in mesh.triangleMeshes:
        triangle = triangle_mesh.data
        a_vertex = vertices[triangle.a]  # Vertice "a" del triangulo
        b_vertex = vertices[triangle.b]  # Vertice "b" del triangulo
        c_vertex = vertices[triangle.c]  # Vertice "c" del triangulo

        # Vemos si alguno de los valores del eje Z de los vertices esta en el rango pedido
        if (min_value <= a_vertex[2] <= max_value) or (min_value <= b_vertex[2] <= max_value) or (
                min_value <= c_vertex[2] <= max_value):
            new_triangles.append(triangle)

    return new_triangles


def filter_triangles_one_by_one_value(mesh, vertices, value):
    """Filters triangles with the Z axis value

    :param mesh: a meshBuilder object.
    :param vertices: a list of vertexes
    :param min_value: min value in Z axis
    :param max_value: max value in Z axis
    :return: a list of triangles
    """
    new_triangles = []
    # Recorremos cada triangulo
    for triangle_mesh in mesh.triangleMeshes:
        triangle = triangle_mesh.data
        z_a = vertices[triangle.a][2]
        z_b = vertices[triangle.b][2]
        z_c = vertices[triangle.c][2]

        # Tomamos los valores en "Z" de sus vertices
        z_min = min(z_a, z_b, z_c)
        z_max = max(z_a, z_b, z_c)

        if ( z_min <= value <= z_max) or\
            (z_min <= value <= z_max) or\
            (z_min <= value <= z_max):
            new_triangles.append(triangle)

    return new_triangles


## Segunda opcion:

def find_triangle_with_value(mesh, vertices, value):
    """Finds a triangle mesh that contains a vertex with the value

    :param mesh: a meshBuilder object
    :param vertices: a list of vertexes
    :param min_value: min value in Z axis
    :param max_value: max value in Z axis
    :return: a TriangleFaceMesh object (or None)
    """

    # Revisamos los triangulos hasta que encontremos uno que nos sirva
    for triangle_mesh in mesh.triangleMeshes:
        triangle = triangle_mesh.data
        z_a = vertices[triangle.a][2]
        z_b = vertices[triangle.b][2]
        z_c = vertices[triangle.c][2]

        # Tomamos los valores en "Z" de sus vertices
        z_min = min(z_a, z_b, z_c)
        z_max = max(z_a, z_b, z_c)

        if ( z_min <= value <= z_max) or\
            (z_min <= value <= z_max) or\
            (z_min <= value <= z_max):
            return triangle_mesh

    return None


def find_triangle_with_range(mesh, vertices, min_value, max_value):
    """Finds a triangle mesh that contains a vertex with the value

    :param mesh: a meshBuilder object
    :param vertices: a list of vertexes
    :param min_value: min value in Z axis
    :param max_value: max value in Z axis
    :return: a TriangleFaceMesh object (or None)
    """

    # Revisamos los triangulos hasta que encontremos uno que nos sirva
    for triangle_mesh in mesh.triangleMeshes:
        triangle = triangle_mesh.data
        a_vertex = vertices[triangle.a]
        b_vertex = vertices[triangle.b]
        c_vertex = vertices[triangle.c]
        if (min_value <= a_vertex[2] <= max_value) or (min_value <= b_vertex[2] <= max_value) or (
                min_value <= c_vertex[2] <= max_value):
            return triangle_mesh

    return None


def explore_side_range(mesh, vertices, min_value, max_value, triangle_face_mesh, filtered_triangles, cantidad):

    # Si no hay elemento, se termina la exploración por esta ruta
    if triangle_face_mesh == None:
        return filtered_triangles, cantidad

    # Si el triangulo ya lo teniamos entonces lo saltamos
    if triangle_face_mesh.data in filtered_triangles:
        return filtered_triangles, cantidad

    # Tomamos los valores en "Z" de sus vertices
    z_axis_from_a = vertices[triangle_face_mesh.data.a][2]
    z_axis_from_b = vertices[triangle_face_mesh.data.b][2]
    z_axis_from_c = vertices[triangle_face_mesh.data.c][2]

    # Vemos si alguno esta en el rango que buscamos
    if (min_value <= z_axis_from_a <= max_value) or (
            min_value <= z_axis_from_b <= max_value) or (
            min_value <= z_axis_from_c <= max_value):
        # Agregamos el triangulo a los filtrados y buscamos en sus conexiones
        filtered_triangles.append(triangle_face_mesh.data)
        filtered_triangles, cantidad = mesh_filter_triangles_range(mesh, vertices, min_value, max_value,
                                                                triangle_face_mesh, filtered_triangles,
                                                                cantidad + 1)

    return filtered_triangles, cantidad


def mesh_filter_triangles_range(mesh, vertices, min_value, max_value, triangle_with_value, filtered_triangles, cantidad=0):
    """Filters triangles with the Z axis value

    :param mesh: a MeshBuilder object
    :param vertices: a list of vertexes
    :param min_value: min value in Z axis
    :param max_value: max value in Z axis
    :param triangle_with_value: a TriangleFaceMesh object that contains the value
    :param filtered_triangles: a list with the filtered triangles
    :param cantidad: triangles visited
    :return: a list of Triangle objects
    """

    # Revisamos que el triangulo este conectado con algun otro triangulo en cada uno de los lados: ab, bc y ca.
    filtered_triangles, cantidad = explore_side_range(mesh, vertices, min_value, max_value, triangle_with_value.ab, filtered_triangles, cantidad)
    filtered_triangles, cantidad = explore_side_range(mesh, vertices, min_value, max_value, triangle_with_value.bc, filtered_triangles, cantidad)
    filtered_triangles, cantidad = explore_side_range(mesh, vertices, min_value, max_value, triangle_with_value.ca, filtered_triangles, cantidad)

    return filtered_triangles, cantidad


def explore_side_value(mesh, vertices, value, triangle_face_mesh, filtered_triangles, cantidad):

    # Si no hay elemento, se termina la exploración por esta ruta
    if triangle_face_mesh == None:
        return filtered_triangles, cantidad

    # Si el triangulo ya lo teniamos entonces lo saltamos
    if triangle_face_mesh.data in filtered_triangles:
        return filtered_triangles, cantidad

    # Tomamos los valores en "Z" de sus vertices
    z_axis_from_a = vertices[triangle_face_mesh.data.a][2]
    z_axis_from_b = vertices[triangle_face_mesh.data.b][2]
    z_axis_from_c = vertices[triangle_face_mesh.data.c][2]

    z_min = min(z_axis_from_a, z_axis_from_b, z_axis_from_c)
    z_max = max(z_axis_from_a, z_axis_from_b, z_axis_from_c)

    # Vemos si el triángulo está en el rango que buscamos
    if (z_min <= value <= z_max) or\
        (z_min <= value <= z_max) or\
        (z_min <= value <= z_max):
        # Agregamos el triangulo a los filtrados y buscamos en sus conexiones
        filtered_triangles.append(triangle_face_mesh.data)
        filtered_triangles, cantidad = mesh_filter_triangles_value(mesh, vertices, value,
                                                                triangle_face_mesh, filtered_triangles,
                                                                cantidad + 1)

    return filtered_triangles, cantidad


def mesh_filter_triangles_value(mesh, vertices, value, triangle_with_value, filtered_triangles,
                          cantidad=0):
    """Filters triangles with the Z axis value

    :param mesh: a MeshBuilder object
    :param vertices: a list of vertexes
    :param min_value: min value in Z axis
    :param max_value: max value in Z axis
    :param triangle_with_value: a TriangleFaceMesh object that contains the value
    :param filtered_triangles: a list with the filtered triangles
    :param cantidad: triangles visited
    :return: a list of Triangle objects
    """

    # Revisamos que el triangulo este conectado con algun otro triangulo en cada uno de los lados: ab, bc y ca.
    filtered_triangles, cantidad = explore_side_value(mesh, vertices, value, triangle_with_value.ab, filtered_triangles, cantidad)
    filtered_triangles, cantidad = explore_side_value(mesh, vertices, value, triangle_with_value.bc, filtered_triangles, cantidad)
    filtered_triangles, cantidad = explore_side_value(mesh, vertices, value, triangle_with_value.ca, filtered_triangles, cantidad)

    return filtered_triangles, cantidad


"""
Pregunta 5: Dibujar los triangulos obtenidos.
Algunas funciones que se usaron para dibujar:
"""


def create_mesh(triangles):
    ## Creamos la malla con un meshBuilder
    mesh_builder = tm.TriangleFaceMeshBuilder()

    for triangle in triangles:
        mesh_builder.addTriangle(triangle)

    return mesh_builder


def draw_mesh(mesh, vertices):
    shape_indices = []
    shape_vertices = []
    # Creamos la lista con indices
    for triangle_mesh in mesh.getTriangleFaceMeshes():
        triangle = triangle_mesh.data
        shape_indices += [triangle.a, triangle.b, triangle.c]

    # Creamos la lista de vertices
    for vertice in vertices:
        shape_vertices += [vertice[0], vertice[1], vertice[2], 0, 0, 0]

    return bs.Shape(shape_vertices, shape_indices)


"""
Para ver la figura dibujada, correr el archivo draw_filtered_paraboloid.py

"""

"""
PROPUESTO!!! Por una decima DORADA!!!!!!!!

Haga una animación en donde una figura se vaya moviendo a través de los triangulos de la isolinea que se vio en el auxiliar, se debe realizar consultando los nodos vecinos.
Nota: El movimiento puede ser discreto, es decir, moverse a un nuevo triangulo cada cierto tiempo.

"""