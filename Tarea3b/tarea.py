import numpy as np
import json
import matplotlib.pyplot as mpl
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import sys

"""Hotel=sys.argv[1]
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
    windows = hotel['windows']"""


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
# Problem setup
HH = P+D+2*W
#H=4
WW = 5*L+ 6*W
#W=3
h = 0.1
# Boundary Dirichlet Conditions:


def finnite_differences():
    # Problem setup
    # H
    # W
    # Boundary Dirichlet Conditions:
    # TOP = 100
    # BOTTOM = 5
    # LEFT = 0
    # RIGHT = 0

    # Number of unknowns
    # left, bottom and top sides are known (Dirichlet condition)
    # right side is unknown (Neumann condition)
    nh = int(WW / h) +1
    nv = int(HH/ h) + 1


    # In this case, the domain is just a rectangle
    N = nh * nv

    # We define a function to convert the indices from i,j to k and viceversa
    # i,j indexes the discrete domain in 2D.
    # k parametrize those i,j, this way we can tidy the unknowns
    # in a column vector and use the standard algebra

    def getK(i, j):
        return j * nh + i

    def getIJ(k):
        i = k % nh
        j = k // nh
        return (i, j)
    #print(getIJ(2540), getIJ(2560))

    # In this matrix we will write all the coefficients of the unknowns
    A = np.zeros((N, N))

    # In this vector we will write all the right side of the equations
    b = np.zeros((N,))

    # Note: To write an equation is equivalent to write a row in the matrix system

    # We iterate over each point inside the domain
    # Each point has an equation associated
    # The equation is different depending on the point location inside the domain
    for i in range(0, nh): #i e [0, 206]              ie range(0,207)
        for j in range(0, nv):  # j  e [0, 62]        ie range (0, 63)


            # We will write the equation associated with row k
            k = getK(i, j)

            # We obtain indices of the other coefficients
            k_up = getK(i, j + 1)
            k_down = getK(i, j - 1)
            k_left = getK(i - 1, j)
            k_right = getK(i + 1, j)
            #MAX NH ES 59, MAX HV ES 199
            # Depending on the location of the point, the equation is different
            if nh - 1 - round(W / h) <= i <= nh - 1 and 0<=j<=round(W/h):
                A[k, k_up] = 1
                A[k, k_left] = 1
                A[k, k] = -2
                b[k] = 0

            elif 0<=j < round(W/h):
                if (round(W/h) < i < nh-1 - round(W/h)):
                    # neumann nula
                    B = 0

                    A[k, k_up] = 1
                    A[k, k] = -1
                    b[k] = 0 #-2 * h * B
                elif 0<=i<=round(W/h) :
                    A[k, k_up] = 1
                    A[k, k_right] = 1
                    A[k, k] = -2
                    b[k] = 0


            elif j== round(W/h):
                if round(H1/h) <= i <= round((H1+H2)/h):

                    # neumann heater
                    B = heater_power

                    A[k, k_up] = 2
                    A[k, k_left] = 1
                    A[k, k_right] = 1
                    A[k, k] = -4
                    b[k] = -2 * h * B

                elif (round(W/h)< i < round(H1/h)) or  ( round((H1+H2)/h) < i < nh-1 - round(W/h) ):
                    # neumann nula
                    B = 0
                    A[k, k_up] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B

                elif 0 <= i <= round(W / h):
                    A[k, k_up] = 1
                    A[k, k_right] = 1
                    A[k, k] = -2
                    b[k] = 0

            elif 0<=i<round(W/h) and round(W/h)<j<=nv-1:
                A[k, k_right] = 1
                A[k, k] = -1
                b[k] = 0  # -2 * h * B
            elif i==round(W/h) and (round(W/h)<j<round((W+P)/h) or round((W+P)/h)<j<round((2*W+P)/h) or round((2*W+P)/h)<j<=nv-1):
                A[k, k_right] = 1
                A[k, k] = -1
                b[k] = 0  # -2 * h * B
            elif nh-1 - round(W/h) <= i <= nh-1 and round(W/h)<j<=nv-1:
                A[k, k_left] = 1
                A[k, k] = -1
                b[k] = 0  # -2 * h * B

            elif (round(W/h) < j < round((W+P)/h)) and ( round(W/h) < i < nh-1 - round(W/h) ):
                # normal
                B = 0

                A[k, k_up] = 1
                A[k, k_down] = 1
                A[k, k_left] = 1
                A[k, k_right] = 1
                A[k, k] = -4
                b[k] = 0 # -2 * h * B

            elif j == round((W+P)/h):
                if round((W)/h) <i < round((L+W-E)/h) or round((L+W)/h) < i < round((2*(L+W)-E)/h) or round((2*(L+W))/h) < i < round((3*(L+W)-E)/h) or round((3*(L+W))/h) < i < round((4*(L+W)-E)/h) or round((4*(L+W))/h) < i < round((5*(L+W)-E)/h):
                    # neumann borde de abajo muro
                    B = 0
                    A[k, k_down] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B
                elif round((1*(L+W)-E)/h) < i < round(1*(L+W)/h) or round((2*(L+W)-E)/h) < i < round(2*(L+W)/h) or round((3*(L+W)-E)/h) < i < round(3*(L+W)/h) or round((4*(L+W)-E)/h) < i < round(4*(L+W)/h) or round((5*(L+W)-E)/h) < i < nh-1 -round(W/h):
                    # normal
                    B = 0
                    A[k, k_up] = 1
                    A[k, k_down] = 1
                    A[k, k_left] = 1
                    A[k, k_right] = 1
                    A[k, k] = -4
                    b[k] = 0  # -2 * h * B
                elif i == round(W/h) or i == round((L+W-E)/h) or i == round((2*(L+W)-E)/h) or i == round((3*(L+W)-E)/h) or i == round((4*(L+W)-E)/h) or i == round((5*(L+W)-E)/h):
                    # esquina muro pieza abajo a la derecha
                    B = 0
                    A[k, k_down] = 1
                    A[k, k_right] = 1
                    A[k, k] = -2
                    b[k] = 0  # -2 * h * B
                elif i== round(1*(L+W)/h) or i== round(2*(L+W)/h) or i== round(3*(L+W)/h) or i== round(4*(L+W)/h):
                    # esquina muro pieza abajo a la izq
                    #print(int((L+W)/h), 'no', int((L+2*W)/h), 'L+W=', L+W, 'L+2W= ',L+2*W)
                    #print(int((P+W)/h),  int((P+2*W)/h),  'P+W=', P+W, ' P+2W= ', P+2*W)
                    #print(round(3*(L+W)/h),  round(3*(L/h + W/h)))
                    B = 0
                    A[k, k_down] = 1
                    A[k, k_left] = 1
                    A[k, k] = -2
                    b[k] = 0  # -2 * h * B
            elif round((P+W)/h) < j < round((P+2*W)/h):  #NO HAY DENTRO DEL MURO PQ EL MURO ES 11, UNA SOLA LINEA
                if round((W) / h) < i < round((L + W - E) / h) or round((L + W) / h) < i < round((2 * (L + W) - E) / h) or round(
                    (2 * (L + W)) / h) < i < round((3 * (L + W) - E) / h) or round((3 * (L + W)) / h) < i < round(
                    (4 * (L + W) - E) / h) or round((4 * (L + W)) / h) < i < round((5 * (L + W) - E) / h):
                    # dentro del muro #ESTRELLA #NO ENTRAAAA
                    #print(' i= ', i, ' j= ', j)
                    B = 0
                    A[k, k_up] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B
                elif round((1 * (L + W) - E) / h) < i < round(1 * (L + W) / h) or round((2 * (L + W) - E) / h) < i < round(
                    2 * (L + W) / h) or round((3 * (L + W) - E) / h) < i < round(3 * (L + W) / h) or round(
                    (4 * (L + W) - E) / h) < i < round(4 * (L + W) / h) or round(
                    (5 * (L + W) - E) / h) < i < nh - 1 - round(W / h):
                    #print(' i= ', i, ' j= ', j) NO ENTRAA
                    # normal #CORAZON
                    B = 0
                    A[k, k_up] = 1
                    A[k, k_down] = 1
                    A[k, k_left] = 1
                    A[k, k_right] = 1
                    A[k, k] = -4
                    b[k] = 0  # -2 * h * B
                elif i == round((L+W-E)/h) or i == round((2*(L+W)-E)/h) or i == round((3*(L+W)-E)/h) or i == round((4*(L+W)-E)/h) or i == round((5*(L+W)-E)/h):
                    #print(' i= ', i, ' j= ', j) #NO ENTRA
                    # ancho de muro #O
                    B = 0
                    A[k, k_right] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B
                elif i== round(1*(L+W)/h) or i== round(2*(L+W)/h) or i== round(3*(L+W)/h) or i== round(4*(L+W)/h):
                    #print(' i= ', i, ' j= ', j) #NO ENTRA
                    # ancho muro #TRIANGULO
                    B = 0
                    A[k, k_left] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B
            elif j==round((P+2*W)/h): #12
                if round((W) / h) < i < round((L + W - E) / h) or round((L + 2*W) / h) < i < round((2 * (L + W) - E) / h) or round(
                        ((2 *L + 3*W)) / h) < i < round((3 *(L + W) - E) / h) or round(((3 *L + 4*W)) / h) < i < round(
                    (4 * (L + W) - E) / h) or round(((4 * L + 5*W)) / h) < i < round((5 * (L + W) - E) / h):
                    #pared del muro por la pieza #casi ESTRELLA
                    B = 0
                    A[k, k_up] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B
                elif round((1 * (L + W) - E) / h) < i < round(1 * (L + W) / h) or round((2 * (L + W) - E) / h) < i < round(
                    2 * (L + W) / h) or round((3 * (L + W) - E) / h) < i < round(3 * (L + W) / h) or round(
                    (4 * (L + W) - E) / h) < i < round(4 * (L + W) / h) or round(
                    (5 * (L + W) - E) / h) < i < nh - 1 - round(W / h):
                    # normal #CORAZON
                    B = 0
                    A[k, k_up] = 1
                    A[k, k_down] = 1
                    A[k, k_left] = 1
                    A[k, k_right] = 1
                    A[k, k] = -4
                    b[k] = 0  # -2 * h * B
                elif round((L+W)/h) <= i < round((L+2*W)/h) or round(2*(L+W)/h) <= i < round((2*L+3*W)/h) or round(3*(L+W)/h) <= i < round((3*L+4*W)/h) or round(4*(L+W)/h) <= i < round((4*L+5*W)/h):
                    # ancho muro #TRIANGULO y más
                    B = 0
                    A[k, k_left] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B
                elif i == round((L+W-E)/h) or i == round((2*(L+W)-E)/h) or i == round((3*(L+W)-E)/h) or i == round((4*(L+W)-E)/h) or i == round((5*(L+W)-E)/h) or i==round(W/h) or i == round((L+2*W)/h) or i == round((2*L+3*W)/h) or i == round((3*L+4*W)/h) or i == round((4*L+5*W)/h):
                    A[k,k]=-2
                    A[k, k_right]=1
                    A[k, k_up] = 1
                    b[k] = 0
            elif round((P+2*W)/h) < j < nv-2:
                if round(W/h)<i<round((L+W)/h) or round((L+2*W)/h)<i<round(2*(L+W)/h) or round((2*L+3*W)/h)<i<round(3*(L+W)/h) or round((3*L+4*W)/h)<i<round(4*(L+W)/h) or round((4*L+5*W)/h)<i< nh-1 - round(W/h):
                    # normal
                    B = 0
                    A[k, k_up] = 1
                    A[k, k_down] = 1
                    A[k, k_left] = 1
                    A[k, k_right] = 1
                    A[k, k] = -4
                    b[k] = 0  # -2 * h * B
                elif i == round((L+2*W)/h) or i == round((2*L+3*W)/h) or i == round((3*L+4*W)/h) or i == round((4*L+5*W)/h):
                    #SOMBREADO
                    A[k,k]=-1
                    A[k, k_right]= 1
                    b[k]=0
                elif round((L+W)/h) <= i < round((L+2*W)/h) or round(2*(L+W)/h) <= i < round((2*L+3*W)/h) or round(3*(L+W)/h) <= i < round((3*L+4*W)/h) or round(4*(L+W)/h) <= i < round((4*L+5*W)/h):
                    #         #TRIANGULO y más
                    B = 0
                    A[k, k_left] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B
            elif j== nv-2:
                if round(W/h)<i<round((L+W)/h) or round((L+2*W)/h)<i<round(2*(L+W)/h) or round((2*L+3*W)/h)<i<round(3*(L+W)/h) or round((3*L+4*W)/h)<i<round(4*(L+W)/h) or round((4*L+5*W)/h)<i< nh-1 - round(W/h):
                    for r in range(len(windows)): #0,1,2,3,4
                        e=(r*L + (1+r)*W)
                        if round(e/h)<i<round((1+r)*(L+W)/h):
                            if windows[r] == 0:  # la ventana esta abierta --> Dirichlet

                                A[k, k_down] = 1
                                A[k, k_left] = 1
                                A[k, k_right] = 1
                                A[k, k] = -4
                                b[k] = -ambient_temperature
                            elif windows[r] == 1:  # ventana cerrrada --> Neumann
                                # normal pq estamos en j= nv-2, en j=nv-1 se aplica
                                A[k, k_up] = 1
                                A[k, k_down] = 1
                                A[k, k_left] = 1
                                A[k, k_right] = 1
                                A[k, k] = -4
                                b[k] = 0  # -2 * h * B

                elif i == round((L+2*W)/h) or i == round((2*L+3*W)/h) or i == round((3*L+4*W)/h) or i == round((4*L+5*W)/h):

                    #SOMBREADO
                    A[k,k]=-1
                    A[k, k_right]= 1
                    b[k]=0
                elif round((L+W)/h) <= i < round((L+2*W)/h) or round(2*(L+W)/h) <= i < round((2*L+3*W)/h) or round(3*(L+W)/h) <= i < round((3*L+4*W)/h) or round(4*(L+W)/h) <= i < round((4*L+5*W)/h):
                    #         #TRIANGULO y más
                    B = 0
                    A[k, k_left] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B
            elif j==nv-1:
                if round(W / h) < i < round((L + W) / h) or round((L + 2 * W) / h) < i < round(2 * (L + W) / h) or round(
                        (2 * L + 3 * W) / h) < i < round(3 * (L + W) / h) or round((3 * L + 4 * W) / h) < i < round(
                        4 * (L + W) / h) or round((4 * L + 5 * W) / h) < i < nh - 1 - round(W / h):
                    #print(j) #hay un error acá. No entra a este if
                    for r in range(len(windows)):  # 0,1,2,3,4
                        #print(r)
                        e = (r * L + (1 + r) * W)
                        if round(e / h) < i < round((1 + r)*(L + W) / h):
                            if windows[r] == 0:  # la ventana esta abierta --> Dirichlet
                                A[k, k] = 1
                                b[k] = ambient_temperature
                            elif windows[r] == 1:  # ventana cerrrada --> Neumann
                                A[k, k_down] = 2
                                A[k, k_left] = 1
                                A[k, k_right] = 1
                                A[k, k] = -4
                                b[k] = -2 * h * window_loss
                elif i == round((L+2*W)/h) or i == round((2*L+3*W)/h) or i == round((3*L+4*W)/h) or i == round((4*L+5*W)/h):
                    #SOMBREADO
                    A[k,k]=-1
                    A[k, k_right]= 1
                    b[k]=0
                elif round((L+W)/h) <= i < round((L+2*W)/h) or round(2*(L+W)/h) <= i < round((2*L+3*W)/h) or round(3*(L+W)/h) <= i < round((3*L+4*W)/h) or round(4*(L+W)/h) <= i < round((4*L+5*W)/h):
                    #         #TRIANGULO y más
                    B = 0
                    A[k, k_left] = 1
                    A[k, k] = -1
                    b[k] = 0  # -2 * h * B

            else:

                print("Point (" + str(i) + ", " + str(j) + ") missed!")
                print("Associated point index is " + str(k))
                raise Exception()

    mpl.spy(A)
    A=csr_matrix(A)
    # A quick view of a sparse matrix


    # Solving our system
    x = spsolve(A, b)


    # Now we return our solution to the 2d discrete domain
    # In this matrix we will store the solution in the 2d domain
    u = np.zeros((nh, nv))

    for k in range(0, N):
        i, j = getIJ(k)
        u[i, j] = x[k]

    # Adding the borders, as they have known values
    #print(u.shape)

    # this visualization locates the (0,0) at the lower left corner
    # given all the references used in this example.
    fig, ax = mpl.subplots(1, 1)
    pcm = ax.pcolormesh(u.T, cmap='RdBu_r')
    fig.colorbar(pcm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hotel')
    ax.set_aspect('equal', 'datalim')

    # Note:
    # imshow is also valid but it uses another coordinate system,
    # a data transformation is required
    # ax.imshow(ub.T)
    mpl.show()
    np.save('suelo', u)
    return u


finnite_differences()

def calculate_gradient_forward(V, hx=h, hy=h):
    dx = np.zeros(shape=V.shape)
    dy = np.zeros(shape=V.shape)
    for i in range(1, V.shape[0]-1):
        for j in range(1, V.shape[1]-1):
            # Filtering results where potetial is zero, so we have a better result
            if V[i, j] and V[i+1, j] and V[i+1, j+1] and V[i, j+1] and V[i-1, j+1] and V[i-1, j] and V[i-1, j-1] and V[i, j-1] and V[i+1, j-1] :
                dx[i, j] = V[i+1, j] - V[i, j]
                dy[i, j] = V[i, j+1] - V[i, j]

    # Explodes in points where the other part is zero
    dx = dx / hx
    dy = dy / hy

    return dx, dy

dx_earth, dy_earth = calculate_gradient_forward(np.load('suelo.npy'))
"""dx_moon, dy_moon = calculate_gradient_forward(pot_moon)

dx_total = dx_earth + dx_moon
dy_total = dy_earth + dy_moon

# We still have to change some values to 0, where is inside of the earth or the sun
dx_total[np.where(dx_earth == 0)] = 0
dy_total[np.where(dy_earth == 0)] = 0

dx_total[np.where(dx_moon == 0)] = 0
dy_total[np.where(dy_moon == 0)] = 0

# Store arrays
np.save('pt2_pot_earth', pot_earth)
np.save('pt2_pot_earth_dx', dx_earth)
np.save('pt2_pot_earth_dy', dy_earth)

np.save('pt2_pot_moon', pot_moon)
np.save('pt2_pot_moon_dx', dx_moon)
np.save('pt2_pot_moon_dy', dy_moon)"""
"""Pt.3 Plotting results with quiver earth"""
fig, ax = plt.subplots(figsize = (15, 10))
# Show earth
#earth_visualization = ax.scatter(earth.get_pos()[0], earth.get_pos()[1], color=(0,0.5,0.5), s=earth.get_radius()*1e5, label="Earth")

X, Y = np.mgrid[0:dx_earth.shape[0], 0:dx_earth.shape[1]]

#print(dx_earth.shape, dy_earth.shape)
Q = ax.quiver(X, Y, dx_earth, dy_earth)

ax.set_title('Quiver using forward gradient')
#ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()