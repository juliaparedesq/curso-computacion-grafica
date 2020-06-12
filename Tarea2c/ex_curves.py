# coding=utf-8
"""
Daniel Calderon
Universidad de Chile, CC3501, 2019-2
Hermite and Bezier curves using python, numpy and matplotlib
"""


import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D


def generateT(t):
    return np.array([[1, t, t**2, t**3]]).T


def hermiteMatrix(P1, P2, T1, T2):
    
    # Generate a matrix concatenating the columns
    G = np.concatenate((P1, P2, T1, T2), axis=1)

    # Hermite base matrix is a constant
    Mh = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])    
    
    return np.matmul(G, Mh)

def SplinesMatrix(P0,P1, P2, P3):

    G=np.concatenate((P0, P1, P2, P3), axis=1)

    Ms = 0.5 * np.array([[0, -1, 2, -1], [2, 0, -5, 3], [0, 1, 4, -3], [0, 0, -1, 1]])

    return np.matmul(G, Ms)

def bezierMatrix(P0, P1, P2, P3):
    
    # Generate a matrix concatenating the columns
    G = np.concatenate((P0, P1, P2, P3), axis=1)

    # Bezier base matrix is a constant
    Mb = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    
    return np.matmul(G, Mb)


def plotCurve(ax, curve, label, color=(0,0,1)):
    
    xs = curve[:, 0]
    ys = curve[:, 1]
    zs = curve[:, 2]
    
    ax.plot(xs, ys, zs, label=label, color=color)
    

# M is the cubic curve matrix, N is the number of samples between 0 and 1
def evalCurve(M, N):
    # The parameter t should move between 0 and 1
    ts = np.linspace(0.0, 1.0, N)
    
    # The computed value in R3 for each sample will be stored here
    curve = np.ndarray(shape=(N, 3), dtype=float)
    
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T
        
    return curve
def CR(p1,p2,p3,p4,p5):
    G1=SplinesMatrix(p1,p2,p3,p4)
    N=50
    C1 = evalCurve(G1,N)

    G2=SplinesMatrix(p2,p3,p4,p5)
    N=50
    C2 = evalCurve(G2,N)
    C2= np.delete(C2, 0, 0)

    C= np.append(C1, C2, 0)
    return C


if __name__ == "__main__":
    
    """
    Example for Hermite curve
    """
    
    P1 = np.array([[0, 0, 1]]).T  #.T hace el vector transpuesto!!!!!!!
    P2 = np.array([[1, 0, 0]]).T
    T1 = np.array([[10, 0, 0]]).T
    T2 = np.array([[0, 10, 0]]).T
    
    GMh = hermiteMatrix(P1, P2, T1, T2)

    # Number of samples to plot
    N = 50

    hermiteCurve = evalCurve(GMh, N)


    """
    Example for Bezier curve
    """
    
    R0 = np.array([[0, 0, 1]]).T
    R1 = np.array([[0, 1, 0]]).T
    R2 = np.array([[1, 0, 1]]).T
    R3 = np.array([[1, 1, 0]]).T
    
    GMb = bezierMatrix(R0, R1, R2, R3)
    bezierCurve = evalCurve(GMb, N)

    """
    Example for CR curve
    """

    Q0 = np.array([[0, -1, 0]]).T
    Q1 = np.array([[0.25, -0.5, 1]]).T
    Q2 = np.array([[0.5, 0, 0.8]]).T
    Q3 = np.array([[0.75, 0.5, 1]]).T
    Q4 = np.array([[1, 1, 0.8]]).T

    GMb = SplinesMatrix(Q1, Q2, Q3, Q4)
    GMb1 = SplinesMatrix(Q0, Q1, Q2, Q3)
    scCurve = evalCurve(GMb, N)
    scCurve1 = evalCurve(GMb1, N)
    print(scCurve)
    #print([scCurve1] + [scCurve] )
    sisi = CR(Q0,Q1,Q2,Q3,Q4)




    controlPoints = np.concatenate((R0, R1, R2, R3), axis=1)


    # Setting up the matplotlib display for 3D
    fig = mpl.figure()
    ax = fig.gca(projection='3d')

    # Plotting the Hermite curve
    plotCurve(ax, hermiteCurve, "Hermite curve", color=(1, 0, 0))
    # Plotting the Bezier curve
    plotCurve(ax, bezierCurve, "Bezier curve", color=(0, 0, 1))


    plotCurve(ax, scCurve, "CR curve", color=(0, 1, 0))
    plotCurve(ax, scCurve1, "CR curve1", color=(0, 1, 1))
    plotCurve(ax, sisi, "CR sisi", color=(1, 0.5, 0.5))


    

    # Adding a visualization of the control points
    ax.scatter(controlPoints[0, :], controlPoints[1, :], controlPoints[2, :], color=(1, 0, 0))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.legend()
    mpl.show()