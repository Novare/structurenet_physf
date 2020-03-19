# compute_moi_util.py

from vis_utils import *
from matplotlib import pyplot as plt
from numpy.linalg import norm

import pyApproxMVBB as ap
import pymesh
import numpy as np
import random
import os
import colorsys
import concurrent.futures

SHOULD_PRINT_DEBUG = False
SHOULD_PRINT_INFO = True
SHOULD_PRINT_ERROR = True

def PRINT_DEBUG(x):
    if SHOULD_PRINT_DEBUG:
        print(x)

def PRINT_INFO(x):
    if SHOULD_PRINT_INFO:
        print(x)

def PRINT_ERROR(x):
    if SHOULD_PRINT_ERROR:
        print(x)

def normalize(x):
    n = norm(x)
    if n > 0:
        return x / n
    else:
        PRINT_ERROR("Called normalize() with zero! x = " + str(x) + ", n = " + str(n))
        return x

def get_triangle_normal(v0, v1, v2):
    d1 = v1 - v0
    d2 = v2 - v0
    return normalize(np.cross(normalize(d1), normalize(d2)))

def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def random_color():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]
    return (r, g, b)

def draw_line(ax, p1, p2, color):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color)
    
def draw_pymesh(ax, mesh, color):
    for face in mesh.faces:
        for i in range(len(face)):
            i1 = 0
            i2 = 0
            if i == len(face) - 1:
                i1 = face[i]
                i2 = face[0]
            else:
                i1 = face[i]
                i2 = face[i+1]
            draw_line(ax=ax, p1=mesh.vertices[i1], p2=mesh.vertices[i2], color=color)
                
def fill_pymesh(ax, mesh, color):
    tri = a3.art3d.Poly3DCollection(mesh.vertices[mesh.faces])
    tri.set_color(color)
    tri.set_edgecolor(color)
    ax.add_collection3d(tri)

def draw_oobbs(ax, cps):
    for cp in cps:
        draw_oobb(ax, cp)

def draw_oobb(ax, cp):
    boxcolor = random_color()
    ax.plot([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], [cp[0][2], cp[1][2]], c=boxcolor)
    ax.plot([cp[0][0], cp[2][0]], [cp[0][1], cp[2][1]], [cp[0][2], cp[2][2]], c=boxcolor)
    ax.plot([cp[1][0], cp[3][0]], [cp[1][1], cp[3][1]], [cp[1][2], cp[3][2]], c=boxcolor)
    ax.plot([cp[2][0], cp[3][0]], [cp[2][1], cp[3][1]], [cp[2][2], cp[3][2]], c=boxcolor)
    ax.plot([cp[4][0], cp[5][0]], [cp[4][1], cp[5][1]], [cp[4][2], cp[5][2]], c=boxcolor)
    ax.plot([cp[4][0], cp[6][0]], [cp[4][1], cp[6][1]], [cp[4][2], cp[6][2]], c=boxcolor)
    ax.plot([cp[5][0], cp[7][0]], [cp[5][1], cp[7][1]], [cp[5][2], cp[7][2]], c=boxcolor)
    ax.plot([cp[6][0], cp[7][0]], [cp[6][1], cp[7][1]], [cp[6][2], cp[7][2]], c=boxcolor)
    ax.plot([cp[0][0], cp[4][0]], [cp[0][1], cp[4][1]], [cp[0][2], cp[4][2]], c=boxcolor)
    ax.plot([cp[1][0], cp[5][0]], [cp[1][1], cp[5][1]], [cp[1][2], cp[5][2]], c=boxcolor)
    ax.plot([cp[2][0], cp[6][0]], [cp[2][1], cp[6][1]], [cp[2][2], cp[6][2]], c=boxcolor)
    ax.plot([cp[3][0], cp[7][0]], [cp[3][1], cp[7][1]], [cp[3][2], cp[7][2]], c=boxcolor)

def draw_cutvolumes(ax, cutvolumes):
    for (cv, _, _) in cutvolumes:
        fill_pymesh(ax, cv, random_color())

def draw_contact_surfaces(ax, contact_surfaces):
    for i, vertices_list in enumerate(contact_surfaces):
        if len(vertices_list) == 0:
            PRINT_DEBUG("Skipped the contact surfaces of cutvolume " + str(i))
            continue
        
        c = random_color()
        for j, vertices in enumerate(vertices_list): 
            mesh = pymesh.form_mesh(vertices=vertices, faces=np.array([[0, 1, 2]]))
            fill_pymesh(ax=ax, mesh=mesh, color=c)
