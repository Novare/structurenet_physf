# compute_moi_util.py

from vis_utils import *
from matplotlib import pyplot as plt
from numpy.linalg import norm

import pyApproxMVBB as ap
import pymesh
import numpy as np
import random
import os
import sys
import time
import colorsys
import concurrent.futures
import pdb
import logging as log

def normalize(x):
    n = norm(x)
    if n > 0:
        return x / n
    else:
        log.error("Called normalize() with zero! x = {}, n = {}".format(str(x), str(n)))
        return None


def is_collinear(p0, p1, p2):
    e1 = normalize(p1 - p0)
    e2 = normalize(p2 - p0)
    return 1.0 - np.dot(e1, e2) < 1e-6


def get_triangle_normal(triangle):
    v0 = triangle[0]
    v1 = triangle[1]
    v2 = triangle[2]

    d1 = v1 - v0
    d2 = v2 - v0
    d1n = normalize(d1)
    d2n = normalize(d2)

    return normalize(np.cross(d1n, d2n))


# Based on https://en.wikipedia.org/wiki/Heron%27s_formula#Numerical_stability
def get_triangle_surface_area(triangle):
    edges = [norm(triangle[1] - triangle[0]), norm(triangle[2] - triangle[0]), norm(triangle[2] - triangle[1])]
    edges.sort()
    a = edges[2]
    b = edges[1]
    c = edges[0]

    # Sometimes the result of this dips below 0 ever so slightly, causing the whole thing to fail,
    # so we take care of this beforehand
    if (c - (a - b)) < 0.0:
        return 0.0

    val = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return 0.25 * np.sqrt(val)


# Based on https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up
def get_mesh_volume(mesh):
    volume = 0.0

    triangles = mesh.vertices[mesh.faces]
    for triangle in triangles:
        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]

        v321 = p3[0] * p2[1] * p1[2]
        v231 = p2[0] * p3[1] * p1[2]
        v312 = p3[0] * p1[1] * p2[2]
        v132 = p1[0] * p3[1] * p2[2]
        v213 = p2[0] * p1[1] * p3[2]
        v123 = p1[0] * p2[1] * p3[2]

        volume += (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)

    return np.abs(volume)


def get_mesh_centroid(mesh):
    area_sum = 0.0
    centroid = np.array([0.0, 0.0, 0.0])

    for triangle in mesh.vertices[mesh.faces]:
        center = np.array((triangle[0] + triangle[1] + triangle[2]) / 3.0)
        area = norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) / 2.0
        centroid += area * center
        area_sum += area

    centroid /= area_sum
    return centroid


def is_zero_edge_triangle(triangle):
    tol = 1e-6
    p0 = triangle[0]
    p1 = triangle[1]
    p2 = triangle[2]
    return norm(p1 - p0) < tol or norm(p2 - p0) < tol or norm(p2 - p1) < tol

def clean_up_mesh(mesh, tol):
    new_mesh, _ = pymesh.remove_isolated_vertices(mesh)
    new_mesh, _ = pymesh.remove_duplicated_vertices(new_mesh)
    new_mesh, _ = pymesh.remove_duplicated_faces(new_mesh)
    new_mesh, _ = pymesh.remove_degenerated_triangles(new_mesh)
    new_mesh, _ = pymesh.collapse_short_edges(new_mesh, rel_threshold=0.2)
    return new_mesh

# Generate a random, bright color. Useful for plots that need to pop
def random_color():
    h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
    r, g, b = [i for i in colorsys.hls_to_rgb(h, l, s)]
    return (r, g, b)


def draw_line(ax, p1, p2, color):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color)


def draw_pymesh(ax, mesh, color):
    tri = a3.art3d.Poly3DCollection(mesh.vertices[mesh.faces])
    tri.set_color((0.0, 0.0, 0.0, 0.0))
    tri.set_edgecolor(color)
    ax.add_collection3d(tri)


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


def draw_cut_volumes(ax, cut_volumes):
    for (cv, _, _) in cut_volumes:
        fill_pymesh(ax, cv, random_color())


def draw_cut_meshes(ax, cut_meshes):
    for mesh in cut_meshes:
        draw_pymesh(ax, mesh, random_color())


def draw_contact_surfaces(ax, contact_surfaces):
    for (i, j, triangles) in contact_surfaces:
        c = random_color()
        for triangle in triangles:
            fill_pymesh(ax=ax, mesh=pymesh.form_mesh(vertices=triangle, faces=np.array([np.array([0, 1, 2])])), color=c)
