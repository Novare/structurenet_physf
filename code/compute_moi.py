# compute_moi.py

from matplotlib import pyplot as plt

import pyApproxMVBB as ap
import pymesh
import numpy as np
import scipy as sp
import random
import os
import pdb
import time
import colorsys
import concurrent.futures

from compute_moi_util import *

# Convert a pointcloud to OOBBs using pyApproxMVBB
def pointcloud_to_oobbs(pointcloud):
    pcl = np.transpose(np.float64(np.array(pointcloud)))       
    oobb = ap.approximateMVBB(pts=pcl,
                              epsilon=0.001,
                              pointSamples=500,
                              gridSize=5,
                              mvbbDiamOptLoops=0,
                              mvbbGridSearchOptLoops=5,
                              seed=1234)

    cps = oobb.getCornerPoints()

    # This rotation matrix is necessary to make the results look nice
    # in a plot and is not strictly necessary.
    coord_rot = np.matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    for i, point in enumerate(cps):
        cps[i] = np.asarray(coord_rot * np.array(point).reshape(-1, 1)).reshape(-1)
    return cps

# Convert the OOBBs to mesh structures using PyMesh, then try intersecting each unique
# pair of meshes. If the intersection is not empty, save the intersection and the adjacency
# relationship, then cut out the intersection from one of the meshes. In the end, the parts
# should not intersect each other anymore. While cutting out the intersections, save all
# faces from the cut mesh that belong to the other mesh as contact faces.
def cut_out_intersections(oobbs):
    meshes = []
    for cps in oobbs:
        # Vertices are in order:
        # [0] = (0, 0, 0)
        # [1] = (1, 0, 0)
        # [2] = (0, 1, 0)
        # [3] = (1, 1, 0)
        # [4] = (0, 0, 1)
        # [5] = (1, 0, 1)
        # [6] = (0, 1, 1)
        # [7] = (1, 1, 1)
        
        v = cps
        f = np.array([
            [0, 2, 3], [0, 3, 1], # front
            [5, 7, 6], [5, 6, 4], # back
            [4, 6, 2], [4, 2, 0], # left
            [1, 3, 7], [1, 7, 5], # right
            [2, 6, 7], [2, 7, 3], # top
            [4, 0, 1], [4, 1, 5]  # bottom
        ])
        meshes.append(pymesh.form_mesh(vertices=v, faces=f))

    unique_pairs = []
    for i in range(len(meshes)):
        for j in range(len(meshes)):
            if not i == j and not (i, j) in unique_pairs and not (j, i) in unique_pairs:
                unique_pairs.append((i, j))
    
    contact_triangles = []
    cut_volumes = []
    neighbours = []
    for (i, j) in unique_pairs:
        cvol = pymesh.boolean(meshes[i], meshes[j], operation="intersection", engine="igl")
        if len(cvol.vertices) == 0:
            continue
        cut_volumes.append((cvol, i, j))
        neighbours.append((i, j))

        # Cut the intersection out of one of the meshes
        meshes[i] = pymesh.boolean(meshes[i], cvol, operation="difference", engine="igl")

        sources = meshes[i].get_attribute("source")
        triangles = []
        for k, face in enumerate(meshes[i].faces):
            if sources[k] == 0:
                triangles.append(meshes[i].vertices[face])
        contact_triangles.append((i, j, triangles))

    return meshes, cut_volumes, neighbours, contact_triangles

def build_Aeq_submatrix(mesh, interface):
    c = get_mesh_centroid(mesh)
    vertex_count = 3 * (interface.vertices[interface.faces])
    Ajk = np.zeros((6, 4 * vertex_count))

    start_index = 0
    for t, triangle in interface.vertices[interface.faces]:
        n = get_triangle_normal(triangle)
        t1 = normalize(triangle[1] - triangle[0])
        t2 = cross(n, t1)

        for vt, vertex in triangle:
            v = normalize(c - vertex)
            nv = cross(n, v)
            t1v = cross(t1, v)
            t2v = cross(t2, v)
 
            Ajk[0:3, start_index + 0] = n
            Ajk[0:3, start_index + 1] = -n
            Ajk[0:3, start_index + 2] = t1
            Ajk[0:3, start_index + 3] = t2 
            Ajk[3:6, start_index + 0] = nv
            Ajk[3:6, start_index + 1] = -nv
            Ajk[3:6, start_index + 2] = t1v
            Ajk[3:6, start_index + 3] = t2v
            start_index += 3
    return Ajk

# TODO: What about the ground mesh? Remove it? Leave it?
def build_constraint_elements(cut_meshes, contact_surfaces):
    # Build A_eq from contact surfaces
    part_count = len(cut_meshes)
    A_eq_rowcount = 6 * part_count
    A_eq_columncount = 0
    for k, cs in enumerate(contact_surfaces):
        si = cs[0]
        sj = cs[1]
        surface_count = len(cs[2])

        # Every contact surface results in two submatrices, one per part
        submatrices.append((si, k, build_Aeq_submatrix(cut_meshes[si], cs[2])))
        submatrices.append((sj, k, build_Aeq_submatrix(cut_meshes[sj], cs[2]))) 
        A_eq_columncount += np.size(submatrices[-1])[1]
    A_eq = np.zeros((A_eq_rowcount, A_eq_columncount)) 

    column_offset = 0
    for k in range(len(contact_surfaces)):
        column_offset_new = 0
        for j in range(len(cut_meshes)):
            fitting_submatrices = [sm for sm in submatrices if sm[0] == j and sm[1] == k]
            if len(fitting_submatrices) == 0:
                continue
            else:
                sm = fitting_submatrices[0]
                column_offset_new = column_offset + np.size(sm)[1]
                A_eq[j:j+6, column_offset:column_offset_new] = sm
        column_offset = column_offset_new

    # Build w from cut_meshes
    w = np.array([])
    for i, mesh in enumerate(cut_meshes):
        weight = get_mesh_volume(mesh)
        w.append(np.array([0, 0, -weight, 0, 0, 0]))

    # Build A_fr from contact surfaces
    alpha = 0.7
    A_fr_sub = np.zeros((2, 4))
    A_fr_sub[0,0] = -alpha
    A_fr_sub[1,0] = -alpha
    A_fr_sub[0,1] = 1
    A_fr_sub[1,2] = 1

    A_fr_column_count = np.size(A_eq)[1]
    A_fr_row_count = A_fr_column_count / 2
    A_fr = np.zeros((A_fr_row_count, A_fr_column_count))
    for i in range(A_fr_size / 4):
        A_fr[i*4:i*4+4,i*4:i*4+4] = A_fr_sub

    return A_eq, w, A_fr

# This assumes that f is made up of vectors containing 4 scalars,
# f_n+, f_n-, f_t1, f_t2 respectively
def f_func(f):
    return np.sum(np.square(f[1::4]))

def f_derv(f):
    return 2 * np.sum(f[1::4])

def optimize_f(A_eq, w, A_fr):
    f_dim = np.size(A_eq)[1]
    f0 = np.random.rand(f_dim)

    constr_static = {
        "type": "eq",  
        "fun": lambda f, A_eq=A_eq, w=w: np.dot(A_eq, f) + w
        "jac": lambda f, A_eq=A_eq, w=w: A_eq
    } 

    constr_fric = {
        "type": "ineq",
        "fun": lambda f, A_fr=A_fr: -np.dot(A_fr, f)
        "jac": lambda f, A_fr=A_fr: -A_fr
    }

    constr_pos_compr = {
        "type": "ineq",
        "fun": lambda f: f[0::4]
        "jac": lambda f: 0
    }

    constr_neg_compr = {
        "type": "ineq",
        "fun": lambda f: f[1::4]
        "jac": lambda f: 0
    }

    res = sp.minimize(f_func, f0, method="SLSQP", jac=f_derv, constraints=[constr_static, constr_pos_compr, constr_neg_compr, constr_fric], options={"ftol": 1e-9, "disp": True})
    moi = f_func(res.x)
    return moi

# Calculate the measure of infeasibility using the other functions defined
# here and in the utility file.
def calculate_measure_of_infeasibility(obj):
    # Get the parts and their relationships from the graph
    part_boxes, part_geos, edges, part_ids, part_sems = obj.graph(leafs_only=True)

    # Compute OOBBs from the pointcloud
    PRINT_INFO("========= COMPUTING OOBBS FROM POINTCLOUD =========\n")

    PRINT_INFO("Using " + str(os.cpu_count()) + " CPU core(s) for OOBB approximation.")
    executor = concurrent.futures.ProcessPoolExecutor(os.cpu_count())
    oobbs = list(executor.map(pointcloud_to_oobbs, [leaf[0].cpu().numpy().reshape(-1, 3) for leaf in part_geos]))

    ## Add ground OOBB by hand
    max_h_distance = 0
    lowest_y = 10000
    for oobb in oobbs:
        for cp in oobb:
            lowest_y = min(lowest_y, cp[2])
            max_h_distance = max(max(abs(cp[0]), abs(cp[1])), max_h_distance)
    lowest_y = lowest_y + 0.05

    ground_extent_h = max_h_distance * 1.25
    ground_extent_v = 0.25
    ground_points = np.array([
        [-ground_extent_h, lowest_y, -ground_extent_h],
        [ ground_extent_h, lowest_y, -ground_extent_h],
        [ ground_extent_h, lowest_y,  ground_extent_h],
        [-ground_extent_h, lowest_y,  ground_extent_h],
        [-ground_extent_h, lowest_y - ground_extent_v, -ground_extent_h],
        [ ground_extent_h, lowest_y - ground_extent_v, -ground_extent_h],
        [ ground_extent_h, lowest_y - ground_extent_v,  ground_extent_h],
        [-ground_extent_h, lowest_y - ground_extent_v,  ground_extent_h]]
    )
    oobbs.append(pointcloud_to_oobbs(ground_points))

    # Compute meshes from OOBBs and iteratively cut out any intersections to find contact surfaces 
    PRINT_INFO("\n========= CUTTING OUT INTERSECTIONS, COLLECTING CONTACT SURFACES =========\n")
    cut_meshes, cut_volumes, neighbours, contact_surfaces = cut_out_intersections(oobbs)

    # Place force vectors and solve the optimization problem to calculate the measure of infeasibility 
    A_eq, w, A_fr = build_constraint_elements(cut_meshes, contact_surfaces)
    moi = optimize_f(A_eq, w, A_fr)

    return moi, oobbs, cut_meshes, cut_volumes, contact_surfaces
