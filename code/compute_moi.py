# compute_moi.py

from matplotlib import pyplot as plt

import pyApproxMVBB as ap
import pymesh
import numpy as np
import random
import os
import colorsys
import concurrent.futures

from compute_moi_util import *

def pointcloud_to_oobbs(pointcloud):
    coord_rot = np.matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    pcl = np.transpose(np.float64(np.array(pointcloud)))       
    oobb = ap.approximateMVBB(pts=pcl,
                              epsilon=0.001,
                              pointSamples=500,
                              gridSize=5,
                              mvbbDiamOptLoops=0,
                              mvbbGridSearchOptLoops=5,
                              seed=1234)

    cps = oobb.getCornerPoints()
    for i, point in enumerate(cps):
        cps[i] = np.asarray(coord_rot * np.array(point).reshape(-1, 1)).reshape(-1)
    return cps

def oobbs_to_meshes_and_cutvolumes(oobbs):
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

    cutvolumes = []
    for (i, j) in unique_pairs:
        cvol = pymesh.boolean(meshes[i], meshes[j], operation="intersection", engine="igl")
        if len(cvol.vertices) == 0:
            continue
        cvol, riv_info = pymesh.remove_isolated_vertices(cvol)
        cvol, rdv_info = pymesh.remove_duplicated_vertices(cvol, 1e-3)
        cvol, rdf_info = pymesh.remove_duplicated_faces(cvol)
        
        if riv_info["num_vertex_removed"] != 0 or rdv_info["num_vertex_merged"] != 0:
            PRINT_DEBUG("Removed " + str(riv_info["num_vertex_removed"]) + " isolated vertices and merged "
                  + str(rdv_info["num_vertex_merged"]) + " duplicated vertices.")

        if len(cvol.vertices) == 0:
            continue
        cutvolumes.append((cvol, i, j))
    return meshes, cutvolumes

def calculate_contact_surfaces(meshtuple):
    ivol_vertices = meshtuple[0]
    ivol_faces = meshtuple[1]
    jvol_vertices = meshtuple[2]
    jvol_faces = meshtuple[3]
    
    THRESH = 0.01
    RAYCOUNT = 1

    contact_surfaces = []

    tries = 0
    failed_first = 0
    failed_second = 0
    failed_third = 0
    
    for iface in ivol_faces:
        for jface in jvol_faces:
            tries += 1
            
            # face needs to be parallel
            inormal = get_triangle_normal(ivol_vertices[iface[0]], ivol_vertices[iface[1]], ivol_vertices[iface[2]])
            jnormal = get_triangle_normal(jvol_vertices[jface[0]], jvol_vertices[jface[1]], jvol_vertices[jface[2]])
            
            if 1 - np.abs(np.dot(inormal, jnormal)) > 0.001:
                failed_first += 1
                continue
                
            # face distance needs to be smaller than threshold
            iv = ivol_vertices[iface[0]]
            jv = jvol_vertices[jface[0]]
            d = norm(np.dot(iv - jv, jnormal))

            if d > THRESH:
                failed_second += 1
                continue
            
            # intersection test using ray tracing
            v0 = ivol_vertices[iface[0]]
            v1 = ivol_vertices[iface[1]]
            v2 = ivol_vertices[iface[2]]

            anyInt = False
            for i in range(RAYCOUNT):
                p = v0 + v1 * np.random.rand() + v2 * np.random.rand()
                anyInt = anyInt or (intersect_plane(p, inormal, jv, jnormal) != np.inf)
                anyInt = anyInt or (intersect_plane(p, -inormal, jv, jnormal) != np.inf)
            
            if not anyInt:
                failed_third += 1
                continue
            
            rv = np.array([ivol_vertices[iface[0]], ivol_vertices[iface[1]], ivol_vertices[iface[2]]])
            contact_surfaces.append(rv)
    
    PRINT_DEBUG("Got " + str(len(contact_surfaces)) + " contact surfaces. Tries: " + str(tries)
         + ", first try fails: " + str(failed_first)
         + ", second try fails: " + str(failed_second)
         + ", third try fails: " + str(failed_third))
    return contact_surfaces

def cut_out_intersection_volumes(meshes, cutvolumes):
    cut_meshes = np.copy(meshes)
    for (cv, i, j) in cutvolumes:
        cut_meshes[i] = pymesh.boolean(cut_meshes[i], cv, operation="difference", engine="igl")
        cut_meshes[i], _ = pymesh.remove_isolated_vertices(cut_meshes[i])
        cut_meshes[i], _ = pymesh.remove_duplicated_vertices(cut_meshes[i], 1e-3)
        cut_meshes[i], _ = pymesh.remove_duplicated_faces(cut_meshes[i])

        if len(cut_meshes[i].vertices) == 0:
            del cut_meshes[i]
    return cut_meshes 
