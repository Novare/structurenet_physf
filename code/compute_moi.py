# compute_moi.py

from matplotlib import pyplot as plt

import pyApproxMVBB as ap
import pymesh
import numpy as np
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

    moi = 0.0
    return moi, oobbs, cut_meshes, cut_volumes, contact_surfaces
