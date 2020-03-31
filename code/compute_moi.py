# compute_moi.py

from matplotlib import pyplot as plt
from compute_moi_util import *

import pyApproxMVBB as ap
import pymesh
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.optimize
import random
import os
import sys
import pdb
import time
import colorsys
import concurrent.futures
import logging as log


"""
    Convert a pointcloud to OOBBs using pyApproxMVBB.

    pointcloud -- The pointcloud to be converted.
"""


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


"""
    Convert the OOBBs to mesh structures using PyMesh, then try intersecting each unique
    pair of meshes. If the intersection is not empty, save the intersection and the adjacency
    relationship, then cut out the intersection from one of the meshes. In the end, the parts
    should not intersect each other anymore. While cutting out the intersections, save all
    faces from the cut mesh that belong to the other mesh as contact faces.

    oobbs -- A list of object-oriented bounding boxes that make up the mesh. 
    options -- Options dictionary:
        output_level -- Sets which log output to print (0: all, 1: info, 2: warnings, 3: errors) [default: 2].
        max_iterations -- Maximum iterations performed by the optimization routine [default: 1000].
        dump_models -- Sets whether to dump the cut meshes and their interfaces as OBJ files [default: False].
        surface_area_tolerance -- Sets the maximum surface area for triangles to be sorted out [default: 1e-3].
        print_surface_area_histogram -- Sets whether a histogram showing the contact surface area distribution should be printed.
                                        Useful for tweaking surface_area_tolerance. The histogram will only be shown when the
                                        output_level is 0 (debug)! [default: False]
"""


def cut_out_intersections(oobbs, options = {"output_level": 2, "max_iterations": 1000, "dump_models": False, "surface_area_tolerance": 1e-3, "print_surface_area_histogram": False}):
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
            [0, 2, 3],
            [0, 3, 1],
            [5, 7, 6],
            [5, 6, 4],
            [4, 6, 2],
            [4, 2, 0],
            [1, 3, 7],
            [1, 7, 5],
            [2, 6, 7],
            [2, 7, 3],
            [4, 0, 1],
            [4, 1, 5]
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
            # Only add contact triangles
            if not sources[k] == 0:
                continue
            triangle = meshes[i].vertices[face]

            # Don't add triangles with a zero edge
            if is_zero_edge_triangle(triangle):
                continue

            # Three collinear points don't form a triangle
            if is_collinear(triangle[0], triangle[1], triangle[2]):
                continue

            triangles.append(triangle)

        tol = options["surface_area_tolerance"]
        size_selected_triangles = [t for t in triangles if get_triangle_surface_area(t) > tol]
        if len(size_selected_triangles) > 0:
            triangles = size_selected_triangles
        else:
            # The size tolerance is so low that no triangle of the interface survived.
            # Since we should at least have one triangle per interface, we try to find
            # a size limit that allows the biggest few triangles to get in.
            new_tol = tol
            augmented_size_selected_triangles = []
            while len(augmented_size_selected_triangles) == 0:
               new_tol *= 0.9
               augmented_size_selected_triangles = [t for t in triangles if get_triangle_surface_area(t) > new_tol]
            triangles = augmented_size_selected_triangles

        if len(triangles) > 0:
            contact_triangles.append((i, j, triangles))
        else:
            lost_interfaces += 1

    if options["dump_models"]:
        for cs in contact_triangles:
            i = cs[0]
            j = cs[1]
            t = cs[2]

            vertices = []
            faces = []

            for triangle in t:
                for vertex in triangle:
                    vertices.append(vertex)

            for index in range(len(vertices)):
                if index % 3 == 0:
                    faces.append(np.array([index, index + 1, index + 2]))

            vertices = np.array(vertices)
            faces = np.array(faces)

            m = pymesh.form_mesh(vertices, faces)
            pymesh.save_mesh("debug_output/interface_{}_{}.obj".format(str(i), str(j)), m)

        # Dump the whole cut mesh
        for i, m in enumerate(meshes):
            pymesh.save_mesh("debug_output/cut_mesh_{}.obj".format(str(i)), m)

    return meshes, cut_volumes, neighbours, contact_triangles


"""
    Build the A_eq submatrix used in one of the optimization constraints.
    
    mesh -- One of the two meshes the interface belongs to. 
    interface -- An interface between two meshes.
"""


def build_Aeq_submatrix(mesh, interface):
    c = get_mesh_centroid(mesh)
    vertex_count = 3 * len(interface)
    Ajk = np.zeros((6, 4 * vertex_count))

    # log.debug("Adding interface to A_eq. Interface has {} triangles.".format(str(len(interface))))

    start_index = 0
    for t, triangle in enumerate(interface):
        n = get_triangle_normal(triangle)
        t1 = normalize(triangle[1] - triangle[0])
        t2 = np.cross(n, t1)

        for vt, vertex in enumerate(triangle):
            v = normalize(c - vertex)
            nv = np.cross(n, v)
            t1v = np.cross(t1, v)
            t2v = np.cross(t2, v)

            Ajk[0:3, start_index + 0] = n
            Ajk[0:3, start_index + 1] = -n
            Ajk[0:3, start_index + 2] = t1
            Ajk[0:3, start_index + 3] = t2
            Ajk[3:6, start_index + 0] = nv
            Ajk[3:6, start_index + 1] = -nv
            Ajk[3:6, start_index + 2] = t1v
            Ajk[3:6, start_index + 3] = t2v
            start_index += 4
    return Ajk


"""
    Build the relevant vectors and matrices to set up the constraint system.

    cut_meshes -- The meshes that make up the structure.
    contact_surfaces -- The contact surfaces between meshes.

    TODO: What about the ground mesh? Remove it? Leave it?
"""


def build_constraint_elements(cut_meshes, contact_surfaces):
    # Build A_eq from contact surfaces
    part_count = len(cut_meshes)
    A_eq_rowcount = 6 * (part_count - 1)
    A_eq_columncount = 0
    submatrices = []
    for k, cs in enumerate(contact_surfaces):
        si = cs[0]
        sj = cs[1]
        surface_count = len(cs[2])

        # Every contact surface results in two submatrices, one per part
        smi = build_Aeq_submatrix(cut_meshes[si], cs[2])
        smj = build_Aeq_submatrix(cut_meshes[sj], cs[2])

        submatrices.append((si, k, smi))
        submatrices.append((sj, k, smj))
        A_eq_columncount += np.shape(smi)[1]

    column_offset = 0
    A_eq_data = []
    A_eq_row_ind = []
    A_eq_col_ind = []
    for k in range(len(contact_surfaces)):
        column_offset_new = 0
        for j in range(len(cut_meshes)):
            # Ignore ground mesh row
            if j == part_count - 1:
                continue

            fitting_submatrices = [sm for sm in submatrices if sm[0] == j and sm[1] == k]
            if len(fitting_submatrices) == 0:
                continue
            else:
                sm = fitting_submatrices[0]
                column_offset_new = column_offset + np.shape(sm[2])[1]

                for r in range(np.shape(sm[2])[0]):
                    for c in range(np.shape(sm[2])[1]):
                        A_eq_row_ind.append(j * 6 + r)
                        A_eq_col_ind.append(column_offset + c)
                        A_eq_data.append((sm[2])[r][c])
        column_offset = column_offset_new

    A_eq_data = np.array(A_eq_data)
    A_eq_row_ind = np.array(A_eq_row_ind)
    A_eq_col_ind = np.array(A_eq_col_ind)
    A_eq = (sp.sparse.coo_matrix((A_eq_data, (A_eq_row_ind, A_eq_col_ind)),
                                 shape=(A_eq_rowcount, A_eq_columncount),
                                 dtype=np.float32)).tocsc()

    # Build w from cut_meshes
    w = np.zeros(6 * (len(cut_meshes) - 1))
    for i, mesh in enumerate(cut_meshes):
        # Exclude ground mesh
        if i == len(cut_meshes) - 1:
            break

        weight = get_mesh_volume(mesh)
        w[i * 6:i * 6 + 6] = np.array([0, 0, -weight, 0, 0, 0])

    # Build A_fr from contact surfaces
    alpha = 0.7
    A_fr_column_count = np.shape(A_eq)[1]
    A_fr_row_count = (int)(A_fr_column_count / 2)

    A_fr_row_ind = []
    A_fr_col_ind = []
    A_fr_data = []
    for i in range((int)(A_fr_column_count / 4)):
        r = i * 2
        c = i * 4

        # [-alpha 1 0]
        # [-alpha 0 1]

        A_fr_row_ind.append(r)
        A_fr_col_ind.append(c)
        A_fr_data.append(-alpha)

        A_fr_row_ind.append(r + 1)
        A_fr_col_ind.append(c)
        A_fr_data.append(-alpha)

        A_fr_row_ind.append(r)
        A_fr_col_ind.append(c + 1)
        A_fr_data.append(1)

        A_fr_row_ind.append(r + 1)
        A_fr_col_ind.append(c + 2)
        A_fr_data.append(1)

    A_fr_data = np.array(A_fr_data)
    A_fr_row_ind = np.array(A_fr_row_ind)
    A_fr_col_ind = np.array(A_fr_col_ind)
    A_fr = (sp.sparse.coo_matrix((A_fr_data, (A_fr_row_ind, A_fr_col_ind)),
                                 shape=(A_fr_row_count, A_fr_column_count),
                                 dtype=np.float32)).tocsc()

    # Build A_compr
    A_compr_column_count = np.shape(A_eq)[1]
    A_compr_row_count = (int)(A_compr_column_count / 2)

    A_compr_data = np.repeat(1, A_compr_row_count)
    A_compr_row_ind = np.array(range(A_compr_row_count))
    A_compr_col_ind = []
    for i in range((int)(A_compr_column_count / 4)):
        A_compr_col_ind.append(4 * i)
        A_compr_col_ind.append(4 * i + 1)
    A_compr_col_ind = np.array(A_compr_col_ind)
    A_compr = (sp.sparse.coo_matrix((A_compr_data, (A_compr_row_ind, A_compr_col_ind)),
                                    shape=(A_compr_row_count, A_compr_column_count),
                                    dtype=np.float32)).tocsc()

    return A_eq, w, A_fr, A_compr

"""
    Since some meshes might not be connected to the ground either directly or indirectly,
    we calculate a hover penalty for every mesh "hovering" above the ground. The penalty
    is the minimum euclidian distance to a mesh that has direct or indirect ground contact.

    meshes -- All meshes in the scene, both grounded and not-grounded.
    neighbours - All neighbour-relationships between individual meshes.
"""

def calculate_hover_penalty(meshes, neighbours):
    ground_index = len(meshes) - 1 # We assume the ground is the last mesh

    grounded_meshes = [ground_index]
    last_len = 0 
    while not len(grounded_meshes) == last_len:
        last_len = len(grounded_meshes)

        for (i, j) in neighbours:
            if (i in grounded_meshes) and (j not in grounded_meshes):
                grounded_meshes.append(j)
            if (j in grounded_meshes) and (i not in grounded_meshes):
                grounded_meshes.append(i)

    hover_penalty = 0.0
    hover_meshes = []
    for i in [not_ground for not_ground in range(0, ground_index) if not not_ground in grounded_meshes]:
        # Find smallest euclidian distance between this mesh and a grounded one
        smallest_distance = np.inf
        for j in grounded_meshes:
            d = norm(get_mesh_centroid(meshes[i]) - get_mesh_centroid(meshes[j]))
            smallest_distance = np.minimum(smallest_distance, d) 
        hover_penalty += smallest_distance
        hover_meshes.append(meshes[i])
    
    return hover_penalty, hover_meshes

"""
    The function calculating the measure of feasibility.
    This assumes that f is made up of vectors containing 4 scalars:
    f_n+, f_n-, f_t1, f_t2 respectively

    f -- The vector containing all forces across the structure.
"""


def f_func(f):
    return np.sum(np.square(f[1::4]))


"""
    Starts the optimization routine.

    A_eq -- Constraint matrix for static equilibrium.
    w -- Weight vector for static equilibrium.
    A_fr -- Friction matrix for friction constraint.
    A_compr -- Compression matrix for compression constraints.
    hover_penalty -- Addition to the MoI based on meshes that do not intersect any mesh at all or
                     any mesh that intersects the ground directly or indirectly. This is to penalize
                     "hovering" meshes.
    options -- Options dictionary:
        output_level -- Sets which log output to print (0: all, 1: info, 2: warnings, 3: errors) [default: 2].
        max_iterations -- Maximum iterations performed by the optimization routine [default: 1000].
        dump_models -- Sets whether to dump the cut meshes and their interfaces as OBJ files [default: False].
        surface_area_tolerance -- Sets the maximum surface area for triangles to be sorted out [default: 1e-3].
        print_surface_area_histogram -- Sets whether a histogram showing the contact surface area distribution should be printed.
                                        Useful for tweaking surface_area_tolerance. The histogram will only be shown when the
                                        output_level is 0 (debug)! [default: False]
"""


def optimize_f(A_eq, w, A_fr, A_compr, hover_penalty, options = {"output_level": 2, "max_iterations": 1000, "dump_models": False, "surface_area_tolerance": 1e-3, "print_surface_area_histogram": False}):
    start_time = time.time()

    # The matrices should not contain any NaN entries!
    assert(not np.isnan(np.sum(A_eq)))
    assert(not np.isnan(np.sum(A_fr)))
    assert(not np.isnan(np.sum(A_compr)))

    f_dim = np.shape(A_eq)[1]
    f0 = np.random.rand(f_dim) * 2.0 - np.repeat(1, f_dim)

    constr_static = sp.optimize.LinearConstraint(A_eq, -w, -w)

    fric_ll = np.repeat(-np.inf, np.shape(A_fr)[0])
    fric_ul = np.repeat(0, np.shape(A_fr)[0])
    constr_fric = sp.optimize.LinearConstraint(A_fr, fric_ll, fric_ul) 

    compr_ll = np.repeat(0, np.shape(A_compr)[0])
    compr_ul = np.repeat(np.inf, np.shape(A_compr)[0])
    constr_compr = sp.optimize.LinearConstraint(A_compr, compr_ll, compr_ul)

    constraints = [constr_static, constr_compr, constr_fric]

    vbs = {
        0: 3,
        1: 1,
        2: 1,
        3: 0,
    } 

    np.savetxt("Aeq.csv", A_eq.todense(), delimiter=",")
    np.savetxt("w.csv", w, delimiter=",")
    np.savetxt("Afr.csv", A_fr.todense(), delimiter=",")
    np.savetxt("Acompr.csv", A_compr.todense(), delimiter=",")
    
    res = sp.optimize.minimize(fun=f_func,
                               x0=f0,
                               method="trust-constr",
                               constraints=constraints,
                               options={"verbose": vbs[options["output_level"]], "maxiter": options["max_iterations"]})
    moi = f_func(res.x) + hover_penalty

    time_diff = (time.time() - start_time) / 60.0
    log.info("Optimization took {} minutes.".format(str(time_diff)))
    return moi

"""
    Apply options where applicable.

    options -- Options dictionary:
        output_level -- Sets which log output to print (0: all, 1: info, 2: warnings, 3: errors) [default: 2].
        max_iterations -- Maximum iterations performed by the optimization routine [default: 1000].
        dump_models -- Sets whether to dump the cut meshes and their interfaces as OBJ files [default: False].
        surface_area_tolerance -- Sets the maximum surface area for triangles to be sorted out [default: 1e-3].
        print_surface_area_histogram -- Sets whether a histogram showing the contact surface area distribution should be printed.
                                        Useful for tweaking surface_area_tolerance. The histogram will only be shown when the
                                        output_level is 0 (debug)! [default: False]
"""
def apply_options(options = {"output_level": 2, "max_iterations": 1000, "dump_models": False, "surface_area_tolerance": 1e-3, "print_surface_area_histogram": False}):
    logformat = "%(asctime)s - %(levelname)-8s - %(message)s"
    dtfmt = "%H:%M:%S"

    if options["output_level"] == 0:
        log.basicConfig(level=log.NOTSET, stream=sys.stdout, format=logformat, datefmt=dtfmt)
    elif options["output_level"] == 1:
        log.basicConfig(level=log.INFO, stream=sys.stdout, format=logformat, datefmt=dtfmt)
    elif options["output_level"] == 2:
        log.basicConfig(level=log.WARNING, stream=sys.stdout, format=logformat, datefmt=dtfmt)
    elif options["output_level"] == 3:
        log.basicConfig(level=log.ERROR, stream=sys.stdout, format=logformat, datefmt=dtfmt)

class MOIResult:
    def __init__(self, moi, hover_penalty, oobbs, cut_meshes, hover_meshes, cut_volumes, contact_surfaces):
        self.moi = moi
        self.hover_penalty = hover_penalty
        self.oobbs = oobbs
        self.cut_meshes = cut_meshes
        self.hover_meshes = hover_meshes
        self.cut_volumes = cut_volumes
        self.contact_surfaces = contact_surfaces

"""
    Calculates the measure of infeasibility from a StructureNet graph.

    obj -- A StructureNet graph object
    options -- Options dictionary:
        output_level -- Sets which log output to print (0: all, 1: info, 2: warnings, 3: errors) [default: 2].
        max_iterations -- Maximum iterations performed by the optimization routine [default: 1000].
        dump_models -- Sets whether to dump the cut meshes and their interfaces as OBJ files [default: False].
        surface_area_tolerance -- Sets the maximum surface area for triangles to be sorted out [default: 1e-3].
        print_surface_area_histogram -- Sets whether a histogram showing the contact surface area distribution should be printed.
                                        Useful for tweaking surface_area_tolerance. The histogram will only be shown when the
                                        output_level is 0 (debug)! [default: False]
"""


def moi_from_graph(obj, options = {"output_level": 2, "max_iterations": 1000, "dump_models": False, "surface_area_tolerance": 1e-3, "print_surface_area_histogram": False}):
    apply_options(options)

    part_boxes, part_geos, edges, part_ids, part_sems = obj.graph(leafs_only=True)

    if not part_boxes is None and len(part_boxes) > 0:
        coord_rot = np.matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        oobbs = []
        for box in part_boxes:
            center = np.array(box[0:3])
            lengths = np.array(box[3:6])
            dir_1 = np.array(box[6:9])
            dir_2 = np.array(box[9:])

            if not np.any(dir_1) or not np.any(dir_2) or not np.any(lengths):
#                log.warning("Graph contains bounding box with a zero vector for directions/lengths. Skipping.")
                continue

            dir_1 = normalize(np.array(box[6:9]))
            dir_2 = normalize(np.array(box[9:]))
            dir_3 = np.cross(dir_1, dir_2)

            d1 = 0.5 * lengths[0] * dir_1
            d2 = 0.5 * lengths[1] * dir_2
            d3 = 0.5 * lengths[2] * dir_3

            oobb = []

            # [0] = (0, 0, 0)
            oobb.append(center - d1 - d2 - d3)

            # [1] = (1, 0, 0)
            oobb.append(center + d1 - d2 - d3)

            # [2] = (0, 1, 0)
            oobb.append(center - d1 + d2 - d3)

            # [3] = (1, 1, 0)
            oobb.append(center + d1 + d2 - d3)

            # [4] = (0, 0, 1)
            oobb.append(center - d1 - d2 + d3)

            # [5] = (1, 0, 1)
            oobb.append(center + d1 - d2 + d3)

            # [6] = (0, 1, 1)
            oobb.append(center - d1 + d2 + d3)

            # [7] = (1, 1, 1)
            oobb.append(center + d1 + d2 + d3)

            for i, point in enumerate(oobb):
                oobb[i] = np.asarray(coord_rot * np.array(point).reshape(-1, 1)).reshape(-1)

            oobbs.append(np.array(oobb))
        return moi_from_bounding_boxes(oobbs, options)
    elif not part_geos is None and len(part_geos) > 0:
        cloud = [leaf[0].cpu().numpy().reshape(-1, 3) for leaf in part_geos]
        return moi_from_pointcloud(cloud, options)
    else:
#        log.warning("Passed empty graph into moi_from_graph. Returning 0.0.")
        return MOIResult(moi=0.0,
                         hover_penalty=0.0,
                         oobbs=[],
                         cut_meshes=[],
                         hover_meshes=[],
                         cut_volumes=[],
                         contact_surfaces=[])


"""
    Calculates the measure of infeasibility from a point cloud representation.

    pointcloud -- list of pointclouds making up individual parts. Needs to be a numpy array of 3D points
                  in the form of an Nx3 matrix, N being the number of points.
    options -- options dictionary:
        output_level -- Sets which log output to print (0: all, 1: info, 2: warnings, 3: errors) [default: 2].
        max_iterations -- Maximum iterations performed by the optimization routine [default: 1000].
        dump_models -- Sets whether to dump the cut meshes and their interfaces as OBJ files [default: False].
        surface_area_tolerance -- Sets the maximum surface area for triangles to be sorted out [default: 1e-3].
        print_surface_area_histogram -- Sets whether a histogram showing the contact surface area distribution should be printed.
                                        Useful for tweaking surface_area_tolerance. The histogram will only be shown when the
                                        output_level is 0 (debug)! [default: False]
"""


def moi_from_pointcloud(pointcloud, options = {"output_level": 2, "max_iterations": 1000, "dump_models": False, "surface_area_tolerance": 1e-3, "print_surface_area_histogram": False}):
    apply_options(options)

    if len(pointcloud) == 0:
#        log.warning("Passed empty pointcloud list into moi_from_pointcloud. Returning 0.0.")
        return MOIResult(moi=0.0,
                         hover_penalty=0.0,
                         oobbs=[],
                         cut_meshes=[],
                         hover_meshes=[],
                         cut_volumes=[],
                         contact_surfaces=[])

    log.info("COMPUTING OOBBS FROM POINTCLOUD")
    log.info("Using {} CPU core(s) for OOBB approximation.".format(str(os.cpu_count())))

    executor = concurrent.futures.ProcessPoolExecutor(os.cpu_count())
    oobbs = list(executor.map(pointcloud_to_oobbs, pointcloud))
    return moi_from_bounding_boxes(oobbs, options)


"""
    Calculates the measure of infeasibility from a bounding box representation.

    oobbs -- list of bounding boxes making up individual parts. Needs to be a numpy array of 3D box vertices
             adhering the following order:
             [0] = (0, 0, 0)
             [1] = (1, 0, 0)
             [2] = (0, 1, 0)
             [3] = (1, 1, 0)
             [4] = (0, 0, 1)
             [5] = (1, 0, 1)
             [6] = (0, 1, 1)
             [7] = (1, 1, 1)
    options -- options dictionary:
        output_level -- Sets which log output to print (0: all, 1: info, 2: warnings, 3: errors) [default: 2].
        max_iterations -- Maximum iterations performed by the optimization routine [default: 1000].
        dump_models -- Sets whether to dump the cut meshes and their interfaces as OBJ files [default: False].
        surface_area_tolerance -- Sets the maximum surface area for triangles to be sorted out [default: 1e-3].
        print_surface_area_histogram -- Sets whether a histogram showing the contact surface area distribution should be printed.
                                        Useful for tweaking surface_area_tolerance. The histogram will only be shown when the
                                        output_level is 0 (debug)! [default: False]
"""


def moi_from_bounding_boxes(oobbs, options = {"output_level": 2, "max_iterations": 1000, "dump_models": False, "surface_area_tolerance": 1e-3, "print_surface_area_histogram": False}):
    apply_options(options)
    
    if len(oobbs) == 0:
#        log.warning("Passed empty bounding box list into moi_from_bounding_boxes. Returning 0.0.")
        return MOIResult(moi=0.0,
                         hover_penalty=0.0,
                         oobbs=[],
                         cut_meshes=[],
                         hover_meshes=[],
                         cut_volumes=[],
                         contact_surfaces=[])

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
    ground_points = np.array([[-ground_extent_h, lowest_y, -ground_extent_h],
                              [ground_extent_h, lowest_y,
                               -ground_extent_h], [ground_extent_h, lowest_y, ground_extent_h],
                              [-ground_extent_h, lowest_y, ground_extent_h],
                              [-ground_extent_h, lowest_y - ground_extent_v, -ground_extent_h],
                              [ground_extent_h, lowest_y - ground_extent_v, -ground_extent_h],
                              [ground_extent_h, lowest_y - ground_extent_v, ground_extent_h],
                              [-ground_extent_h, lowest_y - ground_extent_v, ground_extent_h]])
    oobbs.append(pointcloud_to_oobbs(ground_points))

    # Compute meshes from OOBBs and iteratively cut out any intersections to find contact surfaces
    log.info("CUTTING OUT INTERSECTIONS, COLLECTING CONTACT SURFACES")
    cut_meshes, cut_volumes, neighbours, contact_surfaces = cut_out_intersections(oobbs, options)

    if options["output_level"] == 0:
        cscount = 0
        sareas = []
        for cs in contact_surfaces:
            cscount += len(cs[2])
            for triangle in cs[2]:
                sareas.append(get_triangle_surface_area(triangle))
        log.debug("Optimization has to consider {} contact surfaces!".format(str(cscount)))
        log.debug("Printing distribution of contact surface areas:")
        plt.hist(x=np.array(sareas), bins=64)
        plt.show()

    # Place force vectors and solve the optimization problem to calculate the measure of infeasibility
    log.info("OPTIMIZING")
    A_eq, w, A_fr, A_compr = build_constraint_elements(cut_meshes, contact_surfaces)
    hover_penalty, hover_meshes = calculate_hover_penalty(cut_meshes, neighbours)
    moi = optimize_f(A_eq, w, A_fr, A_compr, hover_penalty, options)

    return MOIResult(moi=moi,
                     hover_penalty=hover_penalty,
                     oobbs=oobbs,
                     cut_meshes=cut_meshes,
                     hover_meshes=hover_meshes,
                     cut_volumes=cut_volumes,
                     contact_surfaces=contact_surfaces)
