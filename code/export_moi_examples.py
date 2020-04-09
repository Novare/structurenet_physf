import os
import matplotlib
from data import PartNetDataset
from vis_utils import draw_partnet_objects

from compute_moi import *
from compute_moi_util import *

options = {
    "output_level": 3,
    "max_iterations": 1000,
    "dump_models": False,
    "surface_area_tolerance": 0.003,
    "print_surface_area_histogramm": True
}

matplotlib.pyplot.ion()

# results directory
root_dir = '../data/results/box_vae_chair'

# read all data
obj_list = sorted([item for item in os.listdir(root_dir) if item.endswith('.json')])

for obj_id in range(1):
    obj = PartNetDataset.load_object(os.path.join(root_dir, obj_list[obj_id]))
    res = moi_from_graph(obj, options)    
    figname = "chairs_with_moi/objnr_{}_moi_{}.png".format(str(obj_id), str(res.moi))
    
    draw_partnet_objects(objects=[obj], object_names=[obj_list[obj_id]], 
                         figsize=(9, 5), leafs_only=True, visu_edges=True, 
                         sem_colors_filename='../stats/semantics_colors/Chair.txt',
                         save_fig=True, save_fig_file=figname)
