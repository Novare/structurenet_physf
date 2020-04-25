import os
import matplotlib
from data import PartNetDataset
from vis_utils import draw_partnet_objects
import datetime
import concurrent.futures

from compute_moi import *
from compute_moi_util import *
import pandas as pd
import pdb
matplotlib.pyplot.ion()

# results directory
root_dir = '../data/results/box_vae_chair'

# read all data
obj_list = sorted([item for item in os.listdir(root_dir) if item.endswith('.json')])

def export_image(obj_id):
    tmstmp = datetime.now().strftime("%d %b %Y - %H:%M:%S")
    print("{} - Exporting figure {}...".format(str(tmstmp), str(obj_id)))
    obj = PartNetDataset.load_object(os.path.join(root_dir, obj_list[obj_id]))

    figname = "chairs_with_moi/objnr_{}.png".format(str(obj_id))
    
    draw_partnet_objects(objects=[obj], object_names=[obj_list[obj_id]], 
                         figsize=(9, 5), leafs_only=True, visu_edges=False, 
                         sem_colors_filename='../stats/semantics_colors/Chair.txt',
                         save_fig=True, save_fig_file=figname)

for i in range(100):
    export_image(i)
