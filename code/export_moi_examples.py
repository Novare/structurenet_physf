import os
import matplotlib
from data import PartNetDataset
from vis_utils import draw_partnet_objects

from compute_moi import *
from compute_moi_util import *
import pandas as pd
import pdb

options = {
    "output_level": 0,
    "max_iterations": 1000,
    "dump_models": False,
    "surface_area_tolerance": 0.01,
    "print_surface_area_histogramm": True
}

matplotlib.pyplot.ion()

# results directory
root_dir = '../data/results/box_vae_chair'

# read all data
obj_list = sorted([item for item in os.listdir(root_dir) if item.endswith('.json')])

# export images
for obj_id in range(100):
    # pdb.set_trace()
    print("Exporting figure {}...".format(str(obj_id)))
    obj = PartNetDataset.load_object(os.path.join(root_dir, obj_list[obj_id]))

    figname = "chairs_with_moi/objnr_{}.png".format(str(obj_id))
    
    draw_partnet_objects(objects=[obj], object_names=[obj_list[obj_id]], 
                         figsize=(9, 5), leafs_only=True, visu_edges=True, 
                         sem_colors_filename='../stats/semantics_colors/Chair.txt',
                         save_fig=True, save_fig_file=figname)

# export moi details
pd_data = {
    "obj_id": [],
    "moi": [],
    "hov": [],
    "overall": []
}

for obj_id in range(100):
    obj = PartNetDataset.load_object(os.path.join(root_dir, obj_list[obj_id]))
    res = moi_from_graph(obj, options)    

    moi = res.moi - res.hover_penalty
    hov = res.hover_penalty

    pd_data["obj_id"].append(obj_id)
    pd_data["moi"].append(res.moi - res.hover_penalty)
    pd_data["hov"].append(res.hover_penalty)
    pd_data["overall"].append(res.moi)

    df = pd.DataFrame(pd_data, columns = ["obj_id", "moi", "hov", "overall"])
    df.to_csv(r"chairs_with_moi/stats.csv", index=False, header=True)

