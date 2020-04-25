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

options = {
    "output_level": 0,
    "max_iterations": 1000,
    "dump_models": False,
    "surface_area_tolerance": 0.003,
    "print_surface_area_histogramm": False
}

# results directory
root_dir = '../data/results/box_vae_chair'

# read all data
obj_list = sorted([item for item in os.listdir(root_dir) if item.endswith('.json')])

def generate_data(obj_id):
    print("Calculating MoI for object {}".format(str(obj_id)))
    obj = PartNetDataset.load_object(os.path.join(root_dir, obj_list[obj_id]))
    res = moi_from_graph(obj, options)    
    return (res.moi, res.hover_penalty)

results = []
res_range = 100
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(generate_data, range(res_range)))

# export moi details
pd_data = {
    "obj_id": [],
    "moi": [],
    "hov": [],
    "overall": []
}

for obj_id, res in enumerate(results):
    moi = res[0] 
    hov = res[1]

    pd_data["obj_id"].append(obj_id)
    pd_data["moi"].append(moi - hov)
    pd_data["hov"].append(hov)
    pd_data["overall"].append(moi)

df = pd.DataFrame(pd_data, columns = ["obj_id", "moi", "hov", "overall"])
df.to_csv(r"chairs_with_moi/stats_thesis.csv", index=False, header=True)

