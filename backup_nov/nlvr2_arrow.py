import os
os.system("/home/taoli1/ViLT/")
from ..ViLT.vilt.utils.write_nlvr2 import make_arrow
root = "/vc_data/users/taoli1/nlvr/"
arrows_root = "/vc_data/users/taoli1/data/"
make_arrow(root, arrows_root)