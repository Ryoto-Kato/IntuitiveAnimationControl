import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace

path_to_src = os.pardir
sys.path.append(path_to_src)
from utils.Dataset_handler import Filehandler

path_to_3DGS = os.path.join(os.getcwd() )

parser = ArgumentParser(description="Summary Creation")
parser.add_argument('--path2folder', type=str, default=None)
parser.add_argument('--save_fig', action='store_true', default=False)
args = parser.parse_args(sys.argv[1:])

assert args.path2folder != None
    
path_to_images = args.path2folder
list_images = []

ext = ".png"
list_fnames, list_fpaths = Filehandler.fileWalker_InDirectory(path_to_images, ext)
print(len(list_fnames))
numImg = len(list_fnames)
rows = int(np.ceil(numImg/5))
colmns = 5

fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(
    top=1.0,
    bottom=0,
    left=0,
    right=1.0,
    hspace=0,
    wspace=0
)

for i, paht2img in enumerate(list_fpaths):
    img = Image.open(paht2img)
    fig.add_subplot(rows, colmns, int(i+1))
    plt.imshow(img)
    plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
if args.save_fig:
    fig.savefig(os.path.join(path_to_images, "summary.png"), dpi=200)

