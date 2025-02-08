import matplotlib.pyplot as plt
import numpy as np
import sys, os
import matplotlib.animation as animation
from argparse import ArgumentParser, Namespace
from PIL import Image
import glob
sys.path.append(os.pardir)
from utils.Dataset_handler import Filehandler

parser = ArgumentParser(description="blendshape_video_createrALL")
parser.add_argument('--path2folder', type=str, default=None)
args = parser.parse_args(sys.argv[1:])

path2folder = args.path2folder
fig, ax = plt.subplots()
fig.tight_layout()
plt.margins(0,0)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.axis("off")

list_dirnames, list_dirpaths = Filehandler.dirwalker_InFolder(path_to_folder=path2folder, prefix="")

image_paths = []
for list_dirname, list_dirpath in zip(list_dirnames, list_dirpaths):
    for image in glob.glob(f"{list_dirpath}/*.png"):
        image_paths.append(image)

print(len(image_paths))

ax_ims = []
for i, image_path in enumerate(image_paths):
    _image = Image.open(image_path)
    img = np.asarray(_image)
    ax_im = ax.imshow(img, animated=True)
    if i == 0:
        ax_im = ax.imshow(img)
    ax_ims.append([ax_im])

ani = animation.ArtistAnimation(fig, ax_ims, interval=1000, blit=True,
                                repeat_delay=1000, repeat = True)

# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# To save the animation, use e.g.
#
ani.save(os.path.join(path2folder,"aLL_exps.mp4"))
#
# o
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

# plt.axis('off')
# plt.show()

