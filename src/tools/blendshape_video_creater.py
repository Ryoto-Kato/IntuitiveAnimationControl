import matplotlib.pyplot as plt
import numpy as np
import sys, os
import matplotlib.animation as animation
from argparse import ArgumentParser, Namespace
from PIL import Image
import glob

parser = ArgumentParser(description="PCA and MBSPCA")
parser.add_argument('--path2folder', type=str, default=None)
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--save_name', type=str, default="all")
args = parser.parse_args(sys.argv[1:])

path2folder = args.path2folder
fig, ax = plt.subplots()
fig.tight_layout()
plt.margins(0,0)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.axis("off")

image_paths = []
for image in glob.glob(f"{path2folder}/*.png"):
    # print(image)
    image_paths.append(image)

image_paths = sorted(image_paths)

counts_frames = len(image_paths)

for i in range(len(image_paths)):
    image_paths.append(image_paths[counts_frames-i-1])

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
print(len(image_paths))

ax_ims = []
for i, image_path in enumerate(image_paths):
    _image = Image.open(image_path)
    img = np.asarray(_image)
    ax_im = ax.imshow(img, animated=True, )
    if i == 0:
        ax_im = ax.imshow(img)
    ax_ims.append([ax_im])

ani = animation.ArtistAnimation(fig, ax_ims, interval=50, blit=True,
                                repeat_delay=1000)

# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# To save the animation, use e.g.
#
path_to_save = path2folder
ani.save(os.path.join(path_to_save, f"{args.save_name}.mp4"))
#
# o
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

# plt.axis('off')
# plt.show()

