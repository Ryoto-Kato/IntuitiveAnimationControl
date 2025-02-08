import glob
import os
import sys
from argparse import ArgumentParser, Namespace
from PIL import Image


def make_gif(frame_folder):
    frames = []
    images = []
    image_paths = []
    for image in glob.glob(f"{frame_folder}/*.png"):
        # print(image)
        image_paths.append(image)
    image_paths = sorted(image_paths)
    # print(image_paths)
    for image_path in image_paths:
        _image = Image.open(image_path)
        for i in range(1):
            frames.append(_image)
        images.append(_image)
    counter=len(images)
    # print(counter)
    for i in range(counter):
        for j in range(1):
            frames.append(images[counter-i-1])
    # print("total frames:", len(frames))
    frame_one = frames[0]
    frame_one.save(os.path.join(frame_folder, "exp.gif"), format="GIF", append_images=frames,
               save_all=True, duration=len(frames), loop=0)
    
# def make_video(frame_folder):
#     frameSize = (1334, 2048)
#     out = cv2.VideoWriter(os.path.join('output_video.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)
#     for filename in glob.glob(f'{frame_folder}/*.png'):
#         img = cv2.imread(filename)
#         out.write(img)

#     out.release()



if __name__ == "__main__":
    parser = ArgumentParser(description="PCA and MBSPCA")
    parser.add_argument('--path2folder', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    make_gif(args.path2folder)
    # make_video(args.path2folder)