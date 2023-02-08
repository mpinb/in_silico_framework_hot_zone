"""
This python file contains multiple useful general methods that can be used throughout the visualisation module.

Author: Maria Royo, Bjorge Meulemeester, Rieke Fruengel, Arco Bast
Date: 05/01/2023
"""

import os
from PIL import Image
import jinja2
import IPython
import glob
import numpy as np
from base64 import b64encode
import subprocess

def write_video_from_images(images_dir, out_path, fps=24, images_format = '.png', quality=5, codec='mpeg4'):
    '''
    Creates a video from a set of images. Images must be specified as a path to a directory that contains the images.
    Uses glob pattern pmatching by default (allows for using the "*" as a wildcard). Glob is not enabled by default on Windows machines.
    If running this command on windows, please set the :@param glob: argument to False and specify a non-glob type match pattern.

    Args:
        - images_dir: path where the images are
        - out_path: dir where the video will be generated + name of the video
        - fps: frames per second
        - match_pattern: the string pattern to match images in the specified directory. E.g. frame00*.png
        - quality
        - codec
    '''
    out = subprocess.call([
        "ffmpeg", "-y", "-r", str(fps), "-i", images_dir+"/%*"+images_format, "-vcodec", codec, "-q:v", str(quality), "-r", str(fps), out_path
        ])
    if out == 0:
        print('Video created successfully!')
    else:
        print('Something went wrong. Make sure:')
        print(' - the module ffmpeg is loaded, otherwise use the command: module load ffmpeg')
        print(' - out_dir contains images')
        print(' - images_format is specified properly')
    
def write_gif_from_images(images_dir, out_path, images_format = '.png', duration=40):
    '''
    Creates a gif from a set of images, and saves it to :@param out_path:.

    Args:
        - Name: path where the gif will be located + name of the gif
        - Files: list of the images that will be used for the gif generation
        - Duration: duration of each frame in ms
    '''
    frames = []
    for f in os.listdir(images_dir):
        if f.endswith(images_format):
            frames.append(Image.open(os.path.join(images_dir,f)))
    # Save into a GIF file that loops forever   
    frames[0].save(out_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    print('Gif created successfully!')

def _load_base64(filename, extension = 'png'):
    #https://github.com/jakevdp/JSAnimation/blob/master/JSAnimation/html_writer.py
        with open(filename, 'rb') as f:
            data = f.read()
        return 'data:image/{0};base64,{1}'.format(extension,b64encode(data).decode('ascii'))
    
def display_animation_from_images(files, interval=10, style=False, animID = None, embedded = False):
        '''
        @TODO: resolve module not found errors
        creates an IPython animation out of files specified in a globstring or a list of paths.

        animID: unique integer to identify the animation in the javascript environment of IPython
        files: globstring or list of paths
        interval: time interval between frames

        CAVEAT: the paths need to be relative to the location of the ipynb / html file, since
        the are resolved in the browser and not by python'''
        if animID is None:
            animID = np.random.randint(10000000000000) # needs to be unique within one ipynb
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))
        template = env.get_template('animation_template.html')

        if isinstance(files, str):
            if os.path.isdir(files): # folder provieded --> convert to globstring
                files = os.path.join(files, '*.png')
            listFrames = sorted(glob.glob(files))
        else:
            listFrames = files
        if embedded:
            listFrames = [_load_base64(f) for f in listFrames]
        htmlSrc = template.render(ID=animID, listFrames=listFrames, interval=interval, style=style)
        IPython.display.display(IPython.display.HTML(htmlSrc))
      