import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, TypedDict, Tuple


def windowing(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img


def get_arrays_by_plot(arr1: np.ndarray, arr2: np.ndarray, title1: str = None, title2: str = None, main_title: str = None) -> np.ndarray:
    """
    Display two arrays as images side by side in a plot.

    Parameters:
    arr1 (ndarray): The first array to be displayed as an image.
    arr2 (ndarray): The second array to be displayed as an image.
    title1 (str, optional): The title for the first image. Defaults to None.
    title2 (str, optional): The title for the second image. Defaults to None.

    Returns:
    ndarray: The image of the plot as a NumPy array.
    """
    fig, ax = plt.subplots(1, 2)
    
    if main_title:
        fig.suptitle(main_title)

    
    ax[0].imshow(arr1, cmap='gray')
    ax[1].imshow(arr2, cmap='gray')
      
    if title1:
        ax[0].set_title(title1)
    if title2:
        ax[1].set_title(title2)
        
    fig.canvas.draw_idle()
        
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, np.uint8).reshape((buf.shape[0], buf.shape[1], 4))
    
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    return img

def write_video(imgs: np.ndarray, out_path: str):
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (imgs.shape[2], imgs.shape[1]))
    for i in range(imgs.shape[0]):
        out.write(imgs[i])
        
    out.release()
    
    
class VisualBlock(TypedDict):
    title: str
    img: np.ndarray
    color_map: List[str]
    
class SubplotBlock(TypedDict):
    title: str
    visual_blocks: list[VisualBlock]
    layout: Tuple[int, int]
    
def get_array_by_subplot(subplot_block: SubplotBlock) -> np.ndarray:
    fig, ax = plt.subplots(subplot_block["layout"][0], subplot_block["layout"][1])
    
    fig.suptitle(subplot_block["title"])
    
    for i, visual_block in enumerate(subplot_block["visual_blocks"]):
        ax[i].imshow(visual_block["img"], cmap='gray')
        ax[i].set_title(visual_block["title"])
        
    fig.canvas.draw_idle()
        
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, np.uint8).reshape((buf.shape[0], buf.shape[1], 4))
    
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    return img