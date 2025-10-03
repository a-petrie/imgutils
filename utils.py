from PIL import Image, ImageDraw
from copy import deepcopy
from cv2 import filter2D
from scipy.ndimage import rotate as scipy_rotate
import numpy as np

FRAME_RATE = 60

def show(image_array: np.ndarray) -> None:
    Image.fromarray(image_array).show()

    
def save(image_array: np.ndarray, fname: str) -> None:
    Image.fromarray(image_array).save(fname)

    
def save_gif(frames: [np.ndarray], fname: str, duration: int = 50) -> None:
    imgs = [Image.fromarray(f) for f in frames] 
    first, rest = imgs[0], imgs[1:]
    first.save(fname, save_all=True, append_images=rest, duration=duration, repeat=1)


def crop(img: np.ndarray, left: int=0, right: int=0, bottom: int=0, top: int=0) -> np.ndarray:
    return img[top:bottom, left:right]


def image_to_numpy(img: Image) -> np.ndarray:
    return np.array(img)

    
def load(fname: str) -> np.ndarray:
    return image_to_numpy(Image.open(fname))

    
def rint(array: np.ndarray) -> np.ndarray:
    return np.array(np.rint(array), dtype=np.uint8)


def x_flip(array: np.ndarray) -> np.ndarray:
    return np.flip(array, axis=0)
    

def y_flip(array: np.ndarray) -> np.ndarray:
    return np.flip(array, axis=1)


def c_flip(array: np.ndarray) -> np.ndarray:
    return np.flip(array, axis=2)


def sine(width: int, height: int, density: int = 10) -> np.ndarray:
    total_pixels = width * height
    time = np.linspace(0, density, total_pixels)
    sines = np.sin(time)
    sines = rint((sines * 128) + 128) # scale values to be in range 0-256 which corresponds with RGB pixel vals
    sines = sines.reshape(width, height)
    return np.dstack([sines, sines, sines])

    
def sine_mesh(side_length: int, density: int = 10) -> np.ndarray:
    sines = sine(side_length, side_length, density)
    return sines + np.rot90(sines)

    
def interp(start: np.ndarray, end: np.ndarray, n_interps: int) -> [np.ndarray]:
    step_size = 1 / n_interps
    c1 = lambda i: 1 - i * step_size
    c2 = lambda i: i * step_size
    return [rint(c1(i) * start + c2(i) * end) for i in range(n_interps)] 
    
    
def interp_sequence(frames: [np.ndarray], n_interps: int) -> [np.ndarray]:
    interpolated = []
    for a, b in zip(frames, frames[1:]):
        interpolated += interp(a, b, n_interps)
    return interpolated + [frames[-1]]


def reflect(items: []) -> []:
    return items + items[::-1]


def add_to_gif(to_add: np.ndarray, frames: [np.ndarray]) -> [np.ndarray]:
    return [f + to_add for f in frames]


def horizontal_bool_mask(width: int, height: int, n_strips: int) -> np.ndarray:
    
    strip_size = height // n_strips
    
    mask = np.ones(strip_size, dtype=bool)
    result = []

    for _ in range(n_strips):
        result.append(mask)
        mask = mask == False

    col = np.hstack(result)
    return np.repeat(col, width).reshape(width, height)
        

def vertical_bool_mask(width: int, height: int, n_strips: int) -> np.ndarray:

    strip_size = width // n_strips
    
    mask = np.ones(strip_size, dtype=bool)
    result = []

    for _ in range(n_strips):
        result.append(mask)
        mask = mask == False

    row = np.hstack(result)
    return row


def solid_circle_mask(width: int, margin: int) -> np.ndarray:
    radius = width//2 
    row = np.linspace(-radius, radius, width)
    x = np.vstack([row for _ in range(width)])
    y = x.T
    return (x**2 + y**2) < (radius-margin)**2
    

def ring_mask(width: int, thickness: int, margin: int=0) -> np.ndarray:
    outer_circle = solid_circle_mask(width, 0)
    inner_circle = solid_circle_mask(width, thickness)
    return np.logical_and(outer_circle, np.logical_not(inner_circle))


def stripes_interleave(img1: np.ndarray, img2: np.ndarray, n_strips: int) -> np.ndarray:
    w, h = img1.shape[0], img1.shape[1]
    return mask_interleave(img1, img2, horizontal_bool_mask(w, h, n_strips))


def solid_circle_interleave(img1: np.ndarray, img2: np.ndarray, margin: int) -> np.ndarray:
    w, h = img1.shape[0], img1.shape[1]
    return mask_interleave(img1, img2, solid_circle_mask(w, margin))
    

def ring_interleave(img1: np.ndarray, img2: np.ndarray, thickness: int, margin: int=0) -> np.ndarray:
    return mask_interleave(img1, img2, ring_mask(img1.shape[0], thickness, margin))


def mask_interleave(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    new1, new2 = deepcopy(img1), deepcopy(img2)
    new1[mask] = img2[mask]
    new2[mask] = img1[mask]
    return new1, new2


def rotate(img: np.ndarray, theta: int) -> np.ndarray:
    return scipy_rotate(img, -theta, reshape=False)


def rescale(img: np.ndarray, new_x: int, new_y: int) -> np.ndarray:
    img = Image.fromarray(img).resize((new_x,new_y), resample=Image.BOX)
    return image_to_numpy(img)

def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return filter2D(img, -1, kernel)


def outline(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1],
                       [-1, 8,-1],
                       [-1, -1, -1]])
    return filter2D(img, -1, kernel)


def blur(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])
    return filter2D(img, -1, kernel)


def white(w: int = 1000, h: int = 1000) -> np.ndarray:
    channel = np.ones(shape=[w, h]) * 255
    return rint(np.dstack([channel, channel, channel]))


def c_channel_wave_scale(A=1, f=1, phase=0, duration=5, framerate=30):
    n_samples = duration*framerate
    X = np.linspace(0, duration, n_samples)
    return (0.5 * A * np.sin(f * X - 0.25 * np.pi) + 1).reshape(n_samples, 1, 1)


def colour_flow(img, duration=5, framerate=30):
    red = rint(img[:,:,0] * c_channel_wave_scale(A=0.2, duration=duration, framerate=framerate))
    green = rint(img[:,:,1] * c_channel_wave_scale(A=0.5, phase=0.5*np.pi, duration=duration, framerate=framerate))
    blue = rint(img[:,:,2] * c_channel_wave_scale(A=0.8, phase=1.5*np.pi, duration=duration, framerate=framerate))
    return np.stack([red, green, blue], axis=3)


def clipadd(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    img1 = img1.astype(np.uint64)
    img2 = img2.astype(np.uint64)
    result = img1 + img2
    clipping = result > 255
    result[clipping] = 255 * np.tanh(result[clipping])
    return rint(result)


def clipprod(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    img1 = img1.astype(np.uint64)
    img2 = img2.astype(np.uint64)
    result = img1 * img2
    clipping = result > 255
    result[clipping] = (255 * np.tanh(result[clipping])).astype(np.uint8)
    return rint(result)

def mandala(img) -> np.ndarray:
    upper_right = img
    lower_right = x_flip(upper_right)
    upper_left = y_flip(upper_right)
    lower_left = x_flip(upper_left)
    mandala = rint(0.25 * upper_right + 0.25 * lower_right + 0.25 * upper_left + 0.25 * lower_left)
    mandala = rint(0.5 * mandala + 0.5 * rotate(mandala, 90))
    return mandala

def channel_stack(channel):
    return np.stack([channel, channel, channel], axis=2)

def greyscale(img) -> np.ndarray:
    mean = rint(np.mean(img, axis=2))
    return np.stack([mean, mean, mean], axis=2)


def flatten_dark(img: np.ndarray, threshold: int = 50):
    img[img < threshold] = 0


def red(img: np.ndarray) -> np.ndarray:
    red = img[:,:,0]
    others = np.zeros(img.shape[:-1])
    return np.stack([red, others, others])


def blue(img: np.ndarray) -> np.ndarray:
    blue = img[:,:,1]
    others = np.zeros(img.shape[:-1])
    return np.stack([others, blue, others])


def green(img: np.ndarray) -> np.ndarray:
    green = img[:,:,2]
    others = np.zeros(img.shape[:-1])
    return np.stack([others, others, green])


def rbg_split(img: np.ndarray) -> np.ndarray:
    return red(img), green(img), blue(img)
