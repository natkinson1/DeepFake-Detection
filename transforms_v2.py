import numpy as np
import torch
import cv2
from facenet_pytorch import MTCNN
from PIL import Image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def face_cropper(img_array):

    '''Crops an image to only have the face.'''
    img_array = img_array.data.numpy()
    frame_len, x, y, _ = img_array.shape
    print(frame_len, x, y,)

    frames = []
    
    #Detect face using MTCNN facial model.
    total_boxes, probs = mtcnn.detect(img_array)
    
    #Crop each image to only include the face.
    for frame in range(frame_len):
        
        if total_boxes[frame] is None:
            
            #CenterCrop
            width_mid = y // 2
            height_mid = x // 2
            
            width = [i for i in range(max(0, width_mid - 112), min(y, width_mid + 112))]
            height = [i for i in range(max(0, height_mid - 112), min(x, height_mid + 112))]
            print(img_array[frame][:,width][height].astype(int).shape)
            frames.append(img_array[frame][:,width][height].astype(int))
            continue
        
        bbox = total_boxes[frame][0]
       
        #Locate the centre of the face
        width_mid = int((bbox[2] + bbox[0]) / 2)
        height_mid = int((bbox[3] + bbox[1]) / 2)

        #Get the columns and rows for where the face is in the array
        #Note min() and max() is used incase the face is located at the edges of the image array.
        width = [i for i in range(max(0, width_mid - 112), min(y, width_mid + 112))]
        height = [i for i in range(max(0, height_mid - 112), min(x, height_mid + 112))]
        
        #print(img_array[frame][:,width][height].astype(int).shape)
        frames.append(img_array[frame][:,width][height].astype(int))
    
    frames = torch.Tensor(np.array(frames, dtype=int))
    
    return frames

def random_frame_selector(video_source):

    frame = None

    while frame is None:

        video = cv2.VideoCapture(video_source)

        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        random_frame = np.random.randint(0, video_length)

        video.set(1, random_frame)

        _, frame = video.read()

    return frame

def normalise(video):
    
    x = video.data.numpy()
    
    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    
    x = (x - x_min)/(x_max - x_min)
    
    mean = np.array([0.43216, 0.394666, 0.37645])
    std = np.array([0.22803, 0.22145, 0.216989])

    x = (x - mean ) / std
     
    return x


class FaceCropper:

    def __call__(self, img):

        #Only want one face, may look to change in the future.
        return face_cropper(img)

class RandomFrameSelector:

    def __call__(self, video):

        return random_frame_selector(video)
    
class Normalise:
    
    def __call__(self, video):
        
        return normalise(video)
