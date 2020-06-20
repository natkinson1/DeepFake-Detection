import numpy as np
import torch
import cv2
from facenet_pytorch import MTCNN
from PIL import Image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def face_cropper(img_array):
    
    '''Crops an image to only have the face.'''
    face_array = []
    
    
    boxes, probs = mtcnn.detect(img_array)
    
    if boxes is None:
        
        return Image.fromarray(img_array)
    
    for box in boxes:
        
        width_mid = int((box[2] + box[0]) / 2)
        height_mid = int((box[3] + box[1]) / 2)
        
        width = [i for i in range(max(0, width_mid - 112), min(1920, width_mid + 112))]
        height = [i for i in range(max(0, height_mid - 112), min(1080, height_mid + 112))]
        
        face_array.append(Image.fromarray(img_array[:,width][height]))
        #face_array.append(img_array[:,width][height])
        
    return face_array[0]

def random_frame_selector(video_source):
    
    video = cv2.VideoCapture(video_source)

    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    random_frame = np.random.randint(0, video_length)

    video.set(1, random_frame)

    _, frame = video.read()
    
    return frame
    

class FaceCropper:
    
    def __call__(self, img):
        
        #Only want one face, may look to change in the future.
        return face_cropper(img)
    
class RandomFrameSelector:
    
    def __call__(self, video):
        
        return random_frame_selector(video)