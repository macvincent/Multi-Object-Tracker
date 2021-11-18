# Helper functions performing calculations mostly adapted from various open source code bases
import numpy as np
import cv2
from load_flag_values import *
from filterpy.kalman import KalmanFilter

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio.
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
  """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right.
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

def draw_image(image, bounding_boxes):
  """
    Given an image, draw bounding boxes around and display the image.
  """
  for bounding_box in bounding_boxes:
    (x_min, x_max, y_min, y_max) = (int(bounding_box[0] * length), int(bounding_box[2] * length),
                                    int(bounding_box[1] * length), int(bounding_box[3] * length))
    v1 = (x_min, y_max)
    v2 = (x_max, y_min)
    color = (255, 100, 255)
    thickness = 4
    cv2.rectangle(image, v1, v2, color, thickness)

  if use_camera or display_image:
    cv2.imshow("Image Display", image)

  return image

def IOU3D(tracks, box2):
  """
    We assume that each bounding box b = (top left x, top left y, width, height).
    Both tracks and box2 are lists of bounding boxes.
  """
  iou_2d = []
  for box1 in tracks:
    x1, y1, x2, y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x3, y3, x4, y4 = box2[:,0], box2[:,1], box2[:,0] + box2[:,2], box2[:,1] + box2[:,3]
    x_inter1 = np.maximum(x1, x3)
    y_inter1 = np.maximum(y1, y3)
    x_inter2 = np.minimum(x2, x4)
    y_inter2 = np.minimum(y2, y4)

    width_inter = np.abs(x_inter2 - x_inter1)
    height_inter = np.abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter

    width_box1 = np.abs(x2 - x1)
    height_box1 = np.abs(y2 - y1)
    width_box2 = np.abs(x4 - x3)
    height_box2 = np.abs(y4 - y3)

    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    area_union = area_box1 + area_box2 - area_inter

    iou = area_inter / area_union
    iou_2d.append(iou)
  return np.asarray(iou_2d)

def IOU2D(track, candidates):
  """
    We assume that each bounding box b = (top left x, top left y, width, height).
    track is a bouding box, while candidates is an array of bounding boxes.
  """
  result = IOU3D(np.asarray([track]), np.asarray(candidates))[0]
  return result

def IOU(track, candidate):
  """
    We assume that each bounding box b = (top left x, top left y, width, height).
    track and candidate are both bounding boxes.
  """
  result = IOU3D(np.asarray([track]), np.asarray([candidate]))[0]
  return result

object_id = 0
class TrackedObject:
  """
    Kalman Filter Implementation,
    adapted from https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py.
  """
  def __init__(self, bounding_box, category):
    global object_id
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bounding_box)
    self.time_since_update = 0
    self.age = 0
    self.category = category
    self.id = object_id
    object_id += 1

  def update(self,bbox):
    """
      Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.kf.update(convert_bbox_to_z(bbox))
  
  def predict(self):
    """
      Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    self.time_since_update += 1
    return convert_x_to_bbox(self.kf.x)

  def get_state(self):
    """
      Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)[0]

def load_camera():
    '''Returns a VideoCapture object that gives access to the device camera or file path'''
    video_capture = cv2.VideoCapture(file_path)

    if use_camera:
      video_capture = cv2.VideoCapture(0)

    video_capture.set(3,length)
    video_capture.set(4,length)
    return video_capture