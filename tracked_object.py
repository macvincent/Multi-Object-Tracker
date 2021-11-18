from helper_functions import TrackedObject, IOU2D, IOU3D, draw_image
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from load_flag_values import length

class Tracker:
  """
    Maintains the state of the objects being tracked.
  """
  def __init__(self):
    self.tracked_objects = []
    self.predicted_states = []
    self.object_detection_module = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures['default']

  def predict_next_state(self):
    """
      Predicts the position of the bounding boxes around our objects in the next time step.
    """
    self.predicted_states = []

    for tracked_object in self.tracked_objects:
      self.predicted_states.append(tracked_object.predict()[0])

  def get_updated_bounding_boxes(self):
    """
      Returns a list with the bounding boxes of the objects we are keeping track of.
    """
    updated_bounding_boxes = []
    for tracked_object in self.tracked_objects:
      updated_bounding_boxes.append(tracked_object.get_state())
    return updated_bounding_boxes

  def track_new_object(self, measured_bounding_box, bounding_box_category):
    """
      Given a bounding box and the category of the object in the bounding box,
      track object.
    """
    tracker = TrackedObject(measured_bounding_box, bounding_box_category)
    self.tracked_objects.append(tracker)
    self.predicted_states.append(measured_bounding_box)

  def update_measurement(self, current_bounding_boxes, current_box_categories):
    """
      Given current bounding boxes as identified by our model, perform association with the predicted 
      states of our tracked objects, identify new objects, and drop tracked objects that are out of the frame.
    """
    self.predict_next_state()

    # Get initial bounding boxes at start of tracking operation. 
    if len(self.predicted_states) == 0:
      self.track_new_object(current_bounding_boxes[0], current_box_categories[0])

      for index, measured_bounding_box in enumerate(current_bounding_boxes):
        bounding_box_category = current_box_categories[index]
        max_intersection = np.max(IOU2D(measured_bounding_box, self.predicted_states))
        # We try to control the level of intersection between new objects.
        if (max_intersection <= 0.3):
          self.track_new_object(measured_bounding_box, bounding_box_category)

    # Perform association between current_bounding_boxes and predicted states of objects. 
    cost_matrix = IOU3D(np.asarray(self.predicted_states), np.asarray(current_bounding_boxes))
    track_ind, measurement_ind = linear_sum_assignment(cost_matrix, True)

    # Update kalman filters of tracked objects with new bounding box measurement. 
    for track_id, meas_id in zip(track_ind, measurement_ind):
      self.tracked_objects[track_id].update(current_bounding_boxes[meas_id])
    
    # Drop tracked boxes not updated in this frame
    new_track_objects = []
    for tracked_object in self.tracked_objects:
      if tracked_object.time_since_update > 0:
        continue
      new_track_objects.append(tracked_object)

    # Add unassociated boxes as new objects
    self.tracked_objects = new_track_objects
    for i in range(len(current_bounding_boxes)):
      if i not in measurement_ind:
        self.track_new_object(current_bounding_boxes[i], current_box_categories[i])

    return self.get_updated_bounding_boxes()

  def identify_boxes(self, image):
    """
      Predict position of bounding boxes in image frame. Returns predicted bounding boxes.
    """
    image = image/255
    input = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis]
    return self.object_detection_module(input)

  def track_objects(self, image):
    """
      Uses image frame to update the position of tracked objects. 
    """
    image = cv2.resize(image, (length, length), interpolation = cv2.INTER_AREA)

    output = self.identify_boxes(image)
    detection_boxes = output["detection_boxes"]
    detection_scores = output["detection_scores"]
    labels = output["detection_class_entities"]
    bounding_boxes = []
    current_box_categories = []
    updated_bounding_boxes = []

    for index, bounding_box in enumerate(detection_boxes):
      label = str(labels[index].numpy())
      # Tracking a single object class leads to a more robust tracker.
      # if label == "Person" and detection_scores[index] >= 0.5:
      if detection_scores[index] >= 0.5:
        (x_min, x_max, y_min, y_max) = (int(bounding_box[1] * length), int(bounding_box[3] * length),
                                      int(bounding_box[0] * length), int(bounding_box[2] * length))
        bounding_boxes.append((x_min, y_min, x_max, y_max))
        current_box_categories.append(label)

    # TODO: Add try and except here to ensure tracker does not break mid-stream
    if len(bounding_boxes) != 0:
      updated_bounding_boxes = np.asarray(
        self.update_measurement(
          bounding_boxes, current_box_categories))/length

    output_image = draw_image(image, updated_bounding_boxes)
    return output_image