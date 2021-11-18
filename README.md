# Multi-Object Tracker
This is an implementation of the [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763) algorithm by Alex Bewley et al., 2016. It was a component of a broader class project. To detect bounding boxes, we use the  [TensoFlow Hub SSD+MobileNetV2]("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1") object detection model to identify bounding boxes for the objects in the current image frame. Then, we use a [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) to predict the positions of objects in the next time step. We use SciPy's [`linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) method to perform [`minimum weight matching`](https://en.wikipedia.org/wiki/Assignment_problem) and association over the [`Intersection Over Union`](https://en.wikipedia.org/wiki/Jaccard_index) cost between the predicted object states and measured objects states. We update the bounding boxes of tracked objects, drop objects that are out of frame, and track new objects that come into the frame.

This is, however, far from the start of the art. For a more robust multi-object tracker that also handles image re-identification after long periods of occlusion see [Deep Sort](https://github.com/nwojke/deep_sort).


## Running the tracker
1. Clone the project repository

```
git clone https://github.com/macvincent/Multi-Object-Tracker.git
```
2. Install required dependencies, preferably from a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment), by running:

```
pip install -r requirements.txt
```
3. Start tracker by running this command:
```
python3 object_tracker.py [--flags]
```
We have the following flags available:
Flag | Type | Description | Default Value
--- | --- | --- | ---
video_path | string | Path to video you can use to test tracker. | Path to video in the test_videos folder.
fps | int | Set frame per second of test video. | 20
use_camera | boolean | Track images using your camera. | False
display_image | boolean | Create OpenCV display of objects being tracked. | False
length | int | Modify length of image frame. | 640 px