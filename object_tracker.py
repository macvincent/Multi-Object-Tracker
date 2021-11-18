"""
  An Implementation of the Simple Online and Realtime Tracking by Alex Bewley et al., 2016.
"""
import cv2
from load_flag_values import use_camera, file_path, frames_per_second, output_path, display_image
from helper_functions import load_camera
from tracked_object import Tracker
from moviepy.editor import VideoFileClip

if __name__ == "__main__":    
    tracker = Tracker()    
    if use_camera or display_image:
        video_capture = load_camera()
        if not video_capture.isOpened():
            print("Video file not opened")
        else:
            while True:
                return_val, temp_frame = video_capture.read()
                if return_val:
                    tracker.track_objects(temp_frame)
                else:
                    print("Ending Stream")
                    break

                if(cv2.waitKey(1) & 0xFF == ord("q")):
                    video_capture.release()
                    cv2.destroyAllWindows()
    else:
        clip1 = VideoFileClip(file_path)
        output_clip = clip1.fl_image(tracker.track_objects).set_fps(frames_per_second)
        output_clip.write_videofile(output_path, audio=False)