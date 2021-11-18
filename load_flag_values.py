# Read command line flags
import argparse

parser = argparse.ArgumentParser(description="Specify flag values for object tracker.")
parser.add_argument('--video_path', action='store', type=str, default="./test_videos/test_video_1.mp4")
parser.add_argument('--length', action='store', type=int, default=640)
parser.add_argument('--fps', action='store', type=int, default=20)
parser.add_argument('--output_path', action='store', type=str, default="./test_videos_output/output.mp4")
parser.add_argument('--use_camera', dest='use_camera', action='store_true')
parser.add_argument('--no-camera', dest='use_camera', action='store_false')
parser.set_defaults(use_camera=False)
parser.add_argument('--display_image', dest='display_image', action='store_true')
parser.add_argument('--do-not-display_image', dest='display_image', action='store_false')
parser.set_defaults(display_image=False)
args = parser.parse_args()

length = args.length
file_path = args.video_path
frames_per_second = args.fps
use_camera = args.use_camera
output_path = args.output_path
display_image = args.display_image