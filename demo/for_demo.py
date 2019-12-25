import pyrealsense2 as rs
import numpy as np
import cv2
from for_demo_help import *


parameter_info = np.load("parameters.npy", allow_pickle=True)
parameter = parameter_info[0]
boundary = parameter_info[1]
pipeline = rs.pipeline()
resolution_width = 1280  # pixels
resolution_height = 720  # pixels
color_resolution_width = 1280  # pixels
color_resolution_height = 720  # p
frame_rate = 15  # fps
rs_config = rs.config()
rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8,
                             frame_rate)
rs_config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8,
                             frame_rate)
rs_config.enable_stream(rs.stream.color, color_resolution_width, color_resolution_height, rs.format.bgr8,
                             frame_rate)
pipeline_profile = pipeline.start(rs_config)
set_someparameter(pipeline, pipeline_profile)

while(1):
    color_image, depth_data, frames = get_primary_photo(pipeline, pipeline_profile)
    primary_image = color_image.copy()
    primary_depth = depth_data.copy()
    label_ok, RGB_dimension, color_image = get_handled_photo(pipeline, pipeline_profile,
                                                                           frames, color_image, depth_data,
                                                                           parameter, boundary)
    cv2.namedWindow("heheh",1)
    cv2.imshow("heheh",color_image)
    cv2.waitKey(1)
    if label_ok:

        ###
        #此处处理 RGB_dimension
        ###
        # for i in range(len(RGB_dimension)):
        #     cv2.namedWindow(str(i),1)
        #     cv2.imshow(str(i),RGB_dimension[i][0])
        #     cv2.waitKey(1)
        #     print(RGB_dimension[i][1])
        pass

pipeline.stop()