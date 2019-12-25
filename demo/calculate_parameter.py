import pyrealsense2 as rs
import numpy as np
import cv2
import calculate_rmsd_kabsch as rmsd


def set_someparameter(pipeline, pipeline_profile):
    path_to_settings_file = "./HighResHighAccuracyPreset.json"
    with open(path_to_settings_file, 'r') as file:
        json_text = file.read().strip()
    device = pipeline_profile.get_device()
    advanced_mode = rs.rs400_advanced_mode(device)
    advanced_mode.load_json(json_text)
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.laser_power, 350)
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
    # depth_sensor.set_option(rs.option.exposure, 25000)

    color_sensor = pipeline_profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    # color_sensor.set_option(rs.option.exposure, 1000)

    dispose_frames_for_stablisation = 30
    for cnt in range(dispose_frames_for_stablisation):
        frames = {}
        while len(frames) < 4:
            streams = pipeline_profile.get_streams()
            frameset = pipeline.poll_for_frames()  # frameset will be a pyrealsense2.composite_frame object
            if frameset.size() == len(streams):
                for stream in streams:
                    if (rs.stream.infrared == stream.stream_type()):
                        frame = frameset.get_infrared_frame(stream.stream_index())
                        key_ = (stream.stream_type(), stream.stream_index())
                    else:
                        frame = frameset.first_or_default(stream.stream_type())
                        key_ = stream.stream_type()
                    frames[key_] = frame
    return

def get_instrisics_frames(pipeline,pipeline_profile):
    dispose_frames_for_stablisation=30
    for cnt in range(dispose_frames_for_stablisation):
        frames = {}
        while len(frames)<4:
            streams = pipeline_profile.get_streams()
            frameset = pipeline.poll_for_frames()  # frameset will be a pyrealsense2.composite_frame object
            if frameset.size() == len(streams):
                for stream in streams:
                    if (rs.stream.infrared == stream.stream_type()):
                        frame = frameset.get_infrared_frame(stream.stream_index())
                        key_ = (stream.stream_type(), stream.stream_index())
                    else:
                        frame = frameset.first_or_default(stream.stream_type())
                        key_ = stream.stream_type()
                    frames[key_] = frame

    device_parameter = {}
    for key, value in frames.items():
        item = value.get_profile().as_video_stream_profile().get_intrinsics()
        device_parameter[key] = [item.width,item.height,item.ppx,item.ppy,item.fx,item.fy]
    extrinsics_device = frames[rs.stream.depth].get_profile().as_video_stream_profile().get_extrinsics_to(frames[rs.stream.color].get_profile())
    device_parameter["external_parameter"]= [np.asanyarray(extrinsics_device.rotation).reshape(3,3),np.asanyarray(extrinsics_device.translation)]
    parameter =[device_parameter[rs.stream.color],device_parameter[rs.stream.depth],device_parameter["external_parameter"][0],device_parameter["external_parameter"][1]]
    return parameter,frames

def post_process_depth_frame(depth_frame, decimation_magnitude=1.0, spatial_magnitude=2.0, spatial_smooth_alpha=0.5,
                             spatial_smooth_delta=20, temporal_smooth_alpha=0.4, temporal_smooth_delta=20):

    assert (depth_frame.is_depth_frame())

    # Available filters and control options for the filters
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    filter_magnitude = rs.option.filter_magnitude
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta

    # Apply the control parameters for the filter
    decimation_filter.set_option(filter_magnitude, decimation_magnitude)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)

    return filtered_frame

def cv_find_chessboard(infrared_image, chessboard_params):
    assert(len(chessboard_params) == 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_found = False
    chessboard_found, corners = cv2.findChessboardCorners(infrared_image, (chessboard_params[0], chessboard_params[1]))
    if chessboard_found:
        corners = cv2.cornerSubPix(infrared_image, corners, (11,11),(-1,-1), criteria)
        corners = np.transpose(corners, (2,0,1))
    return chessboard_found, corners

def get_point_2D_3D(chessboard_params,infrared_image,depth_frame,depth_intrinsics):
    #棋盘格角点检测, use the function provided by opencv
    found_corners, points2D = cv_find_chessboard(infrared_image,chessboard_params)
    corners3D = [found_corners, None, None, None]
    if found_corners:
        points3D = np.zeros((3, len(points2D[0])))
        validPoints = [False] * len(points2D[0])
        for index in range(len(points2D[0])):
            corner = points2D[:, index].flatten()
            depth = depth_frame.as_depth_frame().get_distance(round(corner[0]), round(corner[1]))
            if depth != 0 and depth is not None:
                validPoints[index] = True
                X = (corner[0] - depth_intrinsics[2]) / depth_intrinsics[4] * depth
                Y = (corner[1] - depth_intrinsics[3]) / depth_intrinsics[5] * depth
                Z = depth
                points3D[0, index] = X
                points3D[1, index] = Y
                points3D[2, index] = Z
        corners3D = [found_corners, points2D, points3D, validPoints]
    return corners3D

def get_world_points(chessboard_params):
    width = chessboard_params[0]
    height = chessboard_params[1]
    square_size = chessboard_params[2]
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp[:,2]=-0.005
    objectpoints = objp.transpose() * square_size
    return objectpoints

def calculate_transformation_kabsch(src_points, dst_points):

    assert src_points.shape == dst_points.shape
    #be sure the correct format of point_coordinate array
    if src_points.shape[0] != 3:
        raise Exception("The input data matrix had to be transposed in order to compute transformation.")

    src_points = src_points.transpose()
    dst_points = dst_points.transpose()

    src_points_centered = src_points - rmsd.centroid(src_points)
    dst_points_centered = dst_points - rmsd.centroid(dst_points)

    rotation_matrix = rmsd.kabsch(src_points_centered, dst_points_centered)
    rmsd_value = rmsd.kabsch_rmsd(src_points_centered, dst_points_centered)

    translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

    return rotation_matrix.transpose(), translation_vector.transpose(), rmsd_value

class Transformation:
    def __init__(self, rotation_matrix, translation_vector):
        self.pose_mat = np.zeros((4, 4))
        self.pose_mat[:3, :3] = rotation_matrix
        self.pose_mat[:3, 3] = translation_vector.flatten()
        self.pose_mat[3, 3] = 1

    def apply_transformation(self, points):
        """
        Applies the transformation to the pointcloud

        Parameters:
        -----------
        points : array
            (3, N) matrix where N is the number of points

        Returns:
        ----------
        points_transformed : array
            (3, N) transformed matrix
        """
        assert (points.shape[0] == 3)
        n = points.shape[1]
        points_ = np.vstack((points, np.ones((1, n))))
        points_trans_ = np.matmul(self.pose_mat, points_)
        points_transformed = np.true_divide(points_trans_[:3, :], points_trans_[[-1], :])
        return points_transformed

    def inverse(self):
        """
        Computes the inverse transformation and returns a new Transformation object

        Returns:
        -----------
        inverse: Transformation

        """
        rotation_matrix = self.pose_mat[:3, :3]
        translation_vector = self.pose_mat[:3, 3]

        rot = np.transpose(rotation_matrix)
        trans = - np.matmul(np.transpose(rotation_matrix), translation_vector)
        return Transformation(rot, trans)

def get_boundary_corners_2D(points):

    padding = 0.1
    if points.shape[0] == 3:
        assert (len(points.shape) == 2)
        minPt_3d_x = np.amin(points[0, :])
        maxPt_3d_x = np.amax(points[0, :])
        minPt_3d_y = np.amin(points[1, :])
        maxPt_3d_y = np.amax(points[1, :])

        boudary = [minPt_3d_x - padding, maxPt_3d_x + padding, minPt_3d_y - padding, maxPt_3d_y + padding]

    else:
        raise Exception("wrong dimension of points!")

    return boudary

def get_boundary_in_color(roi_2D,Trans,extrinsics_device,color_internal):
    roi_color = np.zeros((3,4))  #roi_2D[x-,x+,y-,y+]
    roi_color[0,:2]=roi_2D[0]
    roi_color[0,2:4]=roi_2D[1]
    roi_color[1,[0,3]]=roi_2D[2]
    roi_color[1,[1,2]]=roi_2D[3]
    roi_color[2,:]=0
    transform = Transformation(Trans[4],Trans[5])
    roi_color = transform.apply_transformation(roi_color)
    roi_color = roi_color.transpose().tolist()
    color_pixel = []
    for point in roi_color:
        cloud_point = rs.rs2_transform_point_to_point(extrinsics_device,point)
        color_pixel.append(rs.rs2_project_point_to_pixel(color_internal, cloud_point))
    color_boundary_point = np.row_stack(color_pixel).T
    return color_boundary_point

def calibrate_camera(pipeline,pipeline_profile):
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    parameter, frames = get_instrisics_frames(pipeline, pipeline_profile)
    extrinsics_device = frames[rs.stream.depth].get_profile().as_video_stream_profile().get_extrinsics_to(
        frames[rs.stream.color].get_profile())
    color_internal = frames[rs.stream.color].get_profile().as_video_stream_profile().get_intrinsics()
    chessboard_width = 9  # squares
    chessboard_height = 6  # squares
    square_size = 0.026  # m
    chessboard_params = [chessboard_height, chessboard_width, square_size]
    infrared_image = np.asanyarray(frames[(rs.stream.infrared, 1)].get_data())
    depth_frame = post_process_depth_frame(frames[rs.stream.depth])
    p_points = get_point_2D_3D(chessboard_params, infrared_image, depth_frame, parameter[1])
    while not p_points[0]:
        print("can't find the chessboard,please try again")
        return False,None,None
    w_points = get_world_points(chessboard_params)
    valid_object_points = w_points[:, p_points[3]]
    valid_observed_object_points = p_points[2][:, p_points[3]]
    if valid_object_points.shape[1] < 5:
        print("No enough points have a valid depth for calculating the transformation")
        return False,None,None
    else:
        [rotation_matrix, translation_vector, rmsd_value] = calculate_transformation_kabsch(valid_object_points,valid_observed_object_points)
        trans0 = [True, rotation_matrix, translation_vector, rmsd_value]
        print("RMS error for calibration with device number", "is :", rmsd_value, "m")
    if (trans0[0] != True):
        print("faild to calibration the camera")
        return False,None,None

    #             0彩色内参    1深度内参    2深度转彩色旋转    3彩色转深度平移     4旋转    5平移
    Trans0 = [parameter[0], parameter[1], parameter[2], parameter[3],trans0[1], trans0[2]]
    transformation = Transformation(Trans0[4], Trans0[5]).inverse()
    chessboard_points_cumulative_3d = p_points[2][:, p_points[3]]
    chessboard_points_cumulative_3d = transformation.apply_transformation(chessboard_points_cumulative_3d)
    roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)
    roi_2D_color = np.int0(get_boundary_in_color(roi_2D, Trans0,extrinsics_device,color_internal))
    roi = [roi_2D, roi_2D_color]
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    return True, Trans0, roi

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
    frames = {}
    while len(frames) < 4:
        streams = pipeline_profile.get_streams()
        frameset = pipeline.poll_for_frames()  # frameset will be a pyrealsense2.composite_frame object
        if frameset.size() == len(streams):
            for stream in streams:
                if (rs.stream.infrared == stream.stream_type()):
                    frame = frameset.get_infrared_frame(stream.stream_index())
                    key_ = (stream.stream_type(), stream.stream_index())
                else:
                    frame = frameset.first_or_default(stream.stream_type())
                    key_ = stream.stream_type()
                frames[key_] = frame
    color_image = np.asanyarray(frames[rs.stream.color].get_data())
    cv2.namedWindow("color_image",0)
    cv2.imshow("color_image",color_image)
    key=cv2.waitKey(1)

    if key==ord(" "):
        label, parameter, boundary = calibrate_camera(pipeline, pipeline_profile)
        if not label:
            print("faild")
            continue
        else:
            parameters = [parameter, boundary]
            np.save("parameters.npy", parameters)
            print("succeed in sth")
            break



