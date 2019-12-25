import pyrealsense2 as rs
import numpy as np
import cv2
from math import *

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

def get_primary_photo(pipeline,pipeline_profile):
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
    filtered_depth_frame = post_process_depth_frame(frames[rs.stream.depth], temporal_smooth_alpha=0.1,
                                                    temporal_smooth_delta=80)
    depth_data = np.asarray(filtered_depth_frame.get_data())
    return color_image,depth_data,frames

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics[2]) / camera_intrinsics[4]
    y = (v.flatten() - camera_intrinsics[3]) / camera_intrinsics[5]

    z = depth_image.flatten() / 1000.0;
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    # x = x[np.nonzero(z)]
    # y = y[np.nonzero(z)]
    # z = z[np.nonzero(z)]

    return x, y, z

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

def get_clipped_pointcloud(pointcloud, boundary):

    assert (pointcloud.shape[0] >= 2)


    pointcloud[:,np.logical_or(pointcloud[0,:]<boundary[0],pointcloud[0,:]>boundary[1])]=0
    pointcloud[:,np.logical_or(pointcloud[1,:]<boundary[2],pointcloud[1,:]>boundary[3])]=0
    return pointcloud

def calculate_boundingbox_points(pipeline,pipeline_profile,point_cloud, Trans,extrinsics_device,color_internal):
    dimensions = []
    color_images = []
    depth_value0 = point_cloud[2].reshape(720,1280)
    if len(depth_value0>0.001)>500:
        depth_value1 = np.int0(np.where(depth_value0<0.005,0,depth_value0)*1000)

        mask = np.where(depth_value1>0,255,depth_value1).astype(np.uint8)


        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_0 = [];
        # 寻找满足的轮
        for j in range(len(contours)):
            cnt = contours[j]
            area = cv2.contourArea(cnt)
            length = cv2.arcLength(cnt, True)
            if area > 100 and length > 200:
                contours_0.append(contours[j])


        #cv2.drawContours(mask,contours_0,-1,[255],3)

        for i in range(len(contours_0)):
            index = []
            for j in range(len(contours_0[i])):
                x=contours_0[i][j][0][0]
                y=contours_0[i][j][0][1]
                index.append(x+y*1280)

            hull = cv2.convexHull(contours_0[i])
            angle_0 = cv2.minAreaRect(hull)[2]
            point_cloud0 = point_cloud[:,index]
            if point_cloud0.shape[1] > 100:
                coord = np.c_[point_cloud0[0, :], point_cloud0[1, :]].astype('float32')
                min_area_rectangle = cv2.minAreaRect(coord)
                angle_1 = min_area_rectangle[2]
                bounding_box_world_2d = cv2.boxPoints(min_area_rectangle)
                # print("debug111")
                # print(bounding_box_world_2d)

                height = max(point_cloud0[2, :])  # - min(point_cloud[2, :]) + depth_threshold
                height_array = np.array([[-height], [-height], [-height], [-height], [0], [0], [0], [0]])
                bounding_box_world_3d = np.column_stack((np.row_stack((bounding_box_world_2d, bounding_box_world_2d)), height_array))
                trans = Transformation(Trans[4],Trans[5])
                bounding_box_device_3d = trans.apply_transformation(bounding_box_world_3d.transpose())#

                # print("debug222")
                # print(bounding_box_device_3d)

                # trans = Transformation(Trans[2],Trans[3])
                # bounding_box_color_image_point=trans.apply_transformation(bounding_box_device_3d)
                # print("debug333")
                # print(bounding_box_color_image_point) #
                #
                # color_pixel=get_point_3D_2D(bounding_box_color_image_point, Trans[0]).T
                # color_images.append(color_pixel)
                # print("debug444")
                # print(color_pixel)
                #

                bounding_box_device_3d = bounding_box_device_3d.transpose().tolist()
                color_pixel = []
                for bounding_box_point in bounding_box_device_3d:
                    bounding_box_color_image_point = rs.rs2_transform_point_to_point(extrinsics_device,
                                                                                     bounding_box_point)
                    color_pixel.append(
                        rs.rs2_project_point_to_pixel(color_internal, bounding_box_color_image_point))
                color_image = np.row_stack(color_pixel)
                color_images.append(color_image)
                dimensions.append([min_area_rectangle[1][0],min_area_rectangle[1][1],height,angle_0])


        #
        # cv2.imshow("depth11", mask)
        # cv2.waitKey(1)
        return color_images, dimensions
    else:
        return color_images, dimensions

def  resize_picture(img):
    hight=np.shape(img)[0]
    width=np.shape(img)[1]
    if width>hight:
        img_0 = np.zeros([width,width,3],np.uint8)
        img_0[int((width-hight)/2):(int((width-hight)/2)+hight),0:width]=img
    else:
        img_0 = np.zeros([hight,hight,3],np.uint8)
        img_0[0:hight,int((hight - width) / 2):(int((hight - width) / 2)+width)]=img
    return img_0

def segmentation(primary_color_image,box_point,angle):
    box_point[:,0]=np.where(box_point[:,0]<0,0,box_point[:,0])
    box_point[:,0] = np.where(box_point[:, 0]>1279, 1279, box_point[:, 0])
    box_point[:,1] = np.where(box_point[:, 1] < 0, 0, box_point[:, 1])
    box_point[:,1] = np.where(box_point[:, 1] > 719, 719, box_point[:, 1])
    box_point = np.int0(box_point)
    angle = cv2.minAreaRect(box_point)[2]
    #print(angle,angle0)
    Xs = [i[0] for i in box_point]
    Ys = [i[1] for i in box_point]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    # print(hight,width,x1,x2,y1,y2)
    if hight==0 or width ==0:
        return 0,0
    # 提取矩形区域
    else:
        aimImg = primary_color_image[y1:y1 + hight, x1:x1 + width]
        # 坐标转换
        box = np.vstack((box_point.T, np.array([1, 1, 1, 1])))
        box = np.dot([[1, 0, -x1], [0, 1, -y1]], box).T


        heightNew = int(hight * cos(radians(-angle)) + width * sin(radians(-angle)))
        widthNew = int(hight * sin(radians(-angle)) + width * cos(radians(-angle)))

        #旋转矩阵计算
        matRotation = cv2.getRotationMatrix2D((width / 2, hight / 2), angle, 1)
        matRotation[0][2] = matRotation[0][2] + (widthNew - width) / 2
        matRotation[1][2] = matRotation[1][2] + (heightNew - hight) / 2
        # 计算旋转矩阵
        box = np.vstack((box.T, np.array([1, 1, 1, 1])))
        box = np.dot(matRotation, box).T
        # 数据归一化
        box = np.int0(box)
        # print(np.shape(aimImg))
        # cv2.imshow("dsaf",aimImg)
        # 图像旋转
        imgRotation = cv2.warpAffine(aimImg, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
        # 计算区域坐标
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = max(min(Xs),0)
        x2 = min(max(Xs),widthNew)
        y1 = max(min(Ys),0)
        y2 = min(max(Ys),heightNew)
        hight = y2 - y1
        width = x2 - x1
        # 提取区域
        aimImg = imgRotation[y1:y1 + hight, x1:x1 + width]
        # 使长边水平
        hight = np.shape(aimImg)[0]
        width = np.shape(aimImg)[1]
        if width < hight:
            matRotation = cv2.getRotationMatrix2D((width / 2, hight / 2), 90, 1)
            matRotation[0][2] += (hight - width) / 2
            matRotation[1][2] += (width - hight) / 2
            aimImg = cv2.warpAffine(aimImg, matRotation, (hight, width), borderValue=(0, 0, 0))
        return aimImg,1

def visualise_measurements(pipeline,pipeline_profile,color_image, bounding_box_points, roi_2D_color,dimensions):
    for i in range(len(roi_2D_color)):
        cv2.line(color_image,tuple(roi_2D_color[i]),tuple(roi_2D_color[(i+1)%4]),(0,0,255),1)
    for i in range(len(bounding_box_points)):
        length = dimensions[i][0]
        width = dimensions[i][1]
        height = dimensions[i][2]
        box_points =bounding_box_points[i]
        if (length != 0 and width != 0 and height != 0):
            bounding_box_points_device_upper = box_points[0:4, :]
            bounding_box_points_device_lower = box_points[4:8, :]
            box_info = "__"+str(i)+"Length, Width, Height (mm): " + str(int(length * 1000)) + ", " + str(int(width * 1000)) + ", " + str(int(height * 1000))


            #Draw the box as an overlay on the color image
            bounding_box_points_device_upper = tuple(map(tuple, bounding_box_points_device_upper.astype(int)))
            for j in range(len(bounding_box_points_device_upper)):
                cv2.line(color_image, bounding_box_points_device_upper[j],bounding_box_points_device_upper[(j + 1) % 4], (0, 255, 0), 3)

            bounding_box_points_device_lower = tuple(map(tuple, bounding_box_points_device_lower.astype(int)))
            for j in range(len(bounding_box_points_device_upper)):
                cv2.line(color_image, bounding_box_points_device_lower[j],bounding_box_points_device_lower[(j + 1) % 4], (0, 255, 0), 1)

            cv2.line(color_image, bounding_box_points_device_upper[0], bounding_box_points_device_lower[0], (0, 255, 0),
                     1)
            cv2.line(color_image, bounding_box_points_device_upper[1], bounding_box_points_device_lower[1], (0, 255, 0),
                     1)
            cv2.line(color_image, bounding_box_points_device_upper[2], bounding_box_points_device_lower[2], (0, 255, 0),
                     1)
            cv2.line(color_image, bounding_box_points_device_upper[3], bounding_box_points_device_lower[3], (0, 255, 0),
                     1)
            cv2.putText(color_image, box_info, (50, 50+30*i), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

def get_handled_photo(pipeline,pipeline_profile,frames,color_image,depth_data,Trans,roi):
    depth_threshold=0.003
    extrinsics_device = frames[rs.stream.depth].get_profile().as_video_stream_profile().get_extrinsics_to(
        frames[rs.stream.color].get_profile())
    color_internal = frames[rs.stream.color].get_profile().as_video_stream_profile().get_intrinsics()
    point_cloud = convert_depth_frame_to_pointcloud(depth_data, Trans[1])
    point_cloud = np.asanyarray(point_cloud)

    trans = Transformation(Trans[4], Trans[5]).inverse()
    point_cloud = trans.apply_transformation(point_cloud)
    point_cloud[2] = np.maximum(point_cloud[2] * (-1), 0)
    point_cloud[2] = np.where(point_cloud[2] > 0.35, 0, point_cloud[2])
    point_cloud = get_clipped_pointcloud(point_cloud, roi[0])
    point_cloud[2] = np.where(point_cloud[2] < depth_threshold, 0, point_cloud[2])
    bounding_box_points_color_image, box_dimensions = calculate_boundingbox_points(pipeline, pipeline_profile,
                                                                                   point_cloud, Trans,
                                                                                   extrinsics_device, color_internal)
    RGB_dimension = []
    if len(bounding_box_points_color_image) != 0:
        for i in range(len(bounding_box_points_color_image)):
            color_image0 = color_image.copy()
            image, label = segmentation(color_image0, bounding_box_points_color_image[i][0:4],
                                        box_dimensions[i][3])
            if label != 0:
                image = resize_picture(image)
                image = cv2.resize(image, (320, 320))
                temp = box_dimensions[i][0:2]
                temp.sort(reverse=True)
                box_dimensions[i][0:2] = temp
                RGB_dimension.append([image, box_dimensions[i]])
    visualise_measurements(pipeline, pipeline_profile, color_image, bounding_box_points_color_image, roi[1].T,
                           box_dimensions)
    if len(RGB_dimension) != 0:
        return True, RGB_dimension, color_image
    else:
        return False, RGB_dimension, color_image