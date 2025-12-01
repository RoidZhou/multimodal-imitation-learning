#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque 
import imageio
import pyrealsense2 as rs
from multiprocessing import Process, Pipe, Queue, Event
import time
import multiprocessing
from matplotlib.animation import FuncAnimation
multiprocessing.set_start_method('fork')

np.printoptions(3, suppress=True)

rgb_h = 480
rgb_w = 640
depth_h = 480
depth_w = 640
def get_realsense_id():
    ctx = rs.context()
    devices = ctx.query_devices()
    devices = [devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))]
    devices.sort() # Make sure the order is correct
    print("Found {} devices: {}".format(len(devices), devices))
    return devices

def init_given_realsense(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        h, w = depth_h, depth_w
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        h, w = rgb_h, rgb_w
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        align = rs.align(rs.stream.color)  # 如果分辨率一致可改为 rs.align(rs.stream.depth)

        # 多机同步和min_distance设置（除非需要）
        # depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
        # depth_sensor.set_option(rs.option.min_distance, 0.05)

        # 获取相机内参（仅用于点云生成）
        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx,
                                 intrinsics.ppy)
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None


def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                   The first 3 columns represent the coordinates (x, y, z).
                   The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    coords = point_cloud[:, :3]  # Extract coordinates
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)
    
    # Create unique grid keys
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    
    # Select unique points based on grid keys
    _, indices = np.unique(keys, return_index=True)
    
    # Return sampled points
    return point_cloud[indices]


class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale = 1) :
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale
        
class SingleVisionProcess(Process):
    def __init__(self, device, queue,
                enable_rgb=True,
                enable_depth=False,
                enable_pointcloud=False,
                sync_mode=0,
                num_points=2048,
                z_far=1.0,
                z_near=0.1,
                use_grid_sampling=True,
                img_size=224) -> None:
        super(SingleVisionProcess, self).__init__()
        self.queue = queue
        self.device = device

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_pointcloud = enable_pointcloud
        self.sync_mode = sync_mode
            
        self.use_grid_sampling = use_grid_sampling

  
        self.resize = True
        # self.height, self.width = 512, 512
        self.height, self.width = img_size, img_size
        
        # point cloud params
        self.z_far = z_far
        self.z_near = z_near
        self.num_points = num_points
   
    def get_vision(self):
        frame = self.pipeline.wait_for_frames()

        if self.enable_depth:
            aligned_frames = self.align.process(frame)
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
    
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
            
            clip_lower =  0.01
            clip_high = 1.0
            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale
            depth_frame[depth_frame < clip_lower] = clip_lower
            depth_frame[depth_frame > clip_high] = clip_high
            
            if self.enable_pointcloud:
                # Nx6
                point_cloud_frame = self.create_colored_point_cloud(color_frame, depth_frame,
                            far=self.z_far, near=self.z_near, num_points=self.num_points)
            else:
                point_cloud_frame = None
        else:
            color_frame = frame.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = None
            point_cloud_frame = None

        # print("color:", color_frame.shape)
        # print("depth:", depth_frame.shape)
        
        if self.resize:
            if self.enable_rgb:
                color_frame = cv2.resize(color_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.enable_depth:
                depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return color_frame, depth_frame, point_cloud_frame


    def run(self):
        self.pipeline, self.align, self.depth_scale, self.camera_info = init_given_realsense(self.device, 
                    enable_rgb=self.enable_rgb, enable_depth=self.enable_depth,
                    enable_point_cloud=self.enable_pointcloud,
                    sync_mode=self.sync_mode)

        debug = False
        while True:
            color_frame, depth_frame, point_cloud_frame = self.get_vision()
            self.queue.put([color_frame, depth_frame, point_cloud_frame])
            time.sleep(0.05)

    def terminate(self) -> None:
        # self.pipeline.stop()
        return super().terminate()

   
    def create_colored_point_cloud(self, color, depth, far=1.0, near=0.1, num_points=10000):
        assert(depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1])
    
        # Create meshgrid for pixel coordinates
        xmap = np.arange(color.shape[1])
        ymap = np.arange(color.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        # Calculate 3D coordinates
        points_z = depth / self.camera_info.scale
        points_x = (xmap - self.camera_info.cx) * points_z / self.camera_info.fx
        points_y = (ymap - self.camera_info.cy) * points_z / self.camera_info.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = cloud.reshape([-1, 3])
        
        # Clip points based on depth
        mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
        cloud = cloud[mask]
        color = color.reshape([-1, 3])
        color = color[mask]


        colored_cloud = np.hstack([cloud, color.astype(np.float32)])
        if self.use_grid_sampling:
            colored_cloud = grid_sample_pcd(colored_cloud, grid_size=0.005)
        
        if num_points > colored_cloud.shape[0]:
            num_pad = num_points - colored_cloud.shape[0]
            pad_points = np.zeros((num_pad, 6))
            colored_cloud = np.concatenate([colored_cloud, pad_points], axis=0)
        else: 
            # Randomly sample points
            selected_idx = np.random.choice(colored_cloud.shape[0], num_points, replace=True)
            colored_cloud = colored_cloud[selected_idx]
        
        # shuffle
        np.random.shuffle(colored_cloud)
        return colored_cloud


    
class MultiRealSense(object):
    def __init__(self, use_front_cam=True, use_right_cam=False,
                 front_cam_idx=0, right_cam_idx=1, 
                 front_num_points=4096, right_num_points=1024,
                 front_z_far=1.0, front_z_near=0.1,
                 right_z_far=0.5, right_z_near=0.01,
                 use_grid_sampling=True,
                 img_size=384):

        self.devices = get_realsense_id()
    
        self.front_queue = Queue(maxsize=30)
        self.right_queue = Queue(maxsize=30)

      
        # 0: f1380328, 1: f1422212

        # sync_mode: Use 1 for master, 2 for slave, 0 for default (no sync)

        if use_front_cam:
            self.front_process = SingleVisionProcess(self.devices[front_cam_idx], self.front_queue,
                            enable_rgb=True, enable_depth=True, enable_pointcloud=True, sync_mode=0,
                            num_points=front_num_points, z_far=front_z_far, z_near=front_z_near, 
                            use_grid_sampling=use_grid_sampling, img_size=img_size)
        if use_right_cam:
            self.right_process = SingleVisionProcess(self.devices[right_cam_idx], self.right_queue,
                    enable_rgb=True, enable_depth=True, enable_pointcloud=True, sync_mode=0,
                        num_points=right_num_points, z_far=right_z_far, z_near=right_z_near, 
                        use_grid_sampling=use_grid_sampling,  img_size=img_size)


        if use_front_cam:
            self.front_process.start()
            print("front camera start.")

        if use_right_cam:
            self.right_process.start()
            print("right camera start.")

        self.use_front_cam = use_front_cam
        self.use_right_cam = use_right_cam
        
        
    def __call__(self):  
        cam_dict = {}
        if self.use_front_cam:  
            front_color, front_depth, front_point_cloud = self.front_queue.get()
            cam_dict.update({'color': front_color, 'depth': front_depth, 'point_cloud':front_point_cloud})
      
        if self.use_right_cam: 
            right_color, right_depth, right_point_cloud = self.right_queue.get()
            cam_dict.update({'right_color': right_color, 'right_depth': right_depth, 'right_point_cloud':right_point_cloud})
        return cam_dict

    def finalize(self):
        if self.use_front_cam:
            self.front_process.terminate()
        if self.use_right_cam:
            self.right_process.terminate()


    def __del__(self):
        self.finalize()


if __name__ == "__main__":
    cam = MultiRealSense(use_right_cam=True, front_num_points=20000, use_grid_sampling=True)
    import matplotlib.pyplot as plt

    # 启用Matplotlib交互模式
    plt.ion()

    # 创建 2x2 的子图布局（左彩色、左深度、右彩色、右深度）
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # 初始化图像
    left_color_plot = ax1.imshow(np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8))
    left_depth_plot = ax2.imshow(np.zeros((depth_h, depth_w), dtype=np.uint8), cmap='jet')
    right_color_plot = ax3.imshow(np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8))
    right_depth_plot = ax4.imshow(np.zeros((depth_h, depth_w), dtype=np.uint8), cmap='jet')

    # 设置标题
    ax1.set_title("Left Color")
    ax2.set_title("Left Depth")
    ax3.set_title("Right Color")
    ax4.set_title("Right Depth")

    plt.tight_layout()  # 自动调整子图间距
    plt.show()

    while True:
        out = cam()

        # 更新左相机图像
        left_color_plot.set_array(out['color'])
        left_depth_colormap = cv2.normalize(
            out['depth'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        left_depth_colormap = cv2.applyColorMap(left_depth_colormap, cv2.COLORMAP_JET)[:, :, ::-1]
        left_depth_plot.set_array(left_depth_colormap)

        # 更新右相机图像
        right_color_plot.set_array(out['right_color'])
        right_depth_colormap = cv2.normalize(
            out['right_depth'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        right_depth_colormap = cv2.applyColorMap(right_depth_colormap, cv2.COLORMAP_JET)[:, :, ::-1]
        right_depth_plot.set_array(right_depth_colormap)

        # 强制刷新界面
        fig.canvas.flush_events()

        # 按ESC键退出
        if plt.waitforbuttonpress(0.001):
            if plt.gcf().canvas.key_press_event.key == 'escape':
                break
