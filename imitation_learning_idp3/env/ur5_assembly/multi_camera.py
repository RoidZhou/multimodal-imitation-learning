import cv2
import numpy as np
import pyrealsense2 as rs
from multiprocessing import Process, Queue
import time
import math
import os
import multiprocessing

multiprocessing.set_start_method('fork')

IMG_WIDTH, IMG_HEIGHT = 640, 480
NUM_POINTS = 2048
Z_NEAR, Z_FAR = 0.1, 1.5
GRID_SIZE = 0.01


class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale=1.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def get_realsense_ids():
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    serials.sort()
    print(f"Found {len(serials)} devices: {serials}")
    return serials


def grid_sample_pcd(points, grid_size=0.01):
    coords = points[:, :3]
    scaled = np.floor(coords / grid_size).astype(int)
    keys = scaled[:, 0] + scaled[:, 1] * 10000 + scaled[:, 2] * 100000000
    _, indices = np.unique(keys, return_index=True)
    return points[indices]


class CameraProcess(Process):
    def __init__(self, serial, queue, cam_index):
        super().__init__()
        self.serial = serial
        self.queue = queue
        self.cam_index = cam_index

    def run(self):
        try:
            time.sleep(self.cam_index * 2)
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, IMG_WIDTH, IMG_HEIGHT, rs.format.z16, 30)
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)

            device = profile.get_device()
            # for sensor in device.query_sensors():
            #     if sensor.is_depth_sensor():
            #         sensor.set_option(rs.option.inter_cam_sync_mode, 0)

            depth_sensor = device.first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            camera_info = CameraInfo(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy, scale=depth_scale)

            print(f"[Cam {self.cam_index}] Camera {self.serial} initialized.")

            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

                point_cloud = self.create_point_cloud(color, depth, camera_info)

                if self.queue.full():
                    self.queue.get_nowait()
                self.queue.put((color, depth, point_cloud))
        except Exception as e:
            print(f"[Cam {self.cam_index}] Error: {e}")
        finally:
            try:
                pipeline.stop()
            except:
                pass

    def create_point_cloud(self, color, depth, camera_info):
        H, W = depth.shape
        xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))

        z = depth
        x = (xmap - camera_info.cx) * z / camera_info.fx
        y = (ymap - camera_info.cy) * z / camera_info.fy

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = color.reshape(-1, 3)

        valid = (z > Z_NEAR) & (z < Z_FAR)
        valid = valid.reshape(-1)
        points = points[valid]
        colors = colors[valid]

        pc = np.hstack((points, colors))
        pc = grid_sample_pcd(pc, grid_size=GRID_SIZE)

        if pc.shape[0] > NUM_POINTS:
            indices = np.random.choice(pc.shape[0], NUM_POINTS, replace=False)
            pc = pc[indices]
        elif pc.shape[0] < NUM_POINTS:
            pad = np.zeros((NUM_POINTS - pc.shape[0], 6))
            pc = np.vstack((pc, pad))

        return pc


class MultiRealSense:
    def __init__(self, front_cam_idx=0, right_cam_idx=1):
        self.serials = get_realsense_ids()
        self.processes = []
        self.queues = []

        self.front_queue = None
        self.right_queue = None
        self.front_serial = None
        self.right_serial = None

        for idx, serial in enumerate(self.serials):
            q = Queue(maxsize=2)
            p = CameraProcess(serial, q, idx)
            p.start()
            self.processes.append(p)
            self.queues.append(q)

            if idx == front_cam_idx:
                self.front_queue = q
                self.front_serial = serial
            if idx == right_cam_idx:
                self.right_queue = q
                self.right_serial = serial

    def __call__(self):
        cam_dict = {}
        if self.front_queue and not self.front_queue.empty():
            color, depth, pcd = self.front_queue.get()
            cam_dict.update({
                "front_color": color,
                "front_depth": depth,
                "front_pointcloud": pcd
            })
        if self.right_queue and not self.right_queue.empty():
            color, depth, pcd = self.right_queue.get()
            cam_dict.update({
                "right_color": color,
                "right_depth": depth,
                "right_pointcloud": pcd
            })
        return cam_dict

    def finalize(self):
        for p in self.processes:
            p.terminate()
            p.join()

    def __del__(self):
        self.finalize()


def tile_images(images):
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    blank = np.zeros_like(images[0])
    result_rows = []

    for r in range(rows):
        row_imgs = images[r * cols:(r + 1) * cols]
        if len(row_imgs) < cols:
            row_imgs += [blank] * (cols - len(row_imgs))
        result_rows.append(np.hstack(row_imgs))

    return np.vstack(result_rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    cam_system = MultiRealSense()
    latest = {}

    try:
        while True:
            images = []
            frames = cam_system()
            for k, v in frames.items():
                latest[k] = v

            for k, v in latest.items():
                if 'color' in k:
                    disp = cv2.resize(v, (IMG_WIDTH, IMG_HEIGHT))
                    label = "Front RGB" if "front" in k else "Right RGB"
                    cv2.putText(disp, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    images.append(disp)
                if 'depth' in k:
                    norm = np.clip(v / Z_FAR, 0, 1)
                    dcol = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    dcol = cv2.resize(dcol, (IMG_WIDTH, IMG_HEIGHT))
                    label = "Front Depth" if "front" in k else "Right Depth"
                    cv2.putText(dcol, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    images.append(dcol)

            if images:
                canvas = tile_images(images)
                cv2.imshow("Multi-Cam Viewer", canvas)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                for name in ["front", "right"]:
                    if f"{name}_color" in latest:
                        base = os.path.join("saved", name)
                        os.makedirs(os.path.join(base, "rgb"), exist_ok=True)
                        os.makedirs(os.path.join(base, "depth"), exist_ok=True)
                        os.makedirs(os.path.join(base, "pointcloud"), exist_ok=True)

                        cv2.imwrite(os.path.join(base, "rgb", f"{timestamp}.png"), latest[f"{name}_color"])
                        np.save(os.path.join(base, "depth", f"{timestamp}.npy"), latest[f"{name}_depth"])
                        np.save(os.path.join(base, "pointcloud", f"{timestamp}.npy"), latest[f"{name}_pointcloud"])
                        print(f"[Saved] {name} → rgb/depth/pointcloud")

    finally:
        cam_system.finalize()
        cv2.destroyAllWindows()
