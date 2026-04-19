#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from cameras import Camera
import numpy as np
from graphics_utils import fov2focal
from PIL import Image
import cv2
from threading import Lock
from collections import OrderedDict


# =============================================================================
# 图像动态缓冲区 - 按需加载与存放
# 用于小GPU的按需加载，避免一次性加载所有图像导致内存溢出
# 原理：使用LRU缓存策略，只保留最近使用的N张图像在内存中
# =============================================================================
class ImageCache:
    """
    图像动态缓冲区 - 按需加载与存放
    优点：减少GPU内存占用，提高大数据集处理效率
    """
    
    def __init__(self, max_size=10):
        """
        初始化图像缓存
        
        Args:
            max_size: 缓存池最大数量，默认10张图像
        """
        self.max_size = max_size
        # 使用OrderedDict实现LRU缓存
        self._cache = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, image_path):
        """
        获取图像，如果不在缓存中则加载
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            PIL.Image: 加载的图像对象
        """
        with self._lock:
            if image_path in self._cache:
                # 缓存命中，移动到末尾（最新使用）
                self._hits += 1
                self._cache.move_to_end(image_path)
                return self._cache[image_path]
            else:
                # 缓存未命中，加载图像
                self._misses += 1
                image = Image.open(image_path).convert('RGB')
                
                # 如果缓存已满，删除最旧的条目
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                
                self._cache[image_path] = image
                return image
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self):
        """获取缓存统计信息"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'current_size': len(self._cache),
            'max_size': self.max_size
        }


# 全局图像缓存实例 - 训练时共享使用
# 可以通过环境变量或参数调整缓存大小
import os
_default_cache_size = int(os.environ.get('IMAGE_CACHE_SIZE', '10'))
_image_cache = None


def get_image_cache(max_size=None):
    """
    获取全局图像缓存实例（单例模式）
    
    Args:
        max_size: 缓存池大小，默认使用环境变量IMAGE_CACHE_SIZE或10
        
    Returns:
        ImageCache: 全局缓存实例
    """
    global _image_cache
    if _image_cache is None:
        size = max_size if max_size is not None else _default_cache_size
        _image_cache = ImageCache(max_size=size)
    return _image_cache


def clear_image_cache():
    """清空全局图像缓存"""
    global _image_cache
    if _image_cache is not None:
        _image_cache.clear()


WARNED = False


def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    # =============================================================================
    # 【关键修改】使用缓存式加载，避免一次性加载所有图像
    # 如果cam_info.image已经加载（通过延迟加载），直接使用
    # 否则使用ImageCache按需加载
    # =============================================================================
    if cam_info.image is not None:
        # 延迟加载模式：图像已经在CameraInfo中准备好
        image = cam_info.image
    else:
        # 缓存式加载：使用动态缓冲区按需加载
        # 获取全局缓存实例
        image_cache = get_image_cache()
        image = image_cache.get(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry