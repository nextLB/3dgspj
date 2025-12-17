"""4.3 系统集成与测试"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile


class TestGaussianSplatting(unittest.TestCase):
    """单元测试"""

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_model_initialization(self):
        """测试模型初始化"""
        from gaussian_splatting import GaussianSplattingModel

        model = GaussianSplattingModel(max_gaussians=1000, sh_degree=2).to(self.device)

        # 检查参数形状
        self.assertEqual(model.xyz.shape, (1000, 3))
        self.assertEqual(model.rotation.shape, (1000, 4))
        self.assertEqual(model.scale.shape, (1000, 3))
        self.assertEqual(model.opacity.shape, (1000, 1))

        # 检查参数范围
        self.assertTrue(torch.all(model.opacity <= 1))
        self.assertTrue(torch.all(model.opacity >= 0))

    def test_loss_computation(self):
        """测试损失计算"""
        from losses import PhotometricLoss

        loss_fn = PhotometricLoss()

        # 创建测试数据
        pred = torch.rand(2, 3, 256, 256)
        target = torch.rand(2, 3, 256, 256)

        loss = loss_fn(pred, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_data_loading(self):
        """测试数据加载"""
        from dataset import SceneDataset

        # 创建临时测试数据
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 创建模拟数据
            self._create_mock_data(tmpdir)

            # 测试数据集加载
            dataset = SceneDataset(tmpdir)
            self.assertGreater(len(dataset), 0)

            # 测试数据项
            item = dataset[0]
            self.assertIn('image', item)
            self.assertIn('camera', item)

    def _create_mock_data(self, dir_path):
        """创建模拟数据"""
        import json

        # 创建目录
        (dir_path / "images").mkdir(exist_ok=True)

        # 创建相机数据
        cameras = []
        for i in range(5):
            camera = {
                "id": i,
                "img_name": f"images/{i:04d}.png",
                "width": 800,
                "height": 600,
                "fx": 800.0,
                "fy": 800.0,
                "cx": 400.0,
                "cy": 300.0,
                "position": [i * 0.5, 0, 0],
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            }
            cameras.append(camera)

            # 创建模拟图像
            img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(dir_path / f"images/{i:04d}.png")

        # 保存相机数据
        with open(dir_path / "cameras.json", "w") as f:
            json.dump(cameras, f, indent=2)

        # 创建模拟点云
        from plyfile import PlyData, PlyElement
        vertices = np.random.randn(100, 3).astype(np.float32)
        vertex = np.array([tuple(v) for v in vertices],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(str(dir_path / "pointcloud.ply"))


class IntegrationTest(unittest.TestCase):
    """集成测试"""

    def test_training_pipeline(self):
        """测试训练流程"""
        # 这里可以添加端到端的测试
        pass


if __name__ == '__main__':
    unittest.main()