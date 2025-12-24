from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
import os
from PIL import Image
import tempfile


class ImageUploadTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.upload_url = reverse('image_import_module:upload_image')

    def create_test_image(self):
        """创建一个测试图像文件"""
        image = Image.new('RGB', (100, 100), color='red')
        tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image.save(tmp_file.name)
        tmp_file.seek(0)
        return tmp_file

    def test_upload_page_access(self):
        """测试上传页面是否可以访问"""
        response = self.client.get(self.upload_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '图像数据导入系统')

    def test_image_upload(self):
        """测试图像上传功能"""
        # 创建测试图像
        with self.create_test_image() as img_file:
            response = self.client.post(self.upload_url, {
                'image': img_file
            })

            if response.status_code == 200:
                # 检查是否上传成功
                self.assertContains(response, '图像上传成功！')
            else:
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.content.decode()}")

    def test_invalid_file_type(self):
        """测试上传无效文件类型"""
        invalid_file = SimpleUploadedFile(
            "test.txt",
            b"not an image content",
            content_type="text/plain"
        )

        response = self.client.post(self.upload_url, {
            'image': invalid_file
        })

        self.assertContains(response, '只支持以下图像格式')

    def tearDown(self):
        # 清理测试文件
        test_dir = 'media/uploaded_images/test/'
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, f))
            os.rmdir(test_dir)