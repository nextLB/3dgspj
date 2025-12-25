"""
    自主实现的高斯溅射模型的程序文件
"""

import os
import logging
from dataset import MipNeRF360Dataset

BASE_OUTPUT_DIR = './output'

def setupTrainLogging():
    logDir = os.path.join(BASE_OUTPUT_DIR, 'log')
    if not os.path.exists(logDir):
        os.makedirs(logDir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logDir, 'train.log'), mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging




def main():
    print('version V1.1')

    # 创建日志类
    trainLogger = setupTrainLogging()

    # 创建数据集类
    MipNeRF360DatasetInstance = MipNeRF360Dataset()
    trainLogger.info('数据集类创建成功')



if __name__ == '__main__':
    main()


