"""
    自主实现的高斯溅射模型的程序文件
"""

import os
import logging
import subprocess
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
    print('version V1.2')

    # 创建日志类
    trainLogger = setupTrainLogging()

    # 创建数据集类
    MipNeRF360DatasetInstance = MipNeRF360Dataset()
    trainLogger.info('数据集类创建成功')

    # 定义基础的sharp三维重构的命令
    sharpBaseCommand = ['sharp', 'predict', '-i', MipNeRF360DatasetInstance.allSortedImagePath[0], '-o', './output/reconstruction_results']

    # 执行命令
    baseCommandResult = subprocess.run(
        sharpBaseCommand,
        capture_output=True,
        text=True,
        check=True  # 如果命令返回非零退出码，抛出异常
    )
    trainLogger.info(f'命令输出: {baseCommandResult.stdout}')
    if baseCommandResult.stderr:
        print(f'命令错误: {baseCommandResult.stderr}')



if __name__ == '__main__':
    main()


