#!/usr/bin/env python3
"""
合并高斯泼溅PLY文件（简化版，只合并顶点数据）
依赖库: numpy
"""

import os
import sys
import numpy as np
import struct
from pathlib import Path
from tqdm import tqdm


def read_gaussian_ply_header(filepath):
    """读取高斯泼溅PLY文件的头部信息"""
    with open(filepath, 'rb') as f:
        lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            lines.append(line)
            if line == 'end_header':
                break

        # 查找顶点数量
        vertex_count = 0
        for line in lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                break

        return vertex_count, f.tell()  # 返回顶点数量和头部结束位置


def read_gaussian_vertices(filepath, vertex_count, header_end):
    """读取高斯泼溅顶点数据"""
    # 每个顶点有14个float32属性
    vertex_format = '<' + 'f' * 14  # 14个float32: x, y, z, f_dc0-2, opacity, scale0-2, rot0-3
    vertex_size = struct.calcsize(vertex_format)

    with open(filepath, 'rb') as f:
        f.seek(header_end)  # 跳过头部

        vertices = []
        try:
            # 尝试读取所有顶点
            for _ in range(vertex_count):
                data = f.read(vertex_size)
                if len(data) < vertex_size:
                    break
                vertex = struct.unpack(vertex_format, data)
                vertices.append(vertex)
        except Exception as e:
            print(f"读取顶点数据时出错: {e}")

        return np.array(vertices, dtype=np.float32)


def remove_duplicates_gaussians(vertices, tolerance=1e-5):
    """去除重复的高斯点"""
    if len(vertices) == 0:
        return vertices

    # 提取位置（xyz）进行去重
    positions = vertices[:, :3]

    # 四舍五入到指定精度
    rounded_positions = np.round(positions / tolerance) * tolerance

    # 找到唯一的位置
    _, unique_indices = np.unique(rounded_positions, axis=0, return_index=True)

    # 返回去重后的所有属性
    return vertices[unique_indices]


def merge_gaussian_ply_simple(input_folder, output_file=None, remove_duplicates=True,
                              max_files=None, progress=True):
    """
    简单合并高斯泼溅PLY文件（只合并顶点数据）

    参数:
        input_folder: 输入文件夹路径
        output_file: 输出文件路径
        remove_duplicates: 是否去除重复点
        max_files: 最大合并文件数（用于测试）
        progress: 是否显示进度条
    """

    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"错误: 文件夹不存在: {input_folder}")
        return False

    # 查找PLY文件
    ply_files = list(input_path.glob("*.ply"))
    if not ply_files:
        print(f"错误: 未找到PLY文件: {input_folder}")
        return False

    if max_files:
        ply_files = ply_files[:max_files]

    print(f"找到 {len(ply_files)} 个PLY文件")

    # 设置输出路径
    if output_file is None:
        output_file = input_path.parent / f"{input_path.name}_merged_simple.ply"
    else:
        output_file = Path(output_file)

    # 收集所有顶点
    all_vertices = []
    total_vertices = 0

    # 读取所有文件
    file_iter = tqdm(ply_files, desc="读取文件") if progress else ply_files

    for ply_file in file_iter:
        try:
            # 读取头部信息
            vertex_count, header_end = read_gaussian_ply_header(ply_file)

            # 读取顶点数据
            vertices = read_gaussian_vertices(ply_file, vertex_count, header_end)

            if len(vertices) > 0:
                all_vertices.append(vertices)
                total_vertices += len(vertices)

                if progress:
                    file_iter.set_postfix({"当前文件顶点": len(vertices), "累计顶点": total_vertices})

        except Exception as e:
            print(f"处理文件 {ply_file.name} 时出错: {e}")
            continue

    if not all_vertices:
        print("错误: 未读取到任何顶点数据")
        return False

    # 合并顶点
    print(f"\n合并顶点数据...")
    merged_vertices = np.vstack(all_vertices)
    print(f"合并后总顶点数: {len(merged_vertices):,}")

    # 去重
    if remove_duplicates:
        print(f"去除重复顶点...")
        original_count = len(merged_vertices)
        merged_vertices = remove_duplicates_gaussians(merged_vertices)
        removed = original_count - len(merged_vertices)
        print(f"移除了 {removed:,} 个重复点，剩余 {len(merged_vertices):,} 个点")

    # 写入合并后的文件
    print(f"\n写入合并文件: {output_file}")

    try:
        with open(output_file, 'wb') as f:
            # 写入头部
            header = f"""ply
format binary_little_endian 1.0
element vertex {len(merged_vertices)}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
            f.write(header.encode('utf-8'))

            # 写入顶点数据
            vertex_format = '<' + 'f' * 14

            for vertex in merged_vertices:
                vertex_bytes = struct.pack(vertex_format, *vertex)
                f.write(vertex_bytes)

            print(f"文件已保存: {output_file}")
            print(f"文件大小: {output_file.stat().st_size / (1024 * 1024):.2f} MB")

        return True

    except Exception as e:
        print(f"写入文件时出错: {e}")
        return False


def test_single_file(filepath):
    """测试单个PLY文件"""
    print(f"\n测试文件: {filepath}")

    try:
        # 读取头部
        with open(filepath, 'rb') as f:
            lines = []
            while True:
                line = f.readline().decode('utf-8').strip()
                lines.append(line)
                print(f"  {line}")
                if line == 'end_header':
                    break

            # 显示头部后的位置
            header_end = f.tell()
            print(f"头部结束位置: {header_end}")

            # 读取一些顶点数据测试
            vertex_format = '<' + 'f' * 14
            vertex_size = struct.calcsize(vertex_format)

            print(f"\n尝试读取前3个顶点...")
            for i in range(3):
                data = f.read(vertex_size)
                if len(data) == vertex_size:
                    vertex = struct.unpack(vertex_format, data)
                    print(f"  顶点{i + 1}: {vertex[:3]}...")  # 只显示xyz
                else:
                    print(f"  无法读取顶点{i + 1}，数据不足")
                    break

            # 尝试读取其他数据
            print(f"\n头部后剩余数据大小: {os.path.getsize(filepath) - header_end:,} 字节")

    except Exception as e:
        print(f"测试时出错: {e}")


def main():
    """主函数"""

    if len(sys.argv) < 2:
        print("高斯泼溅PLY文件合并工具（简化版）")
        print("=" * 60)
        print("使用方法: python merge_gaussian_simple.py <文件夹路径> [输出文件路径]")
        print("\n示例:")
        print("  python merge_gaussian_simple.py ./output/reconstruction_results")
        print("  python merge_gaussian_simple.py ./scenes ./merged.ply")
        print("\n选项:")
        print("  文件夹路径: 包含高斯泼溅PLY文件的文件夹")
        print("  输出文件路径: (可选) 合并后的输出文件路径")
        print("\n测试单个文件:")
        print("  python merge_gaussian_simple.py --test <文件路径>")
        return

    # 检查是否是测试模式
    if sys.argv[1] == '--test' and len(sys.argv) > 2:
        test_single_file(sys.argv[2])
        return

    input_folder = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"输入文件夹: {input_folder}")
    print(f"输出文件: {output_file or '自动生成'}")
    print("=" * 60)

    # 第一次运行：测试模式，只处理前max fiels个文件
    # print("\n🎯 测试模式：先处理前3个文件...")
    test_success = merge_gaussian_ply_simple(
        input_folder=input_folder,
        output_file=output_file.parent / "test_merged.ply" if output_file else None,
        remove_duplicates=True,
        max_files=10,
        progress=True
    )

    if not test_success:
        print("\n❌ 测试失败，请检查文件格式")
        return

    # 确认是否继续
    print("\n" + "=" * 60)
    response = input("测试成功！是否继续合并所有文件？(y/n): ")

    if response.lower() != 'y':
        print("已取消")
        return

    # 完整合并
    print("\n🚀 开始完整合并...")
    success = merge_gaussian_ply_simple(
        input_folder=input_folder,
        output_file=output_file,
        remove_duplicates=True,
        max_files=None,
        progress=True
    )

    if success:
        print("\n✅ 合并成功!")
        print(f"输出文件: {output_file}")
    else:
        print("\n❌ 合并失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()


