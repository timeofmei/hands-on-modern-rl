"""
第11章：几何图形计数数据集生成器
==========================================================

本脚本为 VLM（视觉语言模型）强化学习实验生成几何图形计数数据集：
  1. 在空白画布上随机绘制三角形、圆形、正方形
  2. 每种形状的数量随机（0~5 个），生成 ground truth 标签
  3. 保存图片至 geometry_dataset/ 目录
  4. 生成 JSON 元数据文件（图片路径、提示词、标注信息）
  5. 划分 50 张训练集 + 10 张测试集
  6. 展示 4 张样本图片
  7. 打印数据集统计信息

用途：
  - 为 VLM GRPO 训练提供视觉推理数据
  - 测试模型对图像中物体计数的能力
  - 研究多模态奖励函数的设计

运行方式：
  pip install -r requirements.txt
  python geometry_counting_dataset.py
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体，确保图表标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：形状绘制函数
# ==========================================
# 定义三种几何图形的绘制方法：三角形、圆形、正方形
# 每种形状有随机颜色、随机位置和随机大小

# 预定义一组鲜艳的颜色，便于在图片中区分不同形状
SHAPE_COLORS = [
    '#e74c3c',  # 红色
    '#3498db',  # 蓝色
    '#2ecc71',  # 绿色
    '#f39c12',  # 橙色
    '#9b59b6',  # 紫色
    '#1abc9c',  # 青色
    '#e67e22',  # 深橙色
    '#2980b9',  # 深蓝色
    '#27ae60',  # 深绿色
    '#c0392b',  # 深红色
    '#8e44ad',  # 深紫色
    '#d35400',  # 棕橙色
]


def draw_triangle(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    在画布上绘制一个等边三角形

    参数：
        draw: PIL ImageDraw 对象
        cx, cy: 三角形中心的坐标
        size: 三角形的大小（外接圆半径）
        color: 填充颜色
        outline_color: 轮廓颜色
    """
    # 计算等边三角形的三个顶点
    # 顶点朝上，底边水平
    points = [
        (cx, cy - size),                          # 顶部顶点
        (cx - size * 0.866, cy + size * 0.5),     # 左下顶点
        (cx + size * 0.866, cy + size * 0.5),     # 右下顶点
    ]
    draw.polygon(points, fill=color, outline=outline_color, width=2)


def draw_circle(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    在画布上绘制一个圆形

    参数：
        draw: PIL ImageDraw 对象
        cx, cy: 圆心的坐标
        size: 半径
        color: 填充颜色
        outline_color: 轮廓颜色
    """
    # PIL 的 ellipse 需要左上角和右下角的坐标
    bbox = [cx - size, cy - size, cx + size, cy + size]
    draw.ellipse(bbox, fill=color, outline=outline_color, width=2)


def draw_square(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    在画布上绘制一个正方形

    参数：
        draw: PIL ImageDraw 对象
        cx, cy: 正方形中心的坐标
        size: 边长的一半
        color: 填充颜色
        outline_color: 轮廓颜色
    """
    bbox = [cx - size, cy - size, cx + size, cy + size]
    draw.rectangle(bbox, fill=color, outline=outline_color, width=2)


# 将形状名称映射到绘制函数
SHAPE_DRAWERS = {
    '三角形': draw_triangle,
    '圆形': draw_circle,
    '正方形': draw_square,
}


# ==========================================
# 第二部分：单张图片生成
# ==========================================
def generate_single_image(img_width=256, img_height=256, seed=None):
    """
    生成一张包含随机几何图形的图片

    流程：
      1. 创建白色背景画布
      2. 对每种形状（三角形、圆形、正方形），随机生成 0~5 个
      3. 每个形状的位置、大小、颜色均随机
      4. 避免形状重叠过多（简单碰撞检测）

    参数：
        img_width: 图片宽度（像素）
        img_height: 图片高度（像素）
        seed: 随机种子（可选，用于可复现性）

    返回：
        image: PIL Image 对象
        ground_truth: dict，包含每种形状的数量
                      例: {'三角形': 3, '圆形': 1, '正方形': 2}
    """
    if seed is not None:
        random.seed(seed)

    # 创建白色背景
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)

    # 记录每种形状的数量
    ground_truth = {'三角形': 0, '圆形': 0, '正方形': 0}

    # 已绘制的形状占用的区域（用于简单碰撞检测）
    occupied_regions = []

    for shape_name in ['三角形', '圆形', '正方形']:
        # 随机决定该形状的数量（0~5）
        count = random.randint(0, 5)
        ground_truth[shape_name] = count

        for _ in range(count):
            # 随机大小：15~30 像素
            size = random.randint(15, 30)

            # 随机颜色
            color = random.choice(SHAPE_COLORS)

            # 尝试找到一个不与已有形状重叠的位置
            # 最多尝试 50 次，避免死循环
            placed = False
            for _attempt in range(50):
                cx = random.randint(size + 10, img_width - size - 10)
                cy = random.randint(size + 10, img_height - size - 10)

                # 简单碰撞检测：检查新形状中心与已有形状中心的距离
                overlap = False
                for (ox, oy, osize) in occupied_regions:
                    dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                    if dist < (size + osize) * 0.8:
                        overlap = True
                        break

                if not overlap:
                    # 绘制形状
                    drawer = SHAPE_DRAWERS[shape_name]
                    drawer(draw, cx, cy, size, color)
                    occupied_regions.append((cx, cy, size))
                    placed = True
                    break

            # 如果多次尝试后仍无法放置，强制放置在随机位置
            if not placed:
                cx = random.randint(size + 10, img_width - size - 10)
                cy = random.randint(size + 10, img_height - size - 10)
                drawer = SHAPE_DRAWERS[shape_name]
                drawer(draw, cx, cy, size, color)
                occupied_regions.append((cx, cy, size))

    return image, ground_truth


# ==========================================
# 第三部分：数据集生成
# ==========================================
def generate_dataset(output_dir='output/geometry_dataset',
                     num_train=50, num_test=10,
                     img_width=256, img_height=256,
                     base_seed=42):
    """
    生成完整的几何图形计数数据集

    数据集结构：
        geometry_dataset/
        ├── train/           # 训练集图片
        │   ├── train_0001.png
        │   ├── train_0002.png
        │   └── ...
        ├── test/            # 测试集图片
        │   ├── test_0001.png
        │   └── ...
        └── metadata.json    # 元数据文件

    元数据 JSON 格式：
        {
            "train": [
                {
                    "image_path": "train/train_0001.png",
                    "prompt": "请数一下图片中有多少个三角形、圆形和正方形",
                    "ground_truth": {"三角形": 3, "圆形": 1, "正方形": 2},
                    "total_shapes": 6
                },
                ...
            ],
            "test": [...]
        }

    参数：
        output_dir: 输出目录
        num_train: 训练集数量
        num_test: 测试集数量
        img_width: 图片宽度
        img_height: 图片高度
        base_seed: 基础随机种子

    返回：
        metadata: dict，完整的数据集元数据
    """
    print("=" * 60)
    print("  几何图形计数数据集生成器")
    print("=" * 60)

    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    metadata = {'train': [], 'test': []}

    # 标准提示词：要求模型数出每种形状的数量
    prompt = "请数一下图片中有多少个三角形、圆形和正方形"

    # ---------- 生成训练集 ----------
    print(f"\n正在生成训练集（{num_train} 张）...")
    for i in range(num_train):
        seed = base_seed + i
        image, gt = generate_single_image(img_width, img_height, seed=seed)

        filename = f"train_{i+1:04d}.png"
        filepath = os.path.join(train_dir, filename)
        image.save(filepath)

        metadata['train'].append({
            'image_path': f"train/{filename}",
            'prompt': prompt,
            'ground_truth': gt,
            'total_shapes': sum(gt.values()),
        })

        # 进度显示
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  已生成 {i+1}/{num_train} 张训练图片")

    # ---------- 生成测试集 ----------
    print(f"\n正在生成测试集（{num_test} 张）...")
    for i in range(num_test):
        seed = base_seed + 1000 + i  # 使用不同的种子范围，避免与训练集重复
        image, gt = generate_single_image(img_width, img_height, seed=seed)

        filename = f"test_{i+1:04d}.png"
        filepath = os.path.join(test_dir, filename)
        image.save(filepath)

        metadata['test'].append({
            'image_path': f"test/{filename}",
            'prompt': prompt,
            'ground_truth': gt,
            'total_shapes': sum(gt.values()),
        })

    # 保存元数据 JSON
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n元数据已保存至: {metadata_path}")
    print(f"训练集: {len(metadata['train'])} 张")
    print(f"测试集: {len(metadata['test'])} 张")

    return metadata


# ==========================================
# 第四部分：数据集统计信息
# ==========================================
def print_dataset_statistics(metadata):
    """
    打印数据集的统计信息

    统计内容：
      1. 总样本数
      2. 每种形状在数据集中出现的频率和数量分布
      3. 总形状数的分布（最小、最大、平均）
      4. 各形状组合的覆盖情况
    """
    print("\n" + "=" * 60)
    print("  数据集统计信息")
    print("=" * 60)

    all_data = metadata['train'] + metadata['test']
    total_samples = len(all_data)

    print(f"\n总样本数: {total_samples}")
    print(f"  训练集: {len(metadata['train'])} 张")
    print(f"  测试集: {len(metadata['test'])} 张")

    # 统计每种形状的数量分布
    print("\n各形状数量分布：")
    print(f"  {'形状':>6s}  {'最少':>4s}  {'最多':>4s}  {'平均':>6s}  {'出现频率':>8s}")
    print(f"  {'------':>6s}  {'----':>4s}  {'----':>4s}  {'------':>6s}  {'--------':>8s}")

    total_shapes_list = []
    for shape_name in ['三角形', '圆形', '正方形']:
        counts = [item['ground_truth'][shape_name] for item in all_data]
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        # 出现频率 = 至少有 1 个该形状的样本占比
        freq = sum(1 for c in counts if c > 0) / len(counts) * 100
        print(f"  {shape_name:>6s}  {min_count:>4d}  {max_count:>4d}  "
              f"{avg_count:>6.2f}  {freq:>7.1f}%")

    # 总形状数分布
    total_counts = [item['total_shapes'] for item in all_data]
    total_shapes_list.extend(total_counts)
    print(f"\n总形状数统计：")
    print(f"  最少: {min(total_counts)} 个")
    print(f"  最多: {max(total_counts)} 个")
    print(f"  平均: {sum(total_counts) / len(total_counts):.1f} 个")

    # 总形状数分布直方图
    print(f"\n总形状数分布：")
    count_dist = {}
    for t in total_counts:
        count_dist[t] = count_dist.get(t, 0) + 1
    for k in sorted(count_dist.keys()):
        bar = '|' * (count_dist[k] * 2)
        print(f"  {k:>2d} 个: {bar} ({count_dist[k]} 张)")


# ==========================================
# 第五部分：展示样本图片
# ==========================================
def display_sample_images(metadata, output_dir='output/geometry_dataset', num_samples=4):
    """
    展示数据集中的样本图片及其标注

    参数：
        metadata: 数据集元数据
        output_dir: 图片所在目录
        num_samples: 展示的样本数量
    """
    print("\n" + "=" * 60)
    print(f"  展示 {num_samples} 张样本图片")
    print("=" * 60)

    # 从训练集中均匀采样
    train_data = metadata['train']
    indices = np.linspace(0, len(train_data) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("几何图形计数数据集 — 样本展示", fontsize=16, fontweight='bold')

    for ax_idx, idx in enumerate(indices):
        row = ax_idx // 2
        col = ax_idx % 2
        ax = axes[row, col]

        item = train_data[idx]
        img_path = os.path.join(output_dir, item['image_path'])
        image = Image.open(img_path)

        ax.imshow(image)
        ax.axis('off')

        # 标题中显示标注信息
        gt = item['ground_truth']
        title = (f"样本 #{idx+1}\n"
                 f"三角形: {gt['三角形']}  圆形: {gt['圆形']}  正方形: {gt['正方形']}\n"
                 f"总计: {item['total_shapes']} 个形状")
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig('output/geometry_dataset_samples.png', dpi=150, bbox_inches='tight')
    print("  样本图片已保存为 output/geometry_dataset_samples.png")
    plt.show()


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 第一步：生成数据集
    metadata = generate_dataset(
        output_dir='output/geometry_dataset',
        num_train=50,
        num_test=10,
        img_width=256,
        img_height=256,
        base_seed=42,
    )

    # 第二步：打印统计信息
    print_dataset_statistics(metadata)

    # 第三步：展示样本图片
    display_sample_images(metadata, output_dir='output/geometry_dataset', num_samples=4)

    # 最终总结
    print("\n" + "=" * 60)
    print("  数据集生成完成")
    print("=" * 60)
    print("""
  数据集用途：
    1. 为 VLM（视觉语言模型）提供视觉推理训练数据
    2. 模型需要理解图像内容并正确计数每种几何图形
    3. 可配合 GRPO 训练，用计数准确度作为奖励信号
    4. 支持研究多模态奖励函数的设计（准确度 + 推理质量 + 格式）

  数据集特点：
    - 每张图片包含 0~5 个三角形、0~5 个圆形、0~5 个正方形
    - 总计 0~15 个形状，覆盖不同难度
    - 形状颜色随机，位置随机，带有简单碰撞检测
    - 标准提示词："请数一下图片中有多少个三角形、圆形和正方形"

  后续步骤：
    - 运行 multi_modal_reward.py：设计多模态奖励函数
    - 运行 vlm_grpo_train.py：进行 VLM GRPO 训练
    """)
