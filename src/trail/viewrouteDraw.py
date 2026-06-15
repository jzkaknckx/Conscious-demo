"""
debug_mark.py - 图像点线标记增强版

支持 torch.Tensor 输入，支持线条渐变色彩，支持独立控制点和线显示，
支持生成纯点和线图（点集放大至占满画幅）及融合图。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _to_numpy(data):
    """将 torch.Tensor 或 numpy 数组转换为 numpy 数组"""
    if HAS_TORCH and torch.is_tensor(data):
        if data.is_cuda:
            data = data.cpu()
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


def parse_points(points_input: Union[str, List[int], List[Tuple[int, int]], 'torch.Tensor']) -> List[Tuple[int, int]]:
    """解析多种格式的点坐标输入"""
    if HAS_TORCH and torch.is_tensor(points_input):
        arr = _to_numpy(points_input)
        if arr.ndim == 1:
            if len(arr) % 2 != 0:
                raise ValueError("点坐标数量必须是偶数（x,y成对出现）")
            return [(int(arr[i]), int(arr[i+1])) for i in range(0, len(arr), 2)]
        elif arr.ndim == 2 and arr.shape[1] == 2:
            return [(int(x), int(y)) for x, y in arr]
        else:
            raise ValueError("torch.Tensor 形状应为 (N,) 或 (N,2)")

    if isinstance(points_input, str):
        parts = points_input.split(',')
        if len(parts) % 2 != 0:
            raise ValueError("点坐标数量必须是偶数（x,y成对出现）")
        coords = [int(p.strip()) for p in parts]
        return [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

    if isinstance(points_input, list):
        if len(points_input) == 0:
            return []
        if all(isinstance(p, tuple) and len(p) == 2 for p in points_input):
            return points_input
        if all(isinstance(p, int) for p in points_input):
            if len(points_input) % 2 != 0:
                raise ValueError("点坐标数量必须是偶数（x,y成对出现）")
            return [(points_input[i], points_input[i+1]) for i in range(0, len(points_input), 2)]
        raise ValueError("无法识别的点坐标格式")

    raise ValueError(f"不支持的点坐标类型: {type(points_input)}")


def _hue_to_bgr(hue: float) -> Tuple[int, int, int]:
    """
    将色相（0~360）转换为 BGR 颜色。
    饱和度=1，明度=1，返回0-255范围的BGR元组。
    """
    # 归一化到0~1
    h = hue / 360.0
    # 使用 HSV 转换，方便且准确
    hsv = np.array([[[h * 179, 255, 255]]], dtype=np.uint8)  # OpenCV HSV: H 0-179, S 0-255, V 0-255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _compute_path_lengths(points: List[Tuple[int, int]]) -> List[float]:
    """计算路径累计长度（欧氏距离），返回每个点处的累计长度（起点为0）"""
    cum_lengths = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        cum_lengths.append(cum_lengths[-1] + np.hypot(dx, dy))
    return cum_lengths


def _draw_on_canvas(
    canvas: np.ndarray,
    points: List[Tuple[int, int]],
    point_size: int,
    line_thickness: int,
    draw_points: bool,
    draw_lines: bool,
    point_color: Tuple[int, int, int],
    line_color: Optional[Tuple[int, int, int]],
    gradient_lines: bool,
    start_hue: float,
    path_cum_lengths: Optional[List[float]] = None
) -> np.ndarray:
    """
    在 canvas 上绘制点和线。
    canvas: 目标图像（会被修改）
    path_cum_lengths: 若渐变线条，需提供每个点的累计长度
    """
    img = canvas  # 直接修改
    total_len = path_cum_lengths[-1] if path_cum_lengths else 1.0

    # 绘制点
    if draw_points:
        for x, y in points:
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), point_size, point_color, -1)

    # 绘制线条
    if draw_lines and len(points) > 1:
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i+1]
            if not (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                    0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                continue

            if gradient_lines and path_cum_lengths:
                # 线段颜色由其起点处的比例决定（也可用中点，效果类似）
                ratio = path_cum_lengths[i] / total_len
                hue = (start_hue + ratio * 360) % 360
                color = _hue_to_bgr(hue)
            else:
                color = line_color if line_color is not None else (0, 255, 0)  # 默认绿色

            cv2.line(img, pt1, pt2, color, line_thickness)

    return img


def _transform_points_to_fit(
    points: List[Tuple[int, int]],
    target_width: int,
    target_height: int,
    margin_ratio: float = 0.05
) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
    """
    将点集坐标映射到目标画布，使点集占据画布的大部分（留边距）。
    返回新点坐标列表和变换参数 (offset_x, offset_y, scale)。
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y

    # 目标区域（考虑边距）
    margin_x = target_width * margin_ratio
    margin_y = target_height * margin_ratio
    target_w = target_width - 2 * margin_x
    target_h = target_height - 2 * margin_y

    if width == 0 or height == 0:
        # 所有点重合，直接居中
        scale = 1.0
        offset_x = target_width // 2 - min_x
        offset_y = target_height // 2 - min_y
    else:
        scale = min(target_w / width, target_h / height)
        offset_x = margin_x + (target_w - scale * width) / 2 - min_x * scale
        offset_y = margin_y + (target_h - scale * height) / 2 - min_y * scale

    new_points = []
    for x, y in points:
        nx = int(round(x * scale + offset_x))
        ny = int(round(y * scale + offset_y))
        # 裁剪到画布内
        nx = max(0, min(target_width - 1, nx))
        ny = max(0, min(target_height - 1, ny))
        new_points.append((nx, ny))

    return new_points, (offset_x, offset_y, scale)


def debug_mark_advanced(
    image_input: Union[str, np.ndarray, 'torch.Tensor'],
    points: Union[str, List[int], List[Tuple[int, int]], 'torch.Tensor'],
    point_size: int = 5,
    line_thickness: int = 2,
    draw_points: bool = True,
    draw_lines: bool = True,
    point_color: Tuple[int, int, int] = (0, 0, 255),   # 红色
    line_color: Optional[Tuple[int, int, int]] = None,  # 若None且gradient_lines=False则使用默认绿色
    gradient_lines: bool = True,
    start_hue: float = 0.0,
    background_color: Tuple[int, int, int] = (255, 255, 255),  # 纯点和线图的背景（白色）
    output_path: Optional[str] = None,   # 保存融合图
    display_in_notebook: bool = True,
    return_images: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    高级调试标记：生成原图、纯点和线图（放大至占满画幅）、融合图。

    参数:
        image_input: 图像路径、numpy数组或torch.Tensor（支持RGB或BGR，值域0-255或0-1）
        points: 点坐标（同 parse_points）
        point_size: 点的半径（像素，整数）
        line_thickness: 线的粗细（像素，整数）
        draw_points: 是否绘制点
        draw_lines: 是否绘制线
        point_color: 点的颜色（BGR），仅在draw_points=True时有效
        line_color: 线的颜色（BGR），仅在gradient_lines=False时有效，若None则使用绿色
        gradient_lines: 是否启用线条渐变（彩虹色）
        start_hue: 渐变起始色相（0~360），线条从该色相开始旋转360度
        background_color: 纯点和线图的背景色（BGR）
        output_path: 若指定，保存融合图
        display_in_notebook: 是否在Jupyter中显示三张图（子图方式）
        return_images: 是否返回三个图像数组（原图，点和线图，融合图）

    返回:
        若return_images=True，返回元组 (original, pure_graph, blended)
    """
    # 强制整数
    point_size = int(point_size)
    line_thickness = int(line_thickness)

    # 读取原图并转换为numpy BGR格式
    if isinstance(image_input, str):
        original = cv2.imread(image_input)
        if original is None:
            raise FileNotFoundError(f"无法读取图像: {image_input}")
    elif HAS_TORCH and torch.is_tensor(image_input):
        original = _to_numpy(image_input)
        # 转置 (C,H,W) -> (H,W,C)
        if original.ndim == 4:
            original = original[0]
        if original.ndim == 3 and original.shape[0] == 3:
            original = original.transpose(1, 2, 0)
        # 值域处理
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
        else:
            original = original.astype(np.uint8)
    elif isinstance(image_input, np.ndarray):
        original = image_input.copy()
    else:
        raise ValueError(f"不支持的图像输入类型: {type(image_input)}")

    original = np.ascontiguousarray(original)

    # 解析点坐标
    point_list = parse_points(points)
    if not point_list:
        raise ValueError("点坐标列表不能为空")

    # 计算路径累计长度（用于渐变线条）
    if gradient_lines and draw_lines:
        path_lengths = _compute_path_lengths(point_list)
    else:
        path_lengths = None

    # 1. 原图
    original_img = original.copy()

    # 2. 融合图（在原图上绘制）
    blended = original.copy()
    # 确定线条颜色（非渐变时的默认值）
    if not gradient_lines and line_color is None:
        line_color = (0, 255, 0)  # 绿色
    _draw_on_canvas(
        blended, point_list, point_size, line_thickness,
        draw_points, draw_lines, point_color, line_color,
        gradient_lines, start_hue, path_lengths
    )

    # 3. 纯点和线图（点集放大至占满画幅）
    height, width = original.shape[:2]
    # 计算变换后的点坐标
    transformed_points, _ = _transform_points_to_fit(point_list, width, height)
    # 创建背景画布
    pure_graph = np.full((height, width, 3), background_color, dtype=np.uint8)
    # 对于纯点和线图，路径累计长度相同（使用原始路径比例，颜色基于原始比例）
    _draw_on_canvas(
        pure_graph, transformed_points, point_size, line_thickness,
        draw_points, draw_lines, point_color, line_color,
        gradient_lines, start_hue, path_lengths
    )

    # 显示三张图（子图）
    if display_in_notebook:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Original', 'Points & Lines (Zoomed)', 'Blended']
        images = [original_img, pure_graph, blended]

        for ax, img, title in zip(axes, images, titles):
            # 转换为RGB显示
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(title)
            ax.axis('on')
        plt.tight_layout()
        plt.show()

    # 保存融合图
    if output_path:
        cv2.imwrite(output_path, blended)
        print(f"融合图已保存至: {output_path}")

    if return_images:
        return original_img, pure_graph, blended
    return None


def mark_points_on_image(
    image_input: Union[str, np.ndarray, 'torch.Tensor'],
    points: Union[str, List[int], List[Tuple[int, int]], 'torch.Tensor'],
    point_size: int = 5,
    line_thickness: int = 2,
    point_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (0, 255, 0),
    show_labels: bool = False,
    output_path: Optional[str] = None,
    display_in_notebook: bool = True,
    return_image: bool = False
) -> Optional[np.ndarray]:
    """
    原始版本：仅在原图上绘制（无渐变，无缩放），保持向后兼容。
    """
    # 强制整数
    point_size = int(point_size)
    line_thickness = int(line_thickness)

    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_input}")
    elif HAS_TORCH and torch.is_tensor(image_input):
        img = _to_numpy(image_input)
        if original.ndim == 4:
            original = original[0]
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise ValueError(f"不支持的图像输入类型: {type(image_input)}")

    img = np.ascontiguousarray(img)
    point_list = parse_points(points)

    if not point_list:
        print("警告: 没有提供点坐标")
        if return_image:
            return img
        if display_in_notebook:
            _display_in_notebook(img)
        return None

    # 绘制点
    if show_labels:
        for i, (x, y) in enumerate(point_list):
            x, y = int(x), int(y)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), point_size, point_color, -1)
                cv2.putText(img, str(i), (x + point_size + 2, y - point_size - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        for x, y in point_list:
            x, y = int(x), int(y)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), point_size, point_color, -1)

    # 绘制线
    for i in range(len(point_list) - 1):
        pt1 = point_list[i]
        pt2 = point_list[i+1]
        if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
            0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
            cv2.line(img, pt1, pt2, line_color, line_thickness)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"图像已保存至: {output_path}")

    if display_in_notebook:
        _display_in_notebook(img)

    if return_image:
        return img
    return None


def _display_in_notebook(img: np.ndarray, figsize: Tuple[int, int] = (10, 8)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def debug_show(image, A):
    """
    调试工具：显示原图、张量A的黑白图、以及它们的叠加图。
    
    输入:
        image: torch.Tensor [1,1,W,H] 或 [1,3,W,H] 或 PIL Image 对象
        A: torch.Tensor [1,1,W,H] 或 PIL Image 对象（灰度图）
    """
    # ---- 辅助函数：将输入转为 H×W numpy 灰度图或 H×W×3 彩色图 ----
    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.squeeze(0).cpu()           # [C, W, H]
            if img.shape[0] == 1:                # 灰度
                # return img.squeeze(0).numpy().T   # (H, W)
                return img.squeeze(0).numpy()
            else:                                # 彩色
                return img.permute(1, 2, 0).numpy()  # (H, W, 3)
        elif isinstance(img, Image.Image):
            if img.mode == 'RGB':
                return np.array(img)             # (H, W, 3)
            else:  # L, LA, etc. 转为灰度
                return np.array(img.convert('L'))  # (H, W)
        else:
            raise TypeError("image must be torch.Tensor or PIL.Image")

    # 转换图像
    img_np = to_numpy(image)          # (H, W) 或 (H, W, 3)
    
    # 转换张量 A（确保是单通道灰度）
    if isinstance(A, torch.Tensor):
        A_np = A.squeeze().cpu().numpy()   # (W, H)
        A_np = A_np.T                      # (H, W)
    elif isinstance(A, Image.Image):
        A_np = np.array(A.convert('L'))    # (H, W)
    else:
        raise TypeError("A must be torch.Tensor or PIL.Image")
    
    # 归一化 A 到 [0,1]
    A_min, A_max = A_np.min(), A_np.max()
    if A_max - A_min > 1e-8:
        A_norm = (A_np - A_min) / (A_max - A_min)
    else:
        A_norm = np.zeros_like(A_np)
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 原图
    if len(img_np.shape) == 2:
        axes[0].imshow(img_np, cmap='gray')
    else:
        axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 张量 A 灰度图
    axes[1].imshow(A_norm, cmap='gray')
    axes[1].set_title("Tensor A (grayscale)")
    axes[1].axis('off')
    
    # 叠加图
    if len(img_np.shape) == 2:
        axes[2].imshow(img_np, cmap='gray')
    else:
        axes[2].imshow(img_np)
    axes[2].imshow(A_norm, cmap='hot', alpha=0.6)
    axes[2].set_title("Overlay (Image + A)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()