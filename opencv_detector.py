#!/usr/bin/env python3
"""
OpenCV LINE-MOD 检测节点（Python 版本）

功能：
1. 加载已生成的 LINE-MOD 模板文件
2. 对输入图像进行目标检测
3. 可视化匹配结果

依赖：
    pip install opencv-contrib-python numpy

使用方法：
    python opencv_detector.py <template.yml> <color_image> [--depth depth_image] [--threshold 80]

示例：
    python opencv_detector.py linemod_templates.yml.gz benchmark/img0.png --depth benchmark/depth0.png
    python opencv_detector.py linemod_templates.yml benchmark/img0.png -t 70
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


class LinemodDetector:
    """LINE-MOD 检测器封装类"""
    
    def __init__(self):
        self.detector = None
        self.class_ids = []
    
    def load_templates(self, template_path: str):
        """
        加载 LINE-MOD 模板文件
        
        Args:
            template_path: 模板文件路径 (.yml 或 .yml.gz)
        """
        # 检查文件是否存在
        if not Path(template_path).exists():
            # 尝试 .gz 版本
            if template_path.endswith('.yml'):
                gz_path = template_path + '.gz'
                if Path(gz_path).exists():
                    template_path = gz_path
                    print(f"Using compressed file: {gz_path}")
            else:
                raise FileNotFoundError(f"Template file not found: {template_path}")
        
        # 读取文件
        fs = cv2.FileStorage(template_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise IOError(f"Cannot open template file: {template_path}")
        
        # 创建检测器
        self.detector = cv2.linemod_Detector()
        
        # 读取检测器
        self.detector.read(fs.getFirstTopLevelNode())
        
        # 读取类别信息
        classes_node = fs.getNode("classes")
        if not classes_node.empty():
            for i in range(classes_node.size()):
                self.detector.readClass(classes_node.at(i))
        
        fs.release()
        
        # 获取类别 ID
        self.class_ids = self.detector.classIds()
        
        print(f"Loaded detector with:")
        print(f"  - {self.detector.numClasses()} classes")
        print(f"  - {self.detector.numTemplates()} templates")
        print(f"  - Class IDs: {self.class_ids}")
    
    def detect(self, color_img: np.ndarray, depth_img: np.ndarray = None, 
               threshold: float = 80.0) -> list:
        """
        执行目标检测
        
        Args:
            color_img: RGB 彩色图像
            depth_img: 深度图像（可选）
            threshold: 检测阈值
            
        Returns:
            检测结果列表
        """
        if self.detector is None:
            raise RuntimeError("Detector not loaded. Call load_templates() first.")
        
        # 准备输入
        sources = [color_img]
        if depth_img is not None:
            sources.append(depth_img)
        
        # 执行匹配
        matches = self.detector.match(sources, threshold)
        
        results = []
        for match in matches:
            result = {
                'class_id': match.class_id,
                'template_id': match.template_id,
                'similarity': match.similarity,
                'x': match.x,
                'y': match.y,
                'position': (match.x, match.y),
                'bounding_box': None
            }
            
            # 获取边界框
            try:
                templates = self.detector.getTemplates(match.class_id, match.template_id)
                if templates:
                    # 计算所有特征的包围盒
                    all_features = []
                    for templ in templates:
                        for feature in templ.features:
                            all_features.append((feature.x, feature.y))
                    
                    if all_features:
                        features = np.array(all_features)
                        min_x = features[:, 0].min()
                        min_y = features[:, 1].min()
                        max_x = features[:, 0].max()
                        max_y = features[:, 1].max()
                        
                        result['bounding_box'] = {
                            'x': match.x + min_x,
                            'y': match.y + min_y,
                            'width': max_x - min_x + 1,
                            'height': max_y - min_y + 1
                        }
            except Exception as e:
                print(f"Warning: Could not get template info: {e}")
            
            results.append(result)
        
        return results
    
    def draw_results(self, image: np.ndarray, results: list) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            results: 检测结果列表
            
        Returns:
            绘制结果后的图像
        """
        result_img = image.copy()
        
        # 定义颜色
        colors = [
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 0, 0),    # 蓝色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 紫色
        ]
        
        for idx, result in enumerate(results):
            color = colors[idx % len(colors)]
            
            # 绘制特征点
            try:
                templates = self.detector.getTemplates(result['class_id'], result['template_id'])
                for templ in templates:
                    for feature in templ.features:
                        pt = (feature.x + result['x'], feature.y + result['y'])
                        cv2.circle(result_img, pt, 2, color, -1)
            except:
                pass
            
            # 绘制边界框
            if result['bounding_box']:
                bbox = result['bounding_box']
                cv2.rectangle(result_img, 
                             (int(bbox['x']), int(bbox['y'])),
                             (int(bbox['x'] + bbox['width']), int(bbox['y'] + bbox['height'])),
                             color, 2)
            
            # 绘制匹配位置
            cv2.circle(result_img, (result['x'], result['y']), 5, (255, 255, 255), -1)
            cv2.circle(result_img, (result['x'], result['y']), 3, color, -1)
            
            # 绘制标签
            label = f"{result['class_id']} (T{result['template_id']}, S{int(result['similarity'])})"
            
            if result['bounding_box']:
                text_org = (int(result['bounding_box']['x']), int(result['bounding_box']['y']) - 5)
            else:
                text_org = (result['x'] + 10, result['y'] - 10)
            
            # 绘制背景
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_img,
                         (text_org[0] - 2, text_org[1] - text_h - 2),
                         (text_org[0] + text_w + 2, text_org[1] + 2),
                         (0, 0, 0), -1)
            
            cv2.putText(result_img, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_img


def main():
    parser = argparse.ArgumentParser(description='OpenCV LINE-MOD Detector')
    parser.add_argument('template', help='Template file path (.yml or .yml.gz)')
    parser.add_argument('color', help='Color image path')
    parser.add_argument('--depth', '-d', help='Depth image path (optional)', default=None)
    parser.add_argument('--threshold', '-t', type=float, default=80.0, 
                        help='Detection threshold (default: 80.0)')
    parser.add_argument('--output', '-o', help='Output image path', default='detection_result.png')
    parser.add_argument('--no-display', action='store_true', help='Do not display result window')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("OpenCV LINE-MOD Detector (Python)")
    print("=" * 50)
    print(f"Template file: {args.template}")
    print(f"Color image: {args.color}")
    if args.depth:
        print(f"Depth image: {args.depth}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # 创建检测器
    detector = LinemodDetector()
    
    # 加载模板
    print("Loading templates...")
    detector.load_templates(args.template)
    print()
    
    # 加载图像
    print("Loading images...")
    color_img = cv2.imread(args.color, cv2.IMREAD_COLOR)
    if color_img is None:
        print(f"Error: Cannot load color image: {args.color}")
        return -1
    
    depth_img = None
    if args.depth:
        depth_img = cv2.imread(args.depth, cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            print(f"Warning: Cannot load depth image: {args.depth}")
    
    print(f"Color image size: {color_img.shape[1]}x{color_img.shape[0]}")
    if depth_img is not None:
        print(f"Depth image size: {depth_img.shape[1]}x{depth_img.shape[0]}")
    print()
    
    # 执行检测
    print("Running detection...")
    import time
    start_time = time.time()
    
    results = detector.detect(color_img, depth_img, args.threshold)
    
    elapsed_time = (time.time() - start_time) * 1000
    print(f"Detection time: {elapsed_time:.2f} ms")
    print()
    
    # 打印结果
    if results:
        print("Detection Results:")
        print("-" * 50)
        for i, r in enumerate(results):
            print(f"[{i}] Class: {r['class_id']}, Template: {r['template_id']}, "
                  f"Similarity: {r['similarity']:.1f}, "
                  f"Position: ({r['x']}, {r['y']})")
            if r['bounding_box']:
                print(f"    BBox: {r['bounding_box']['width']}x{r['bounding_box']['height']}")
        print()
    else:
        print("No matches found!")
    
    # 可视化
    result_img = detector.draw_results(color_img, results)
    
    # 保存结果
    cv2.imwrite(args.output, result_img)
    print(f"Result saved to: {args.output}")
    
    # 显示结果
    if not args.no_display:
        cv2.namedWindow("LINE-MOD Detection Result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("LINE-MOD Detection Result", result_img)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
