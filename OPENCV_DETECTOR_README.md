# OpenCV LINE-MOD Detector 使用说明

## 文件列表

| 文件 | 说明 |
|------|------|
| `opencv_detector.cpp` | C++ 版本检测器 |
| `opencv_detector.py` | Python 版本检测器 |
| `CMakeLists_detector.txt` | C++ 版本 CMake 配置 |

---

## Python 版本（推荐）

### 安装依赖

```bash
pip install opencv-contrib-python numpy
```

### 使用方法

```bash
# 基本用法
python opencv_detector.py <template.yml> <color_image>

# 带深度图
python opencv_detector.py linemod_templates.yml.gz benchmark/img0.png --depth benchmark/depth0.png

# 调整阈值
python opencv_detector.py linemod_templates.yml benchmark/img0.png -t 70

# 保存结果不显示
python opencv_detector.py linemod_templates.yml benchmark/img0.png --output result.png --no-display
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `template` | 模板文件路径 (.yml 或 .yml.gz) | 必需 |
| `color` | 彩色图像路径 | 必需 |
| `--depth, -d` | 深度图像路径 | 无 |
| `--threshold, -t` | 检测阈值 | 80.0 |
| `--output, -o` | 输出图像路径 | detection_result.png |
| `--no-display` | 不显示结果窗口 | False |

---

## C++ 版本

### 编译

```bash
# 方法1: 使用独立的 CMakeLists
cp CMakeLists_detector.txt CMakeLists.txt
mkdir build && cd build
cmake ..
make

# 方法2: 使用 pkg-config 一键编译
g++ -o opencv_detector opencv_detector.cpp `pkg-config --cflags --libs opencv4`
```

### 使用方法

```bash
# 基本用法
./opencv_detector <template.yml> <color_image> [depth_image]

# 带深度图
./opencv_detector linemod_templates.yml.gz benchmark/img0.png benchmark/depth0.png

# 仅使用颜色模态
./opencv_detector -c linemod_templates.yml benchmark/img0.png

# 调整阈值
./opencv_detector -t 70 linemod_templates.yml benchmark/img0.png
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<template.yml>` | 模板文件路径 | 必需 |
| `<color_image>` | 彩色图像路径 | 必需 |
| `[depth_image]` | 深度图像路径 | 可选 |
| `-t <threshold>` | 检测阈值 | 80.0 |
| `-c` | 仅使用颜色模态 | False |
| `-d` | 使用深度模态 | True |
| `-h` | 显示帮助 | - |

---

## 输出说明

### 检测结果结构

每个匹配包含：

| 字段 | 说明 |
|------|------|
| `class_id` | 类别名称（如 "lagergehaeuse.ply"） |
| `template_id` | 模板 ID |
| `similarity` | 相似度得分（越高越好） |
| `position` | 匹配位置 (x, y) |
| `bounding_box` | 边界框 (x, y, width, height) |

### 可视化说明

- **彩色点**: 模板特征点
- **矩形框**: 匹配边界框
- **白点**: 匹配中心位置
- **文字标签**: 类别名 + 模板ID + 相似度

---

## 示例

### 测试当前项目生成的模板

```bash
# Python 版本
python opencv_detector.py linemod_templates.yml.gz benchmark/img0.png --depth benchmark/depth0.png

# C++ 版本
./opencv_detector linemod_templates.yml.gz benchmark/img0.png benchmark/depth0.png
```

### 预期输出

```
==================================================
OpenCV LINE-MOD Detector (Python)
==================================================
Template file: linemod_templates.yml.gz
Color image: benchmark/img0.png
Depth image: benchmark/depth0.png
Threshold: 80.0

Loading templates...
Loaded detector with:
  - 1 classes
  - 1950 templates
  - Class IDs: ['lagergehaeuse.ply']

Loading images...
Color image size: 640x480
Depth image size: 640x480

Running detection...
Detection time: 15.23 ms

Detection Results:
--------------------------------------------------
[0] Class: lagergehaeuse.ply, Template: 123, Similarity: 92.5, Position: (320, 240)
    BBox: 150x120

Result saved to: detection_result.png
```

---

## 注意事项

1. **模板文件**: 必须使用当前项目生成的 `linemod_templates.yml.gz` 文件

2. **深度图像**: 
   - 如果模板是用颜色+深度生成的，检测时也应提供深度图
   - 深度图应为 16-bit PNG 格式（单位：毫米）

3. **阈值调整**:
   - 阈值越高，匹配越严格（误检少，漏检多）
   - 阈值越低，匹配越宽松（误检多，漏检少）
   - 推荐范围：70-90

4. **多目标检测**:
   - 当前版本会返回所有超过阈值的匹配
   - 可通过位置聚类过滤重复检测

---

## 与原项目 Detector 的对比

| 功能 | 原项目 Detector | OpenCV Detector |
|------|----------------|-----------------|
| 模板加载 | ✅ | ✅ |
| 目标检测 | ✅ | ✅ |
| 位姿估计 | ✅ | ❌ |
| ICP 优化 | ✅ | ❌ |
| 颜色验证 | ✅ | ❌ |
| 深度验证 | ✅ | ❌ |
| Benchmark | ✅ | ❌ |

**总结**: OpenCV Detector 是简化版本，仅提供基础的 2D 检测功能，适合快速验证模板效果。
