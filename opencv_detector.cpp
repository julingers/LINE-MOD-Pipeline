#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd/linemod.hpp>

struct CameraParams {
  float fx, fy, cx, cy;
  int width, height;
};

struct DetectionResult {
  std::string classId;
  int templateId;
  float similarity;
  cv::Point position;
  cv::Rect boundingBox;
};

// 计算IoU
float calculateIoU(const cv::Rect& a, const cv::Rect& b) {
  int interX1 = std::max(a.x, b.x);
  int interY1 = std::max(a.y, b.y);
  int interX2 = std::min(a.x + a.width, b.x + b.width);
  int interY2 = std::min(a.y + a.height, b.y + b.height);

  if (interX2 < interX1 || interY2 < interY1) {
    return 0.0f;
  }

  float interArea = (interX2 - interX1) * (interY2 - interY1);
  float unionArea = a.area() + b.area() - interArea;

  return interArea / unionArea;
}

// NMS过滤
std::vector<DetectionResult> nmsFilter(std::vector<DetectionResult>& results,
                                       float iouThreshold = 0.3f,
                                       int maxResults = 10) {
  if (results.empty()) {
    return results;
  }

  // 按相似度降序排序
  std::sort(results.begin(), results.end(),
            [](const DetectionResult& a, const DetectionResult& b) {
              return a.similarity > b.similarity;
            });

  std::vector<DetectionResult> filtered;
  std::vector<bool> suppressed(results.size(), false);

  for (size_t i = 0; i < results.size() && filtered.size() < (size_t)maxResults;
       i++) {
    if (suppressed[i]) {
      continue;
    }

    filtered.push_back(results[i]);

    // 抑制与当前结果重叠度高的结果
    for (size_t j = i + 1; j < results.size(); j++) {
      if (suppressed[j]) {
        continue;
      }

      float iou = calculateIoU(results[i].boundingBox, results[j].boundingBox);
      if (iou > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return filtered;
}

cv::Ptr<cv::linemod::Detector> loadDetector(const std::string& filename) {
  cv::Ptr<cv::linemod::Detector> detector =
      cv::makePtr<cv::linemod::Detector>();

  std::string actualFile = filename;
  if (filename.find(".gz") == std::string::npos) {
    std::string gzFile = filename;
    if (gzFile.substr(gzFile.length() - 4) == ".yml") {
      gzFile = gzFile.substr(0, gzFile.length() - 4) + ".yml.gz";
    }
    std::ifstream test(gzFile);
    if (test.good()) {
      actualFile = gzFile;
      std::cout << "Using compressed file: " << gzFile << std::endl;
    }
    test.close();
  }

  cv::FileStorage fs(actualFile, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Error: Cannot open template file: " << actualFile
              << std::endl;
    return nullptr;
  }

  detector->read(fs.root());

  cv::FileNode classesNode = fs["classes"];
  if (!classesNode.empty()) {
    for (cv::FileNodeIterator it = classesNode.begin(); it != classesNode.end();
         ++it) {
      detector->readClass(*it);
    }
  }

  fs.release();

  std::cout << "Loaded detector with:" << std::endl;
  std::cout << "  - " << detector->numClasses() << " classes" << std::endl;
  std::cout << "  - " << detector->numTemplates() << " templates" << std::endl;

  std::vector<cv::String> classIds = detector->classIds();
  std::cout << "  - Class IDs: ";
  for (const auto& id : classIds) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  return detector;
}

std::vector<DetectionResult> detect(cv::Ptr<cv::linemod::Detector>& detector,
                                    const cv::Mat& colorImg,
                                    const cv::Mat& depthImg,
                                    float threshold = 80.0f,
                                    bool useDepth = true) {
  std::vector<DetectionResult> results;

  int numModalities = detector->getModalities().size();
  std::cout << "Detector uses " << numModalities << " modalities" << std::endl;

  std::vector<cv::Mat> sources;
  sources.push_back(colorImg);

  if (numModalities == 2) {
    if (depthImg.empty()) {
      std::cerr << "Error: Detector requires depth image but none provided!"
                << std::endl;
      return results;
    }
    sources.push_back(depthImg);
    std::cout << "Using color + depth modalities" << std::endl;
  } else if (numModalities == 1) {
    std::cout << "Using color modality only" << std::endl;
  }

  std::vector<cv::linemod::Match> matches;
  detector->match(sources, threshold, matches);

  std::cout << "Found " << matches.size() << " matches" << std::endl;

  for (const auto& match : matches) {
    DetectionResult result;
    result.classId = match.class_id;
    result.templateId = match.template_id;
    result.similarity = match.similarity;
    result.position = cv::Point(match.x, match.y);

    // 获取边界框（只使用第一个模态）
    try {
      std::vector<cv::linemod::Template> templates =
          detector->getTemplates(match.class_id, match.template_id);
      if (!templates.empty()) {
        // 只使用第一个模态（ColorGradient）计算包围盒
        const auto& t = templates[0];
        int minX = INT_MAX, minY = INT_MAX;
        int maxX = INT_MIN, maxY = INT_MIN;
        for (const auto& f : t.features) {
          minX = std::min(minX, (int)f.x);
          minY = std::min(minY, (int)f.y);
          maxX = std::max(maxX, (int)f.x);
          maxY = std::max(maxY, (int)f.y);
        }
        result.boundingBox = cv::Rect(match.x + minX, match.y + minY,
                                      maxX - minX + 1, maxY - minY + 1);
      }
    } catch (...) {
      result.boundingBox = cv::Rect(match.x - 50, match.y - 50, 100, 100);
    }

    results.push_back(result);
  }

  return results;
}

void drawResults(cv::Mat& image, cv::Ptr<cv::linemod::Detector>& detector,
                 const std::vector<DetectionResult>& results) {
  cv::Scalar colors[] = {
      cv::Scalar(0, 255, 0),    // 绿色
      cv::Scalar(0, 0, 255),    // 红色
      cv::Scalar(255, 0, 0),    // 蓝色
      cv::Scalar(0, 255, 255),  // 黄色
      cv::Scalar(255, 0, 255),  // 紫色
  };
  int colorIdx = 0;

  for (const auto& result : results) {
    cv::Scalar color = colors[colorIdx % 5];
    colorIdx++;

    // 绘制特征点（只绘制第一个模态ColorGradient）
    try {
      std::vector<cv::linemod::Template> templates =
          detector->getTemplates(result.classId, result.templateId);

      if (!templates.empty()) {
        // 获取第一个模态的T参数用于绘制圆半径
        int T = detector->getT(0);
        for (const auto& feature : templates[0].features) {
          cv::Point pt(feature.x + result.position.x,
                       feature.y + result.position.y);
          cv::circle(image, pt, T / 2, color, -1);
        }
      }
    } catch (...) {
      // 如果无法获取模板，跳过特征点绘制
    }

    // 绘制边界框
    cv::rectangle(image, result.boundingBox, color, 2);

    // 绘制匹配位置（中心点）
    cv::circle(image, result.position, 5, cv::Scalar(255, 255, 255), -1);
    cv::circle(image, result.position, 3, color, -1);

    // 绘制标签
    std::string label = result.classId + " (T" +
                        std::to_string(result.templateId) + ", S" +
                        std::to_string((int)result.similarity) + ")";

    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::Point textOrg(result.boundingBox.x, result.boundingBox.y - 5);

    // 绘制背景矩形
    cv::rectangle(image,
                  cv::Point(textOrg.x - 2, textOrg.y - textSize.height - 2),
                  cv::Point(textOrg.x + textSize.width + 2, textOrg.y + 2),
                  cv::Scalar(0, 0, 0), -1);

    // 绘制文字
    cv::putText(image, label, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }
}

void printUsage(const char* programName) {
  std::cout << "Usage: " << programName
            << " [options] <template.yml> <color_image> [depth_image]"
            << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -t <threshold>   Detection threshold (default: 80.0)"
            << std::endl;
  std::cout << "  -i <iou>         IoU threshold for NMS (default: 0.3)"
            << std::endl;
  std::cout << "  -n <max>         Max number of results (default: 10)"
            << std::endl;
  std::cout << "  --no-nms         Disable NMS filtering" << std::endl;
  std::cout << "  -d               Use depth modality (default: true)"
            << std::endl;
  std::cout << "  -c               Color modality only (ignore depth)"
            << std::endl;
  std::cout << "  -h               Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout
      << "  " << programName
      << " linemod_templates.yml.gz benchmark/img0.png benchmark/depth0.png"
      << std::endl;
  std::cout << "  " << programName
            << " -t 70 -i 0.5 -n 5 linemod_templates.yml benchmark/img0.png"
            << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Usage: " << argv[0]
            << " <template.yml> <color_image> [depth_image]" << std::endl;
  std::string templateFile;
  std::string colorFile;
  std::string depthFile;
  float threshold = 80.0f;
  float iouThreshold = 0.1f;
  int maxResults = 1;
  bool useNms = true;
  bool useDepth = true;

  int argIdx = 1;
  while (argIdx < argc) {
    std::string arg = argv[argIdx];
    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "-t") {
      if (argIdx + 1 < argc) {
        threshold = std::stof(argv[++argIdx]);
      }
      argIdx++;
    } else if (arg == "-i") {
      if (argIdx + 1 < argc) {
        iouThreshold = std::stof(argv[++argIdx]);
      }
      argIdx++;
    } else if (arg == "-n") {
      if (argIdx + 1 < argc) {
        maxResults = std::stoi(argv[++argIdx]);
      }
      argIdx++;
    } else if (arg == "--no-nms") {
      useNms = false;
      argIdx++;
    } else if (arg == "-c") {
      useDepth = false;
      argIdx++;
    } else if (arg == "-d") {
      useDepth = true;
      argIdx++;
    } else if (arg[0] != '-') {
      if (templateFile.empty()) {
        templateFile = arg;
      } else if (colorFile.empty()) {
        colorFile = arg;
      } else if (depthFile.empty()) {
        depthFile = arg;
      }
      argIdx++;
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      printUsage(argv[0]);
      return -1;
    }
  }

  if (templateFile.empty() || colorFile.empty()) {
    std::cerr << "Error: Template file and color image are required!"
              << std::endl;
    printUsage(argv[0]);
    return -1;
  }

  std::cout << "===== OpenCV LINE-MOD Detector =====" << std::endl;
  std::cout << "Template file: " << templateFile << std::endl;
  std::cout << "Color image: " << colorFile << std::endl;
  if (!depthFile.empty()) {
    std::cout << "Depth image: " << depthFile << std::endl;
  }
  std::cout << "Threshold: " << threshold << std::endl;
  std::cout << "Use depth: " << (useDepth ? "yes" : "no") << std::endl;
  std::cout << "NMS: " << (useNms ? "enabled" : "disabled");
  if (useNms) {
    std::cout << " (IoU=" << iouThreshold << ", max=" << maxResults << ")";
  }
  std::cout << std::endl << std::endl;

  std::cout << "Loading detector..." << std::endl;
  cv::Ptr<cv::linemod::Detector> detector = loadDetector(templateFile);
  if (!detector) {
    return -1;
  }

  std::cout << "Loading images..." << std::endl;
  cv::Mat colorImg = cv::imread(colorFile, cv::IMREAD_COLOR);
  if (colorImg.empty()) {
    std::cerr << "Error: Cannot load color image: " << colorFile << std::endl;
    return -1;
  }

  cv::Mat depthImg;
  if (!depthFile.empty()) {
    depthImg = cv::imread(depthFile, cv::IMREAD_ANYDEPTH);
    if (depthImg.empty()) {
      std::cerr << "Warning: Cannot load depth image: " << depthFile
                << std::endl;
      std::cerr << "         Continuing without depth..." << std::endl;
    }
  }

  std::cout << "Color image size: " << colorImg.cols << "x" << colorImg.rows
            << std::endl;
  if (!depthImg.empty()) {
    std::cout << "Depth image size: " << depthImg.cols << "x" << depthImg.rows
              << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Running detection..." << std::endl;
  cv::TickMeter tm;
  tm.start();

  std::vector<DetectionResult> results =
      detect(detector, colorImg, depthImg, threshold, useDepth);

  tm.stop();
  std::cout << "Detection time: " << tm.getTimeMilli() << " ms" << std::endl;

  // 应用 NMS
  if (useNms && !results.empty()) {
    size_t beforeNms = results.size();
    results = nmsFilter(results, iouThreshold, maxResults);
    std::cout << "NMS: " << beforeNms << " -> " << results.size() << " results"
              << std::endl;
  }

  std::cout << std::endl;

  if (!results.empty()) {
    std::cout << "Detection Results:" << std::endl;
    std::cout << "------------------" << std::endl;
    if (1) {
      for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        std::cout << "[" << i << "] Class: " << r.classId
                  << ", Template: " << r.templateId
                  << ", Similarity: " << r.similarity << ", Position: ("
                  << r.position.x << ", " << r.position.y << ")"
                  << ", BBox: " << r.boundingBox.width << "x"
                  << r.boundingBox.height << std::endl;
      }
    }

    std::cout << std::endl;
  } else {
    std::cout << "No matches found!" << std::endl;
  }

  cv::Mat resultImg = colorImg.clone();
  drawResults(resultImg, detector, results);

  cv::namedWindow("LINE-MOD Detection Result", cv::WINDOW_AUTOSIZE);
  cv::imshow("LINE-MOD Detection Result", resultImg);

  std::cout << "Press any key to exit..." << std::endl;
  cv::waitKey(0);

  std::string outputFile = "detection_result.png";
  cv::imwrite(outputFile, resultImg);
  std::cout << "Result saved to: " << outputFile << std::endl;

  return 0;
}
