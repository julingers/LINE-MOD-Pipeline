/*
 * @Author: juling julinger@qq.com
 * @Date: 2026-03-11 14:18:01
 * @LastEditors: juling julinger@qq.com
 * @LastEditTime: 2026-03-19 11:27:15
 */

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd/linemod.hpp>

// 相机参数
struct CameraParams {
  float fx, fy, cx, cy;
  int width, height;
  cv::Mat cameraMatrix;
};

// 模板位姿数据（与linemod_tempPosFile.bin格式一致）
struct TemplatePose {
  float tx, ty, tz;        // 平移
  float qx, qy, qz, qw;    // 四元数旋转
  int bbX, bbY, bbW, bbH;  // 边界框
  uint16_t medianDepth;    // 深度中值
};

struct DetectionResult {
  std::string classId;
  int templateId;
  float similarity;
  cv::Point position;
  cv::Rect boundingBox;
  // 计算后的位姿
  cv::Vec3f translation;
  cv::Mat rotation;
};

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

CameraParams loadCameraParams(
    const std::string& filename = "linemod_settings.yml") {
  CameraParams params;
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    LOG(ERROR) << "Cannot open settings file: " << filename
               << ". Using default camera parameters";
    // 默认参数
    params.width = 640;
    params.height = 480;
    params.fx = 1044.87f;
    params.fy = 1045.69f;
    params.cx = 320.0f;
    params.cy = 240.0f;
  } else {
    fs["video width"] >> params.width;
    fs["video height"] >> params.height;
    fs["camera fx"] >> params.fx;
    fs["camera fy"] >> params.fy;
    fs["camera cx"] >> params.cx;
    fs["camera cy"] >> params.cy;
    fs.release();
  }

  params.cameraMatrix = (cv::Mat1d(3, 3) << params.fx, 0, params.cx, 0,
                         params.fy, params.cy, 0, 0, 1);

  LOG(INFO) << "Camera parameters: fx=" << params.fx << ", fy=" << params.fy
            << ", cx=" << params.cx << ", cy=" << params.cy;

  return params;
}

std::vector<std::vector<TemplatePose>> loadTemplatePoses(
    const std::string& filename) {
  std::vector<std::vector<TemplatePose>> modelTemplates;

  std::ifstream input(filename, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    LOG(ERROR) << "Cannot open template pose file: " << filename;
    return modelTemplates;
  }

  uint32_t numTempVecs;
  input.read((char*)&numTempVecs, sizeof(uint32_t));
  LOG(INFO) << "Loading " << numTempVecs << " classes of template poses";

  for (uint32_t i = 0; i < numTempVecs; i++) {
    std::vector<TemplatePose> templates;
    uint64_t numTemp;
    input.read((char*)&numTemp, sizeof(uint64_t));

    for (uint64_t j = 0; j < numTemp; j++) {
      TemplatePose tp;
      input.read((char*)&tp, sizeof(TemplatePose));
      templates.push_back(tp);
    }
    modelTemplates.push_back(templates);
    LOG(INFO) << "  Class " << i << ": " << templates.size() << " templates";
  }

  input.close();
  return modelTemplates;
}

// 四元数转旋转矩阵
cv::Mat quaternionToRotationMatrix(float qx, float qy, float qz, float qw) {
  cv::Mat R(3, 3, CV_32F);

  // 归一化
  float norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  qx /= norm;
  qy /= norm;
  qz /= norm;
  qw /= norm;

  R.at<float>(0, 0) = 1 - 2 * (qy * qy + qz * qz);
  R.at<float>(0, 1) = 2 * (qx * qy - qz * qw);
  R.at<float>(0, 2) = 2 * (qx * qz + qy * qw);
  R.at<float>(1, 0) = 2 * (qx * qy + qz * qw);
  R.at<float>(1, 1) = 1 - 2 * (qx * qx + qz * qz);
  R.at<float>(1, 2) = 2 * (qy * qz - qx * qw);
  R.at<float>(2, 0) = 2 * (qx * qz - qy * qw);
  R.at<float>(2, 1) = 2 * (qy * qz + qx * qw);
  R.at<float>(2, 2) = 1 - 2 * (qx * qx + qy * qy);

  return R;
}

// 计算最终位姿
void computeFinalPose(DetectionResult& result, const TemplatePose& templatePose,
                      const CameraParams& camParams, const cv::Mat& depthImg) {
  // 1. 获取模板的初始旋转
  cv::Mat templateRot = quaternionToRotationMatrix(
      templatePose.qx, templatePose.qy, templatePose.qz, templatePose.qw);

  // 2. 计算匹配位置对应的像素坐标
  float pixelX = result.position.x + camParams.width / 2 - templatePose.bbX;
  float pixelY = result.position.y + camParams.height / 2 - templatePose.bbY;

  // 3. 获取深度（从深度图或使用模板的中值深度）
  float depth = templatePose.tz;  // 默认使用模板深度
  if (!depthImg.empty()) {
    // 从深度图获取中心深度
    int centerZ = result.position.y + result.boundingBox.height / 2;
    int centerY = result.position.x + result.boundingBox.width / 2;
    if (centerY >= 0 && centerY < depthImg.cols && centerZ >= 0 &&
        centerZ < depthImg.rows) {
      uint16_t depthValue = depthImg.at<uint16_t>(centerZ, centerY);
      if (depthValue > 0) {
        depth = depthValue;
      }
    }
  }

  // 4. 计算到图像中心的像素距离
  float offsetX = pixelX - camParams.width / 2;
  float offsetY = pixelY - camParams.height / 2;
  float pixelDist = std::sqrt(offsetX * offsetX + offsetY * offsetY);

  // 5. 计算真实的Z值（考虑透视）
  float trueZ =
      std::sqrt(depth * depth - pixelDist * pixelDist * (depth / camParams.fy) *
                                    (depth / camParams.fy));
  if (std::isnan(trueZ) || trueZ < 0) trueZ = depth;

  // 6. 计算3D平移
  float scale = trueZ / camParams.fy;
  float tx = (pixelX - camParams.cx) * scale;
  float ty = (pixelY - camParams.cy) * scale;
  float tz = trueZ;

  result.translation = cv::Vec3f(tx, ty, tz);

  // 7. 调整旋转矩阵（根据新的位置调整look-at方向）
  // 简化处理：直接使用模板旋转
  result.rotation = templateRot;
}

// 绘制坐标轴
void drawCoordinateAxis(cv::Mat& image, const CameraParams& camParams,
                        const DetectionResult& result,
                        float axisLength = 50.0f) {
  std::vector<cv::Point3f> axisPoints;
  axisPoints.push_back(cv::Point3f(0, 0, 0));           // 原点
  axisPoints.push_back(cv::Point3f(axisLength, 0, 0));  // X轴
  axisPoints.push_back(cv::Point3f(0, axisLength, 0));  // Y轴
  axisPoints.push_back(cv::Point3f(0, 0, axisLength));  // Z轴

  // 旋转向量
  cv::Mat rvec;
  cv::Rodrigues(result.rotation, rvec);

  // 平移向量
  cv::Mat tvec = (cv::Mat1f(3, 1) << result.translation[0],
                  result.translation[1], result.translation[2]);

  // 投影到2D
  std::vector<cv::Point2f> projectedPoints;
  cv::projectPoints(axisPoints, rvec, tvec, camParams.cameraMatrix, cv::Mat(),
                    projectedPoints);

  // 绘制坐标轴
  cv::line(image, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255),
           2);  // X-红
  cv::line(image, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0),
           2);  // Y-绿
  cv::line(image, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0),
           2);  // Z-蓝
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
      LOG(INFO) << "Using compressed file: " << gzFile;
    }
    test.close();
  }

  cv::FileStorage fs(actualFile, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Cannot open template file: " << actualFile;
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

  LOG(INFO) << "Loaded detector with:";
  LOG(INFO) << "  - " << detector->numClasses() << " classes";
  LOG(INFO) << "  - " << detector->numTemplates() << " templates";

  std::vector<cv::String> classIds = detector->classIds();
  LOG(INFO) << "  - Class IDs: ";
  for (const auto& id : classIds) {
    LOG(INFO) << id << " ";
  }
  std::cout << std::endl;

  return detector;
}

std::vector<DetectionResult> detect(
    cv::Ptr<cv::linemod::Detector>& detector, const cv::Mat& colorImg,
    const cv::Mat& depthImg,
    const std::vector<std::vector<TemplatePose>>& templatePoses,
    const CameraParams& camParams, float threshold = 80.0f,
    bool useDepth = true) {
  std::vector<DetectionResult> results;

  int numModalities = detector->getModalities().size();
  LOG(INFO) << "Detector uses " << numModalities << " modalities";

  std::vector<cv::Mat> sources;
  sources.push_back(colorImg);

  if (numModalities == 2) {
    if (depthImg.empty()) {
      LOG(ERROR) << "Error: Detector requires depth image but none provided!";
      return results;
    }
    sources.push_back(depthImg);
    LOG(INFO) << "Using color + depth modalities";
  } else if (numModalities == 1) {
    LOG(INFO) << "Using color modality only";
  }

  std::vector<cv::linemod::Match> matches;
  detector->match(sources, threshold, matches);

  LOG(INFO) << "Found " << matches.size() << " matches";

  // 获取 class id 到索引的映射
  std::vector<cv::String> classIds = detector->classIds();

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

    // 计算位姿
    int classIdx = -1;
    for (size_t i = 0; i < classIds.size(); i++) {
      if (classIds[i] == match.class_id) {
        classIdx = i;
        break;
      }
    }

    if (classIdx >= 0 && classIdx < (int)templatePoses.size() &&
        match.template_id < (int)templatePoses[classIdx].size()) {
      computeFinalPose(result, templatePoses[classIdx][match.template_id],
                       camParams, depthImg);
    } else {
      // 默认位姿
      result.translation = cv::Vec3f(0, 0, 500);
      result.rotation = cv::Mat::eye(3, 3, CV_32F);
    }

    results.push_back(result);
  }

  return results;
}

void drawResults(cv::Mat& image, cv::Ptr<cv::linemod::Detector>& detector,
                 const std::vector<DetectionResult>& results,
                 const CameraParams& camParams) {
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

    // 绘制坐标轴
    drawCoordinateAxis(image, camParams, result, 50.0f);

    // 绘制边界框
    cv::rectangle(image, result.boundingBox, color, 2);

    // 绘制匹配位置（中心点）
    cv::circle(image, result.position, 5, cv::Scalar(255, 255, 255), -1);
    cv::circle(image, result.position, 3, color, -1);

    // 绘制标签（包含位姿信息）
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
  std::cout << "  -i <iou>         IoU threshold for NMS (default: 0.1)"
            << std::endl;
  std::cout << "  -n <max>         Max number of results (default: 3)"
            << std::endl;
  std::cout << "  -h               Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout
      << "  " << programName
      << " linemod_templates.yml.gz benchmark/img0.png benchmark/depth0.png"
      << std::endl;
  std::cout << "  " << programName
            << " -t 70 -i 0.5 linemod_templates.yml benchmark/img0.png"
            << std::endl;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  FLAGS_logbufsecs = 0;

  std::cout << "Usage: " << argv[0]
            << " <template.yml> <color_image> [depth_image]" << std::endl;
  std::string templateFile;
  std::string colorFile;
  std::string depthFile;
  float threshold = 80.0f;
  float iouThreshold = 0.1f;
  int maxResults = 3;

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
  bool useDepth = !depthFile.empty();

  std::cout << "\n===== OpenCV LINE-MOD Detector =====" << std::endl;
  std::cout << "Template file: " << templateFile << std::endl;
  std::cout << "Color image: " << colorFile << std::endl;
  if (!depthFile.empty()) {
    std::cout << "Depth image: " << depthFile << std::endl;
  }
  std::cout << "Threshold: " << threshold << std::endl;
  std::cout << "Use depth: " << (useDepth ? "yes" : "no") << std::endl;
  std::cout << "NMS: " << "enabled";
  std::cout << " (IoU=" << iouThreshold << ", MaxResults: " << maxResults << ")"
            << std::endl;
  std::cout << "===== OpenCV LINE-MOD Detector =====\n" << std::endl;

  CameraParams camParams = loadCameraParams();
  cv::Ptr<cv::linemod::Detector> detector = loadDetector(templateFile);
  if (!detector) {
    LOG(ERROR) << "Failed to load detector!";
    return -1;
  }

  cv::Mat colorImg = cv::imread(colorFile, cv::IMREAD_COLOR);
  if (colorImg.empty()) {
    LOG(ERROR) << "Cannot load color image: " << colorFile;
    return -1;
  }

  cv::Mat depthImg;
  if (!depthFile.empty()) {
    depthImg = cv::imread(depthFile, cv::IMREAD_ANYDEPTH);
    if (depthImg.empty()) {
      LOG(WARNING) << "Cannot load depth image: " << depthFile
                   << ". Continuing without depth...";
    }
  }

  LOG(INFO) << "Color image size: " << colorImg.cols << "x" << colorImg.rows;
  if (!depthImg.empty()) {
    LOG(INFO) << "Depth image size: " << depthImg.cols << "x" << depthImg.rows;
  }

  cv::TickMeter tm;
  tm.start();

  std::string postfix = !useDepth ? "_color" : "_color_depth";
  std::string tmp_pose_file = "linemod_tempPosFile" + postfix + ".bin";
  std::vector<std::vector<TemplatePose>> templatePoses =
      loadTemplatePoses(tmp_pose_file);
  std::vector<DetectionResult> results =
      detect(detector, colorImg, depthImg, templatePoses, camParams, threshold,
             useDepth);

  tm.stop();
  LOG(WARNING) << "Detection time: " << tm.getTimeMilli() << " ms";

  if (!results.empty()) {
    size_t beforeNms = results.size();
    results = nmsFilter(results, iouThreshold, maxResults);
    LOG(INFO) << "NMS: " << beforeNms << " -> " << results.size() << " results";
  }

  std::cout << std::endl;

  if (!results.empty()) {
    std::cout << "Detection Results:" << std::endl;
    std::cout << "------------------" << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
      const auto& r = results[i];
      std::cout << "[" << i << "] Class: " << r.classId
                << ", Template: " << r.templateId
                << ", Similarity: " << r.similarity << std::endl;
      std::cout << "    Position: (" << r.position.x << ", " << r.position.y
                << ")"
                << ", BBox: " << r.boundingBox.width << "x"
                << r.boundingBox.height << std::endl;
      std::cout << "    Translation (mm): [" << r.translation[0] << ", "
                << r.translation[1] << ", " << r.translation[2] << "]"
                << std::endl;
    }
    std::cout << std::endl;
  } else {
    LOG(ERROR) << "No matches found!";
  }

  cv::Mat resultImg = colorImg.clone();
  drawResults(resultImg, detector, results, camParams);

  cv::namedWindow("LINE-MOD Detection Result", cv::WINDOW_AUTOSIZE);
  cv::imshow("LINE-MOD Detection Result", resultImg);
  cv::waitKey(0);

  std::string outputFile = "detection_result.png";
  cv::imwrite(outputFile, resultImg);
  LOG(INFO) << "Result saved to: " << outputFile;

  return 0;
}
