#include <glog/logging.h>

#include "PoseDetection.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  FLAGS_logbufsecs = 0;

  PoseDetection poseDetect;
  poseDetect.setupBenchmark(
      "lagergehaeuse.ply");  // Uncomment if Benchmark is not wanted

  std::string root_path = "/home/juling/Documents/projects/LINE-MOD-Pipeline/";
  cv::Mat colorImg, depthImg;
  colorImg = cv::imread(root_path + "benchmark/img0.png", cv::IMREAD_COLOR);
  depthImg =
      cv::imread(root_path + "benchmark/depth0.png", cv::IMREAD_ANYDEPTH);

  cv::Mat depth8u;
  cv::normalize(depthImg, depth8u, 0, 255, cv::NORM_MINMAX);
  depth8u.convertTo(depth8u, CV_8U);

  std::vector<cv::Mat> imgs;
  imgs.push_back(colorImg);
  imgs.push_back(depthImg);

  std::vector<ObjectPose> objPose;
  cv::TickMeter tm;
  tm.start();
  poseDetect.detect(imgs, "lagergehaeuse.ply", 1, objPose, true);
  tm.stop();
  LOG(WARNING) << "Detection time: " << tm.getTimeMilli() << " ms" << std::endl;

  // cv::imshow("view depth", depth8u);
  // cv::imshow("view color", colorImg);
  cv::waitKey(0);

  google::ShutdownGoogleLogging();
  return 0;

  if (0) {
    int counter = 0;

    ///////IMAGE SOURCES:
    cv::VideoCapture sequence(root_path + "benchmark/img%0d.png");
    // cv::VideoCapture sequence("benchmarkLINEMOD/color%0d.jpg");
    // Kinect2 kin2;
    /////////////////////

    if (!sequence.isOpened()) {
      LOG(ERROR) << "Failed to open image sequence!";
      return -1;
    }

    while (true) {
      std::vector<cv::Mat> imgs;
      cv::Mat colorImg;
      cv::Mat depthImg;

      ///////IMAGE SOURCES:
      sequence >> colorImg;
      depthImg =
          cv::imread("benchmark/depth" + std::to_string(counter) + ".png",
                     cv::IMREAD_ANYDEPTH);
      // Video Capture does not work with 16bit png on linux
      // depthImg = loadDepthLineModDataset("benchmarkLINEMOD/depth" +
      // std::to_string(counter) + ".dpt"); kin2.getKinectFrames(colorImg,
      // depthImg);
      ////////////////////

      imgs.push_back(colorImg);
      imgs.push_back(depthImg);

      std::vector<ObjectPose> objPose;

      cv::TickMeter tm;
      tm.start();
      poseDetect.detect(imgs, "lagergehaeuse.ply", 1, objPose, true);
      tm.stop();
      LOG(ERROR) << "Detection time: " << tm.getTimeMilli() << " ms"
                 << std::endl;

      counter++;
    }
  }
}
