// ============================================================
// Mediapipe Holistic - C++ Implementation
// Target: Hands + Pose detection (lebih cepat dari Python)
// Build: Lihat README_CPP_BUILD.md
// ============================================================

#include <cstdlib>
#include <string>
#include <chrono>
#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

// ── Konfigurasi ──────────────────────────────────────────────
ABSL_FLAG(std::string, input_video, "",
          "Path ke file video. Kosongkan untuk pakai kamera.");
ABSL_FLAG(int, camera_index, 0, "Index kamera (default: 0)");
ABSL_FLAG(int, process_width, 640, "Lebar frame untuk processing (default: 640)");

// ── Graph Config: Holistic (Pose + Hands) ────────────────────
constexpr char kHolisticGraphConfig[] = R"pb(
  input_stream: "input_video"
  output_stream: "pose_landmarks"
  output_stream: "left_hand_landmarks"
  output_stream: "right_hand_landmarks"

  node {
    calculator: "HolisticLandmarkCpu"
    input_stream: "IMAGE:input_video"
    output_stream: "POSE_LANDMARKS:pose_landmarks"
    output_stream: "LEFT_HAND_LANDMARKS:left_hand_landmarks"
    output_stream: "RIGHT_HAND_LANDMARKS:right_hand_landmarks"
    node_options: {
      [type.googleapis.com/mediapipe.HolisticLandmarkCalculatorOptions] {
        # Hanya Pose + Hands, tanpa Face Mesh (lebih ringan)
        use_face_detection_input_source: false
      }
    }
  }
)pb";

// ── Gambar landmark di frame ─────────────────────────────────
void DrawLandmarks(cv::Mat& frame,
                   const mediapipe::NormalizedLandmarkList& landmarks,
                   const cv::Scalar& color, int radius = 4) {
    int w = frame.cols, h = frame.rows;
    for (const auto& lm : landmarks.landmark()) {
        int x = static_cast<int>(lm.x() * w);
        int y = static_cast<int>(lm.y() * h);
        cv::circle(frame, {x, y}, radius, color, -1);
    }
}

// ── Main ─────────────────────────────────────────────────────
absl::Status RunGraph() {
    // Setup graph Mediapipe
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            kHolisticGraphConfig)));

    // Output streams
    MP_ASSIGN_OR_RETURN(
        auto pose_poller,
        graph.AddOutputStreamPoller("pose_landmarks"));
    MP_ASSIGN_OR_RETURN(
        auto lhand_poller,
        graph.AddOutputStreamPoller("left_hand_landmarks"));
    MP_ASSIGN_OR_RETURN(
        auto rhand_poller,
        graph.AddOutputStreamPoller("right_hand_landmarks"));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    // Buka sumber video: file atau kamera
    cv::VideoCapture cap;
    std::string input_path = absl::GetFlag(FLAGS_input_video);
    bool is_camera = input_path.empty();

    if (is_camera) {
        cap.open(absl::GetFlag(FLAGS_camera_index));
        std::cout << "[INFO] Menggunakan kamera index "
                  << absl::GetFlag(FLAGS_camera_index) << std::endl;
    } else {
        cap.open(input_path);
        std::cout << "[INFO] Membaca file: " << input_path << std::endl;
    }

    if (!cap.isOpened()) {
        return absl::NotFoundError("Tidak bisa membuka sumber video.");
    }

    // Ambil FPS video untuk sinkronisasi (hanya untuk file)
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    if (video_fps <= 0) video_fps = 30.0;

    const int process_width = absl::GetFlag(FLAGS_process_width);

    // FPS counter
    auto prev_time = std::chrono::steady_clock::now();
    size_t frame_idx = 0;

    while (cap.isOpened()) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // [OPTIMASI] Resize frame ke process_width untuk input Mediapipe
        cv::Mat small_frame;
        double scale = static_cast<double>(process_width) / frame.cols;
        cv::resize(frame, small_frame,
                   {process_width, static_cast<int>(frame.rows * scale)},
                   0, 0, cv::INTER_LINEAR);

        // Konversi BGR → RGB untuk Mediapipe
        cv::Mat small_rgb;
        cv::cvtColor(small_frame, small_rgb, cv::COLOR_BGR2RGB);

        // Kirim frame ke graph Mediapipe
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB,
            small_rgb.cols, small_rgb.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        small_rgb.copyTo(mediapipe::formats::MatView(input_frame.get()));

        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            "input_video",
            mediapipe::Adopt(input_frame.release())
                .At(mediapipe::Timestamp(frame_idx++))));

        // Ambil hasil deteksi
        mediapipe::Packet pose_pkt, lhand_pkt, rhand_pkt;

        // Gambar hasil di frame ORIGINAL (koordinat normalized → skala ke asli)
        cv::Mat display = frame.clone();

        if (pose_poller.QueueSize() > 0 && pose_poller.Next(&pose_pkt)) {
            const auto& pose = pose_pkt.Get<mediapipe::NormalizedLandmarkList>();
            DrawLandmarks(display, pose, cv::Scalar(66, 117, 245), 5);
        }

        if (lhand_poller.QueueSize() > 0 && lhand_poller.Next(&lhand_pkt)) {
            const auto& lhand = lhand_pkt.Get<mediapipe::NormalizedLandmarkList>();
            DrawLandmarks(display, lhand, cv::Scalar(76, 22, 121), 4);
        }

        if (rhand_poller.QueueSize() > 0 && rhand_poller.Next(&rhand_pkt)) {
            const auto& rhand = rhand_pkt.Get<mediapipe::NormalizedLandmarkList>();
            DrawLandmarks(display, rhand, cv::Scalar(10, 22, 80), 4);
        }

        // Hitung & tampilkan FPS
        auto curr_time = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(
            curr_time - prev_time).count();
        double fps = (elapsed_ms > 0) ? (1000.0 / elapsed_ms) : 0;
        prev_time = curr_time;

        std::string label = "FPS: " + std::to_string(static_cast<int>(fps))
                          + " | C++ Mediapipe";
        cv::putText(display, label, {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 255, 0), 2);

        cv::imshow("Mediapipe C++ - Holistic", display);

        if (cv::waitKey(1) == 'q') break;
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream("input_video"));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    absl::Status status = RunGraph();
    if (!status.ok()) {
        std::cerr << "[ERROR] " << status.message() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
