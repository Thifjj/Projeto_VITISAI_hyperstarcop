// ============================================================================
// HyperSTARCOP - ZCU104 optimized benchmark
// Vitis AI 3.5 / VART / DPUCZDX8G
//
// BATCH IS ALWAYS 1.
//
// Profiles:
//   baseline:
//       - sequential batch=1
//       - 1 runner
//       - 1 inference in flight
//       - reports baseline_model_only + baseline_end_to_end
//
//   max-model-only:
//       - inputs already preprocessed/quantized before timing
//       - N independent VART runners, one host thread per runner
//       - batch=1 for every inference
//       - measures sustained throughput = completed / wall time
//
//   max-e2e:
//       - threaded pipeline:
//             PRE workers -> per-runner DPU queues -> POST workers
//       - multiple independent aligned slots/buffers
//       - disk TIFF I/O INCLUDED
//       - normalization + INT8 quantization INCLUDED
//       - VART sync + DPU INCLUDED
//       - dequantization + sigmoid + threshold INCLUDED
//       - labels/metrics/CSV NOT INCLUDED in timed pipeline
//
// Accuracy validation:
//   TP, FP, FN, TN, Precision, Recall, F1, IoU, Accuracy
//
// Statistical output:
//   throughput FPS (wall)
//   latency mean / median / min / max / p90 / p95 / p99
//   stddev / CV / p99-p50 jitter
//   inter-completion interval mean/p95/p99
//
// Typical build command is provided in build_hyperstarcop_zcu104_optimized.sh.
//
// ============================================================================

#include <vart/runner.hpp>
#include <vart/tensor_buffer.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static constexpr int kBatch = 1;
static constexpr int kH = 512;
static constexpr int kW = 512;
static constexpr int kC = 4;

static constexpr int kFallbackInputFix = 5;
static constexpr int kFallbackOutputFix = 2;

static constexpr uint64_t kDocTP = 40310;
static constexpr uint64_t kDocFP = 4847;
static constexpr uint64_t kDocFN = 3467;
static constexpr uint64_t kDocTN = 2310672;

// ============================================================================
// OPTIONS
// ============================================================================

struct Options {
  std::string profile = "all";  // all|baseline|max-model-only|max-e2e

  std::string model = "/home/root/hyperstarcop.xmodel";
  std::string dataset = "/home/root/STARCOP_mini";
  std::string csv = "/home/root/STARCOP_mini/test_mini10.csv";
  std::string out = "/home/root/hyperstarcop_optimized_results";

  int runners = 2;
  int pre_workers = 2;
  int post_workers = 1;
  int slots_per_runner = 3;

  int iterations = 500;
  int warmup = 20;
  int baseline_repeats = 100;
  int baseline_e2e_passes = 5;

  bool pin = false;
  bool validate = true;
};

static void usage(const char* argv0) {
  std::cout
      << "HyperSTARCOP ZCU104 optimized benchmark - batch=1 always\n\n"
      << "Usage:\n"
      << "  " << argv0 << " --profile all|baseline|max-model-only|max-e2e [options]\n\n"
      << "Profile aliases:\n"
      << "  --baseline\n"
      << "  --maxthroughputmodelonly\n"
      << "  --maxendend\n\n"
      << "Core:\n"
      << "  --model PATH\n"
      << "  --dataset DIR\n"
      << "  --csv PATH\n"
      << "  --out DIR\n\n"
      << "Concurrency:\n"
      << "  --runners N\n"
      << "  --pre-workers N\n"
      << "  --post-workers N\n"
      << "  --slots-per-runner N\n"
      << "  --pin | --no-pin\n\n"
      << "Benchmark:\n"
      << "  --iterations N\n"
      << "  --warmup N\n"
      << "  --baseline-repeats N\n"
      << "  --baseline-e2e-passes N\n"
      << "  --validate | --no-validate\n";
}

static Options parse_options(int argc, char** argv) {
  Options o;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];

    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after " + a);
      }
      return argv[++i];
    };

    if (a == "--profile") o.profile = value();
    else if (a == "--baseline") o.profile = "baseline";
    else if (a == "--maxthroughputmodelonly") o.profile = "max-model-only";
    else if (a == "--maxendend") o.profile = "max-e2e";

    else if (a == "--model") o.model = value();
    else if (a == "--dataset") o.dataset = value();
    else if (a == "--csv") o.csv = value();
    else if (a == "--out") o.out = value();

    else if (a == "--runners") o.runners = std::stoi(value());
    else if (a == "--pre-workers") o.pre_workers = std::stoi(value());
    else if (a == "--post-workers") o.post_workers = std::stoi(value());
    else if (a == "--slots-per-runner") o.slots_per_runner = std::stoi(value());

    else if (a == "--iterations") o.iterations = std::stoi(value());
    else if (a == "--warmup") o.warmup = std::stoi(value());
    else if (a == "--baseline-repeats") o.baseline_repeats = std::stoi(value());
    else if (a == "--baseline-e2e-passes") o.baseline_e2e_passes = std::stoi(value());

    else if (a == "--pin") o.pin = true;
    else if (a == "--no-pin") o.pin = false;

    else if (a == "--validate") o.validate = true;
    else if (a == "--no-validate") o.validate = false;

    else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }

  if (o.profile != "all" &&
      o.profile != "baseline" &&
      o.profile != "max-model-only" &&
      o.profile != "max-e2e") {
    throw std::runtime_error(
        "--profile must be all, baseline, max-model-only or max-e2e");
  }

  if (o.runners <= 0 ||
      o.pre_workers <= 0 ||
      o.post_workers <= 0 ||
      o.slots_per_runner <= 0 ||
      o.iterations <= 0 ||
      o.warmup < 0 ||
      o.baseline_repeats <= 0 ||
      o.baseline_e2e_passes <= 0) {
    throw std::runtime_error("Invalid numeric option");
  }

  return o;
}

// ============================================================================
// HELPERS
// ============================================================================

static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  return a.back() == '/' ? a + b : a + "/" + b;
}

static void ensure_dir(const fs::path& p) {
  fs::create_directories(p);
}

static unsigned cpu_count() {
  unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 1u : n;
}

static void pin_current_thread(int logical_slot) {
  const unsigned n = cpu_count();
  const int core = logical_slot % static_cast<int>(n);

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);

  int rc = pthread_setaffinity_np(
      pthread_self(),
      sizeof(set),
      &set);

  if (rc != 0) {
    std::cerr
        << "WARN: pthread_setaffinity_np(core="
        << core
        << ") failed rc="
        << rc
        << "\n";
  }
}

// ============================================================================
// CSV DATASET
// ============================================================================

struct DatasetItem {
  std::string id;
  std::string has_plume;
};

static std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::string cur;
  bool quoted = false;

  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];

    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        cur += '"';
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(cur);
      cur.clear();
    } else {
      cur += ch;
    }
  }

  fields.push_back(cur);
  return fields;
}

static std::vector<DatasetItem> load_dataset_csv(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open CSV: " + path);

  std::string line;
  if (!std::getline(f, line)) {
    throw std::runtime_error("Empty CSV: " + path);
  }

  auto header = parse_csv_line(line);

  int id_col = -1;
  int plume_col = -1;

  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == "id") id_col = static_cast<int>(i);
    if (header[i] == "has_plume") plume_col = static_cast<int>(i);
  }

  if (id_col < 0) {
    throw std::runtime_error("CSV has no id column");
  }

  std::vector<DatasetItem> out;

  while (std::getline(f, line)) {
    if (line.empty()) continue;

    auto fields = parse_csv_line(line);

    if (static_cast<size_t>(id_col) >= fields.size()) continue;

    DatasetItem item;
    item.id = fields[id_col];

    if (plume_col >= 0 &&
        static_cast<size_t>(plume_col) < fields.size()) {
      item.has_plume = fields[plume_col];
    }

    out.push_back(std::move(item));
  }

  if (out.empty()) {
    throw std::runtime_error("No samples in CSV");
  }

  return out;
}

// ============================================================================
// ALIGNED STORAGE / TENSOR BUFFER
// ============================================================================

class AlignedStorage {
 public:
  AlignedStorage() = default;

  explicit AlignedStorage(size_t bytes) {
    allocate(bytes);
  }

  AlignedStorage(const AlignedStorage&) = delete;
  AlignedStorage& operator=(const AlignedStorage&) = delete;

  AlignedStorage(AlignedStorage&&) noexcept = default;
  AlignedStorage& operator=(AlignedStorage&&) noexcept = default;

  void allocate(size_t bytes) {
    bytes_ = bytes;

    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes_) != 0 || p == nullptr) {
      throw std::bad_alloc();
    }

    ptr_.reset(static_cast<uint8_t*>(p));
    std::memset(ptr_.get(), 0, bytes_);
  }

  uint8_t* data() { return ptr_.get(); }
  const uint8_t* data() const { return ptr_.get(); }
  size_t size() const { return bytes_; }

 private:
  struct Free {
    void operator()(uint8_t* p) const {
      std::free(p);
    }
  };

  std::unique_ptr<uint8_t, Free> ptr_;
  size_t bytes_ = 0;
};

class CpuFlatTensorBuffer final : public vart::TensorBuffer {
 public:
  CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
      : vart::TensorBuffer(tensor),
        data_(static_cast<uint8_t*>(data)) {}

  std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override {
    const auto shape = tensor_->get_shape();

    size_t offset = 0;

    if (!idx.empty()) {
      if (idx.size() != shape.size()) {
        throw std::runtime_error("TensorBuffer rank mismatch");
      }

      for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= shape[i]) {
          throw std::runtime_error("TensorBuffer index out of range");
        }

        offset =
            offset * static_cast<size_t>(shape[i])
            + static_cast<size_t>(idx[i]);
      }
    }

    const size_t total_bytes = tensor_->get_data_size();
    const size_t elements =
        static_cast<size_t>(tensor_->get_element_num());

    const size_t element_bytes =
        total_bytes / elements;

    const size_t byte_offset =
        offset * element_bytes;

    return {
        reinterpret_cast<uint64_t>(data_ + byte_offset),
        total_bytes - byte_offset
    };
  }

 private:
  uint8_t* data_;
};

struct OwnedTensorBuffer {
  explicit OwnedTensorBuffer(const xir::Tensor* tensor)
      : storage(tensor->get_data_size()),
        buffer(storage.data(), tensor) {}

  AlignedStorage storage;
  CpuFlatTensorBuffer buffer;
};

// ============================================================================
// MODEL / RUNNER
// ============================================================================

static const xir::Subgraph* find_dpu_subgraph(const xir::Subgraph* root) {
  if (root->has_attr("device") &&
      root->get_attr<std::string>("device") == "DPU") {
    return root;
  }

  for (auto* child : root->children_topological_sort()) {
    if (auto* found = find_dpu_subgraph(child)) {
      return found;
    }
  }

  return nullptr;
}

struct ModelContext {
  explicit ModelContext(const std::string& path) {
    graph = xir::Graph::deserialize(path);

    if (!graph) {
      throw std::runtime_error(
          "Failed to deserialize XMODEL: " + path);
    }

    dpu = find_dpu_subgraph(
        graph->get_root_subgraph());

    if (!dpu) {
      throw std::runtime_error("No DPU subgraph found");
    }
  }

  std::unique_ptr<xir::Graph> graph;
  const xir::Subgraph* dpu = nullptr;
};

static int get_fix_point(const xir::Tensor* tensor, int fallback) {
  try {
    if (tensor->has_attr("fix_point")) {
      return tensor->get_attr<int>("fix_point");
    }
  } catch (...) {}

  try {
    if (tensor->has_attr("fixpos")) {
      return tensor->get_attr<int>("fixpos");
    }
  } catch (...) {}

  return fallback;
}

struct StageTimes {
  size_t job = 0;
  size_t dataset_index = 0;
  int lane = -1;
  int pre_worker = -1;
  int post_worker = -1;

  double slot_wait_ms = 0.0;
  double io_ms = 0.0;
  double preprocess_ms = 0.0;
  double pre_dpu_queue_ms = 0.0;
  double input_sync_ms = 0.0;
  double dpu_ms = 0.0;
  double output_sync_ms = 0.0;
  double dpu_post_queue_ms = 0.0;
  double postprocess_ms = 0.0;
  double e2e_ms = 0.0;
  double completion_s = 0.0;
};

class FrameSlot {
 public:
  FrameSlot(
      int lane_id,
      int slot_id,
      const xir::Tensor* input_tensor,
      const xir::Tensor* output_tensor)
      : lane(lane_id),
        id(slot_id),
        input(std::make_unique<OwnedTensorBuffer>(input_tensor)),
        output(std::make_unique<OwnedTensorBuffer>(output_tensor)),
        mask(static_cast<size_t>(kH * kW), 0) {
    input_ptrs.push_back(&input->buffer);
    output_ptrs.push_back(&output->buffer);
  }

  int8_t* input_data() {
    return reinterpret_cast<int8_t*>(
        input->storage.data());
  }

  int8_t* output_data() {
    return reinterpret_cast<int8_t*>(
        output->storage.data());
  }

  int lane = 0;
  int id = 0;

  size_t job = 0;
  size_t dataset_index = 0;

  Clock::time_point arrival{};
  Clock::time_point queued_dpu{};
  Clock::time_point queued_post{};

  std::unique_ptr<OwnedTensorBuffer> input;
  std::unique_ptr<OwnedTensorBuffer> output;

  std::vector<vart::TensorBuffer*> input_ptrs;
  std::vector<vart::TensorBuffer*> output_ptrs;

  std::vector<uint8_t> mask;
};

class DpuLane {
 public:
  DpuLane(
      int lane_id,
      const xir::Subgraph* dpu,
      int slots)
      : id_(lane_id) {
    runner_ = vart::Runner::create_runner(
        dpu,
        "run");

    if (!runner_) {
      throw std::runtime_error("Failed to create VART runner");
    }

    auto inputs = runner_->get_input_tensors();
    auto outputs = runner_->get_output_tensors();

    if (inputs.size() != 1 || outputs.size() != 1) {
      throw std::runtime_error(
          "HyperSTARCOP expects exactly 1 input and 1 output");
    }

    input_tensor_ = inputs[0];
    output_tensor_ = outputs[0];

    validate_tensors();

    input_fix_ = get_fix_point(
        input_tensor_,
        kFallbackInputFix);

    output_fix_ = get_fix_point(
        output_tensor_,
        kFallbackOutputFix);

    input_scale_ =
        std::exp2(static_cast<float>(input_fix_));

    output_scale_ =
        std::exp2(-static_cast<float>(output_fix_));

    slots_.reserve(static_cast<size_t>(slots));

    for (int i = 0; i < slots; ++i) {
      slots_.push_back(
          std::make_unique<FrameSlot>(
              id_,
              i,
              input_tensor_,
              output_tensor_));
    }
  }

  int id() const { return id_; }
  float input_scale() const { return input_scale_; }
  float output_scale() const { return output_scale_; }

  size_t input_bytes() const {
    return input_tensor_->get_data_size();
  }

  size_t output_bytes() const {
    return output_tensor_->get_data_size();
  }

  const std::vector<std::unique_ptr<FrameSlot>>& slots() const {
    return slots_;
  }

  void print_metadata() const {
    auto in = input_tensor_->get_shape();
    auto out = output_tensor_->get_shape();

    std::cout
        << "Input  : ["
        << in[0] << ","
        << in[1] << ","
        << in[2] << ","
        << in[3] << "] INT8 fix="
        << input_fix_
        << "\n";

    std::cout
        << "Output : ["
        << out[0] << ","
        << out[1] << ","
        << out[2] << ","
        << out[3] << "] INT8 fix="
        << output_fix_
        << "\n";
  }

  void run_full(FrameSlot& slot, StageTimes& t) {
    auto s0 = Clock::now();

    slot.input->buffer.sync_for_write(
        0,
        slot.input->storage.size());

    auto s1 = Clock::now();

    auto job = runner_->execute_async(
        slot.input_ptrs,
        slot.output_ptrs);

    if (job.second != 0) {
      throw std::runtime_error(
          "execute_async returned non-zero status");
    }

    int status = runner_->wait(
        static_cast<int>(job.first),
        -1);

    auto s2 = Clock::now();

    if (status != 0) {
      throw std::runtime_error(
          "VART wait failed status=" +
          std::to_string(status));
    }

    slot.output->buffer.sync_for_read(
        0,
        slot.output->storage.size());

    auto s3 = Clock::now();

    t.input_sync_ms += elapsed_ms(s0, s1);
    t.dpu_ms += elapsed_ms(s1, s2);
    t.output_sync_ms += elapsed_ms(s2, s3);
  }

  double run_model_only(FrameSlot& slot) {
    auto t0 = Clock::now();

    auto job = runner_->execute_async(
        slot.input_ptrs,
        slot.output_ptrs);

    if (job.second != 0) {
      throw std::runtime_error(
          "execute_async returned non-zero status");
    }

    int status = runner_->wait(
        static_cast<int>(job.first),
        -1);

    auto t1 = Clock::now();

    if (status != 0) {
      throw std::runtime_error(
          "VART wait failed status=" +
          std::to_string(status));
    }

    return elapsed_ms(t0, t1);
  }

 private:
  void validate_tensors() {
    const auto in = input_tensor_->get_shape();
    const auto out = output_tensor_->get_shape();

    if (in != std::vector<int32_t>({1, kH, kW, kC})) {
      throw std::runtime_error(
          "Expected input [1,512,512,4]");
    }

    if (out != std::vector<int32_t>({1, kH, kW, 1})) {
      throw std::runtime_error(
          "Expected output [1,512,512,1]");
    }

    if (input_tensor_->get_data_size() !=
        static_cast<size_t>(kH * kW * kC)) {
      throw std::runtime_error(
          "Expected byte-sized INT8 input");
    }

    if (output_tensor_->get_data_size() !=
        static_cast<size_t>(kH * kW)) {
      throw std::runtime_error(
          "Expected byte-sized INT8 output");
    }
  }

  int id_ = 0;

  std::unique_ptr<vart::Runner> runner_;

  const xir::Tensor* input_tensor_ = nullptr;
  const xir::Tensor* output_tensor_ = nullptr;

  int input_fix_ = kFallbackInputFix;
  int output_fix_ = kFallbackOutputFix;

  float input_scale_ = 32.0f;
  float output_scale_ = 0.25f;

  std::vector<std::unique_ptr<FrameSlot>> slots_;
};

// ============================================================================
// TIFF / PREPROCESS
// ============================================================================

struct PreprocessWorkspace {
  cv::Mat mag1c;
  cv::Mat red;
  cv::Mat green;
  cv::Mat blue;
};

static cv::Mat read_tif_float(const fs::path& path) {
  cv::Mat img = cv::imread(
      path.string(),
      cv::IMREAD_UNCHANGED);

  if (img.empty()) {
    throw std::runtime_error(
        "Cannot read TIFF: " + path.string());
  }

  if (img.channels() != 1) {
    throw std::runtime_error(
        "Expected one-channel TIFF: " + path.string());
  }

  cv::Mat out;
  img.convertTo(out, CV_32F);

  if (out.rows != kH || out.cols != kW) {
    std::ostringstream oss;
    oss
        << "Expected "
        << kW
        << "x"
        << kH
        << ", got "
        << out.cols
        << "x"
        << out.rows
        << " for "
        << path;

    throw std::runtime_error(oss.str());
  }

  return out;
}

static inline float clip02(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 2.0f) return 2.0f;
  return x;
}

static void preprocess_into(
    const fs::path& folder,
    float input_scale,
    int8_t* dst,
    PreprocessWorkspace& ws,
    StageTimes& t) {
  auto io0 = Clock::now();

  ws.mag1c = read_tif_float(
      folder / "mag1c.tif");

  ws.red = read_tif_float(
      folder / "TOA_AVIRIS_640nm.tif");

  ws.green = read_tif_float(
      folder / "TOA_AVIRIS_550nm.tif");

  ws.blue = read_tif_float(
      folder / "TOA_AVIRIS_460nm.tif");

  auto io1 = Clock::now();

  t.io_ms += elapsed_ms(io0, io1);

  auto p0 = Clock::now();

  for (int y = 0; y < kH; ++y) {
    const float* m = ws.mag1c.ptr<float>(y);
    const float* r = ws.red.ptr<float>(y);
    const float* g = ws.green.ptr<float>(y);
    const float* b = ws.blue.ptr<float>(y);

    int8_t* out =
        dst
        + static_cast<size_t>(y)
          * kW
          * kC;

    for (int x = 0; x < kW; ++x) {
      const float values[4] = {
          clip02(m[x] / 1750.0f),
          clip02(r[x] / 60.0f),
          clip02(g[x] / 60.0f),
          clip02(b[x] / 60.0f)
      };

      for (int c = 0; c < 4; ++c) {
        int q = static_cast<int>(
            std::lrint(values[c] * input_scale));

        q = std::max(
            -128,
            std::min(127, q));

        out[x * 4 + c] =
            static_cast<int8_t>(q);
      }
    }
  }

  auto p1 = Clock::now();

  t.preprocess_ms +=
      t.io_ms
      + elapsed_ms(p0, p1);
}

static cv::Mat load_label(const fs::path& folder) {
  cv::Mat f =
      read_tif_float(
          folder / "labelbinary.tif");

  cv::Mat label(
      kH,
      kW,
      CV_8U);

  for (int y = 0; y < kH; ++y) {
    const float* src = f.ptr<float>(y);
    uint8_t* dst = label.ptr<uint8_t>(y);

    for (int x = 0; x < kW; ++x) {
      dst[x] = src[x] > 0.0f ? 1 : 0;
    }
  }

  return label;
}

// ============================================================================
// POSTPROCESS
// ============================================================================

static void postprocess_into_mask(
    const int8_t* output,
    float output_scale,
    uint8_t* mask) {
  for (size_t i = 0;
       i < static_cast<size_t>(kH * kW);
       ++i) {
    const float logit =
        static_cast<float>(output[i])
        * output_scale;

    const float z = std::max(
        -80.0f,
        std::min(80.0f, logit));

    const float prob =
        1.0f /
        (1.0f + std::exp(-z));

    mask[i] =
        prob > 0.5f
        ? 1
        : 0;
  }
}

// ============================================================================
// METRICS
// ============================================================================

struct Counts {
  uint64_t tp = 0;
  uint64_t fp = 0;
  uint64_t fn = 0;
  uint64_t tn = 0;
};

struct Metrics {
  double precision = 0.0;
  double recall = 0.0;
  double f1 = 0.0;
  double iou = 0.0;
  double accuracy = 0.0;
};

static Counts confusion(
    const uint8_t* pred,
    const cv::Mat& label) {
  Counts c;

  for (int y = 0; y < kH; ++y) {
    const uint8_t* l =
        label.ptr<uint8_t>(y);

    for (int x = 0; x < kW; ++x) {
      const bool p =
          pred[
              static_cast<size_t>(y)
              * kW
              + x]
          != 0;

      const bool g =
          l[x] != 0;

      if (p && g) ++c.tp;
      else if (p && !g) ++c.fp;
      else if (!p && g) ++c.fn;
      else ++c.tn;
    }
  }

  return c;
}

static Metrics calc_metrics(const Counts& c) {
  Metrics m;

  if (c.tp + c.fp) {
    m.precision =
        static_cast<double>(c.tp)
        /
        static_cast<double>(c.tp + c.fp);
  }

  if (c.tp + c.fn) {
    m.recall =
        static_cast<double>(c.tp)
        /
        static_cast<double>(c.tp + c.fn);
  }

  if (m.precision + m.recall > 0.0) {
    m.f1 =
        2.0
        * m.precision
        * m.recall
        /
        (m.precision + m.recall);
  }

  if (c.tp + c.fp + c.fn) {
    m.iou =
        static_cast<double>(c.tp)
        /
        static_cast<double>(
            c.tp + c.fp + c.fn);
  }

  const uint64_t total =
      c.tp + c.fp + c.fn + c.tn;

  if (total) {
    m.accuracy =
        static_cast<double>(c.tp + c.tn)
        /
        static_cast<double>(total);
  }

  return m;
}

// ============================================================================
// STATS
// ============================================================================

struct Stats {
  size_t count = 0;

  double mean = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double min = 0.0;
  double max = 0.0;
  double stddev = 0.0;
  double cv = 0.0;
};

static Stats summarize(std::vector<double> v) {
  Stats s;

  if (v.empty()) return s;

  std::sort(v.begin(), v.end());

  s.count = v.size();
  s.min = v.front();
  s.max = v.back();

  s.mean =
      std::accumulate(
          v.begin(),
          v.end(),
          0.0)
      /
      static_cast<double>(v.size());

  auto q = [&](double p) {
    const double x =
        p
        * static_cast<double>(
            v.size() - 1);

    const size_t i =
        static_cast<size_t>(x);

    const double f =
        x
        - static_cast<double>(i);

    return
        v[i] * (1.0 - f)
        +
        v[
            std::min(
                i + 1,
                v.size() - 1)]
        * f;
  };

  s.median = q(0.50);
  s.p90 = q(0.90);
  s.p95 = q(0.95);
  s.p99 = q(0.99);

  double ss = 0.0;

  for (double x : v) {
    const double d = x - s.mean;
    ss += d * d;
  }

  s.stddev =
      std::sqrt(
          ss
          /
          static_cast<double>(v.size()));

  s.cv =
      s.mean != 0.0
      ? s.stddev / s.mean
      : 0.0;

  return s;
}

struct BenchmarkResult {
  std::string mode;

  int batch = 1;
  int runners = 1;
  int pre_workers = 0;
  int post_workers = 0;
  int slots_per_runner = 1;

  size_t completed = 0;

  double wall_s = 0.0;
  double throughput_fps = 0.0;

  Stats latency;
  Stats dpu;
  Stats preprocess;
  Stats postprocess;
  Stats inter_completion;

  std::vector<StageTimes> samples;
};

// ============================================================================
// BOUNDED QUEUE / START GATE
// ============================================================================

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity)
      : buf_(capacity),
        capacity_(capacity) {
    if (capacity_ == 0) {
      throw std::runtime_error(
          "Queue capacity cannot be zero");
    }
  }

  void push(T v) {
    std::unique_lock<std::mutex> lock(mu_);

    not_full_.wait(
        lock,
        [&] {
          return count_ < capacity_;
        });

    buf_[tail_] = std::move(v);

    tail_ =
        (tail_ + 1)
        % capacity_;

    ++count_;

    lock.unlock();
    not_empty_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mu_);

    not_empty_.wait(
        lock,
        [&] {
          return count_ > 0;
        });

    T v =
        std::move(buf_[head_]);

    head_ =
        (head_ + 1)
        % capacity_;

    --count_;

    lock.unlock();
    not_full_.notify_one();

    return v;
  }

 private:
  std::vector<T> buf_;

  size_t capacity_ = 0;
  size_t head_ = 0;
  size_t tail_ = 0;
  size_t count_ = 0;

  std::mutex mu_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
};

class StartGate {
 public:
  explicit StartGate(int expected)
      : expected_(expected) {}

  void worker_ready_and_wait() {
    std::unique_lock<std::mutex> lock(mu_);

    ++ready_;
    cv_.notify_all();

    cv_.wait(
        lock,
        [&] {
          return released_;
        });
  }

  void wait_ready() {
    std::unique_lock<std::mutex> lock(mu_);

    cv_.wait(
        lock,
        [&] {
          return ready_ == expected_;
        });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mu_);

    released_ = true;
    cv_.notify_all();
  }

 private:
  int expected_ = 0;
  int ready_ = 0;

  bool released_ = false;

  std::mutex mu_;
  std::condition_variable cv_;
};

// ============================================================================
// CREATE LANES
// ============================================================================

static std::vector<std::unique_ptr<DpuLane>>
make_lanes(
    const Options& o,
    const ModelContext& model,
    int runners,
    int slots_per_runner) {
  std::vector<std::unique_ptr<DpuLane>> lanes;

  lanes.reserve(
      static_cast<size_t>(runners));

  for (int i = 0; i < runners; ++i) {
    lanes.push_back(
        std::make_unique<DpuLane>(
            i,
            model.dpu,
            slots_per_runner));
  }

  return lanes;
}

// ============================================================================
// PREPARE INPUTS
// ============================================================================

static std::vector<std::vector<int8_t>>
prepare_inputs(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    float input_scale) {
  std::vector<std::vector<int8_t>> cache;

  cache.reserve(dataset.size());

  PreprocessWorkspace ws;

  for (size_t i = 0; i < dataset.size(); ++i) {
    std::vector<int8_t> input(
        static_cast<size_t>(kH * kW * kC));

    StageTimes t;

    preprocess_into(
        fs::path(o.dataset) / dataset[i].id,
        input_scale,
        input.data(),
        ws,
        t);

    cache.push_back(
        std::move(input));
  }

  return cache;
}

// ============================================================================
// VALIDATION
// ============================================================================

static void run_validation(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    ModelContext& model) {
  auto lanes =
      make_lanes(
          o,
          model,
          1,
          1);

  auto& lane =
      *lanes.front();

  auto& slot =
      *lane.slots().front();

  lane.print_metadata();

  Counts global;

  std::ofstream per_image(
      fs::path(o.out)
      / "metricas_por_imagem.csv");

  per_image
      << "id,has_plume,TP,FP,FN,TN,"
      << "precision,recall,f1,iou,accuracy\n";

  PreprocessWorkspace ws;

  for (size_t i = 0; i < dataset.size(); ++i) {
    const auto& item = dataset[i];

    const fs::path folder =
        fs::path(o.dataset)
        / item.id;

    StageTimes t;

    preprocess_into(
        folder,
        lane.input_scale(),
        slot.input_data(),
        ws,
        t);

    lane.run_full(
        slot,
        t);

    postprocess_into_mask(
        slot.output_data(),
        lane.output_scale(),
        slot.mask.data());

    cv::Mat label =
        load_label(folder);

    Counts c =
        confusion(
            slot.mask.data(),
            label);

    Metrics m =
        calc_metrics(c);

    global.tp += c.tp;
    global.fp += c.fp;
    global.fn += c.fn;
    global.tn += c.tn;

    per_image
        << item.id << ","
        << item.has_plume << ","
        << c.tp << ","
        << c.fp << ","
        << c.fn << ","
        << c.tn << ","
        << std::setprecision(12)
        << m.precision << ","
        << m.recall << ","
        << m.f1 << ","
        << m.iou << ","
        << m.accuracy
        << "\n";

    std::cout
        << "[VAL "
        << (i + 1)
        << "/"
        << dataset.size()
        << "] "
        << item.id
        << " F1="
        << std::fixed
        << std::setprecision(4)
        << m.f1
        << " IoU="
        << m.iou
        << "\n";
  }

  Metrics gm =
      calc_metrics(global);

  std::ofstream g(
      fs::path(o.out)
      / "metricas_globais.csv");

  g
      << "num_imagens,TP,FP,FN,TN,"
      << "precision_global,recall_global,"
      << "f1_global,iou_global,accuracy_global\n";

  g
      << dataset.size() << ","
      << global.tp << ","
      << global.fp << ","
      << global.fn << ","
      << global.tn << ","
      << std::setprecision(12)
      << gm.precision << ","
      << gm.recall << ","
      << gm.f1 << ","
      << gm.iou << ","
      << gm.accuracy
      << "\n";

  Counts doc;
  doc.tp = kDocTP;
  doc.fp = kDocFP;
  doc.fn = kDocFN;
  doc.tn = kDocTN;

  Metrics dm =
      calc_metrics(doc);

  std::ofstream c(
      fs::path(o.out)
      / "comparacao_documentacao.csv");

  c
      << "metrica,documentacao,zcu104,delta_abs,delta_pct\n";

  auto write_metric =
      [&](const std::string& name,
          double d,
          double z) {
        c
            << name << ","
            << std::setprecision(12)
            << d << ","
            << z << ","
            << (z - d) << ","
            << (
                d != 0.0
                ? (z - d) / d * 100.0
                : 0.0)
            << "\n";
      };

  write_metric(
      "precision",
      dm.precision,
      gm.precision);

  write_metric(
      "recall",
      dm.recall,
      gm.recall);

  write_metric(
      "f1",
      dm.f1,
      gm.f1);

  write_metric(
      "iou",
      dm.iou,
      gm.iou);

  write_metric(
      "accuracy",
      dm.accuracy,
      gm.accuracy);

  std::cout
      << "\nVALIDATION GLOBAL\n"
      << "Precision="
      << gm.precision
      << " Recall="
      << gm.recall
      << " F1="
      << gm.f1
      << " IoU="
      << gm.iou
      << " Accuracy="
      << gm.accuracy
      << "\n";
}

// ============================================================================
// BUILD RESULT STATS
// ============================================================================

static BenchmarkResult finalize_result(
    BenchmarkResult r) {
  std::vector<double> lat;
  std::vector<double> dpu;
  std::vector<double> prep;
  std::vector<double> post;
  std::vector<double> completion;

  for (const auto& t : r.samples) {
    lat.push_back(t.e2e_ms);

    if (t.dpu_ms > 0.0) {
      dpu.push_back(t.dpu_ms);
    }

    if (t.preprocess_ms > 0.0) {
      prep.push_back(t.preprocess_ms);
    }

    if (t.postprocess_ms > 0.0) {
      post.push_back(t.postprocess_ms);
    }

    completion.push_back(t.completion_s);
  }

  r.latency = summarize(lat);
  r.dpu = summarize(dpu);
  r.preprocess = summarize(prep);
  r.postprocess = summarize(post);

  std::sort(
      completion.begin(),
      completion.end());

  std::vector<double> intervals;

  if (completion.size() > 1) {
    intervals.reserve(
        completion.size() - 1);

    for (size_t i = 1;
         i < completion.size();
         ++i) {
      intervals.push_back(
          (completion[i] - completion[i - 1])
          * 1000.0);
    }
  }

  r.inter_completion =
      summarize(intervals);

  return r;
}

// ============================================================================
// BASELINE BATCH1
// ============================================================================

static std::vector<BenchmarkResult>
run_baseline(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    ModelContext& model) {
  auto lanes =
      make_lanes(
          o,
          model,
          1,
          1);

  auto& lane =
      *lanes.front();

  auto& slot =
      *lane.slots().front();

  auto cache =
      prepare_inputs(
          o,
          dataset,
          lane.input_scale());

  // ------------------------------------------------------------------------
  // baseline model-only:
  // only execute_async + wait timed
  // ------------------------------------------------------------------------

  // Resident prepared input. Model-only timing contains no preprocessing,
  // memcpy, quantization or buffer sync.
  const auto& baseline_src = cache.front();

  std::memcpy(
      slot.input_data(),
      baseline_src.data(),
      baseline_src.size());

  slot.input->buffer.sync_for_write(
      0,
      slot.input->storage.size());

  for (int i = 0; i < o.warmup; ++i) {
    (void)lane.run_model_only(slot);
  }

  BenchmarkResult mo;
  mo.mode = "baseline_model_only";
  mo.batch = 1;
  mo.runners = 1;
  mo.slots_per_runner = 1;
  mo.samples.resize(
      static_cast<size_t>(
          o.baseline_repeats));

  auto wall0 = Clock::now();

  for (int i = 0;
       i < o.baseline_repeats;
       ++i) {
    StageTimes& t =
        mo.samples[
            static_cast<size_t>(i)];

    t.job = static_cast<size_t>(i);
    t.dataset_index =
        static_cast<size_t>(i)
        % dataset.size();

    const double dpu_ms =
        lane.run_model_only(slot);

    t.dpu_ms = dpu_ms;
    t.e2e_ms = dpu_ms;

    t.completion_s =
        std::chrono::duration<double>(
            Clock::now() - wall0)
        .count();
  }

  auto wall1 = Clock::now();

  mo.wall_s =
      std::chrono::duration<double>(
          wall1 - wall0)
      .count();

  mo.completed =
      mo.samples.size();

  mo.throughput_fps =
      static_cast<double>(mo.completed)
      /
      mo.wall_s;

  mo = finalize_result(
      std::move(mo));

  // ------------------------------------------------------------------------
  // baseline end-to-end:
  // TIFF -> preprocess -> DPU -> dequant/sigmoid/threshold -> mask
  // ------------------------------------------------------------------------

  PreprocessWorkspace ws;

  for (int pass = 0;
       pass < 1;
       ++pass) {
    for (size_t i = 0;
         i < dataset.size();
         ++i) {
      StageTimes t;

      preprocess_into(
          fs::path(o.dataset)
          / dataset[i].id,
          lane.input_scale(),
          slot.input_data(),
          ws,
          t);

      lane.run_full(
          slot,
          t);

      postprocess_into_mask(
          slot.output_data(),
          lane.output_scale(),
          slot.mask.data());
    }
  }

  BenchmarkResult e2e;
  e2e.mode = "baseline_end_to_end";
  e2e.batch = 1;
  e2e.runners = 1;
  e2e.pre_workers = 1;
  e2e.post_workers = 1;
  e2e.slots_per_runner = 1;

  const size_t total =
      static_cast<size_t>(
          o.baseline_e2e_passes)
      * dataset.size();

  e2e.samples.resize(total);

  auto e0 = Clock::now();

  size_t job_index = 0;

  for (int pass = 0;
       pass < o.baseline_e2e_passes;
       ++pass) {
    for (size_t di = 0;
         di < dataset.size();
         ++di) {
      StageTimes& t =
          e2e.samples[job_index];

      t.job = job_index;
      t.dataset_index = di;

      auto t0 = Clock::now();

      preprocess_into(
          fs::path(o.dataset)
          / dataset[di].id,
          lane.input_scale(),
          slot.input_data(),
          ws,
          t);

      lane.run_full(
          slot,
          t);

      auto p0 = Clock::now();

      postprocess_into_mask(
          slot.output_data(),
          lane.output_scale(),
          slot.mask.data());

      auto p1 = Clock::now();

      t.postprocess_ms =
          elapsed_ms(p0, p1);

      t.e2e_ms =
          elapsed_ms(t0, p1);

      t.completion_s =
          std::chrono::duration<double>(
              p1 - e0)
          .count();

      ++job_index;
    }
  }

  auto e1 = Clock::now();

  e2e.wall_s =
      std::chrono::duration<double>(
          e1 - e0)
      .count();

  e2e.completed =
      e2e.samples.size();

  e2e.throughput_fps =
      static_cast<double>(e2e.completed)
      /
      e2e.wall_s;

  e2e = finalize_result(
      std::move(e2e));

  return {
      std::move(mo),
      std::move(e2e)
  };
}

// ============================================================================
// MAX MODEL-ONLY THROUGHPUT
// ============================================================================

static BenchmarkResult
run_max_model_only(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    ModelContext& model) {
  auto lanes =
      make_lanes(
          o,
          model,
          o.runners,
          1);

  lanes.front()->print_metadata();

  auto cache =
      prepare_inputs(
          o,
          dataset,
          lanes.front()->input_scale());

  // Pre-fill one resident input per runner.
  for (int r = 0; r < o.runners; ++r) {
    auto& lane =
        *lanes[
            static_cast<size_t>(r)];

    auto& slot =
        *lane.slots().front();

    const auto& src =
        cache[
            static_cast<size_t>(r)
            % cache.size()];

    std::memcpy(
        slot.input_data(),
        src.data(),
        src.size());

    // unchanged resident input: sync once before benchmark
    slot.input->buffer.sync_for_write(
        0,
        slot.input->storage.size());

    for (int i = 0;
         i < o.warmup;
         ++i) {
      (void)lane.run_model_only(slot);
    }
  }

  BenchmarkResult result;
  result.mode = "max_model_only_throughput";
  result.batch = 1;
  result.runners = o.runners;
  result.slots_per_runner = 1;
  result.samples.resize(
      static_cast<size_t>(
          o.iterations));

  std::atomic<size_t> next{0};

  StartGate gate(
      o.runners);

  Clock::time_point run_start{};

  std::vector<std::thread> threads;
  threads.reserve(
      static_cast<size_t>(
          o.runners));

  for (int r = 0;
       r < o.runners;
       ++r) {
    threads.emplace_back(
        [&, r] {
          if (o.pin) {
            pin_current_thread(r);
          }

          auto& lane =
              *lanes[
                  static_cast<size_t>(r)];

          auto& slot =
              *lane.slots().front();

          gate.worker_ready_and_wait();

          for (;;) {
            const size_t j =
                next.fetch_add(1);

            if (j >=
                static_cast<size_t>(
                    o.iterations)) {
              break;
            }

            StageTimes& t =
                result.samples[j];

            t.job = j;
            t.dataset_index =
                j % dataset.size();
            t.lane = r;

            const double dpu_ms =
                lane.run_model_only(slot);

            t.dpu_ms = dpu_ms;
            t.e2e_ms = dpu_ms;

            t.completion_s =
                std::chrono::duration<double>(
                    Clock::now()
                    - run_start)
                .count();
          }
        });
  }

  gate.wait_ready();

  run_start =
      Clock::now();

  gate.release();

  for (auto& t : threads) {
    t.join();
  }

  auto run_end =
      Clock::now();

  result.wall_s =
      std::chrono::duration<double>(
          run_end - run_start)
      .count();

  result.completed =
      result.samples.size();

  result.throughput_fps =
      static_cast<double>(
          result.completed)
      /
      result.wall_s;

  return finalize_result(
      std::move(result));
}

// ============================================================================
// MAX END-TO-END PIPELINE
// ============================================================================

static BenchmarkResult
run_max_e2e(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    ModelContext& model) {
  auto lanes =
      make_lanes(
          o,
          model,
          o.runners,
          o.slots_per_runner);

  lanes.front()->print_metadata();

  // Warm each lane with first sample.
  for (int r = 0;
       r < o.runners;
       ++r) {
    auto& lane =
        *lanes[
            static_cast<size_t>(r)];

    auto& slot =
        *lane.slots().front();

    PreprocessWorkspace ws;
    StageTimes t;

    preprocess_into(
        fs::path(o.dataset)
        / dataset.front().id,
        lane.input_scale(),
        slot.input_data(),
        ws,
        t);

    for (int i = 0;
         i < o.warmup;
         ++i) {
      StageTimes w;
      lane.run_full(
          slot,
          w);
    }
  }

  BenchmarkResult result;
  result.mode = "max_end_to_end_throughput";
  result.batch = 1;
  result.runners = o.runners;
  result.pre_workers = o.pre_workers;
  result.post_workers = o.post_workers;
  result.slots_per_runner =
      o.slots_per_runner;
  result.samples.resize(
      static_cast<size_t>(
          o.iterations));

  const size_t total_slots =
      static_cast<size_t>(
          o.runners
          * o.slots_per_runner);

  BoundedQueue<FrameSlot*> free_q(
      total_slots);

  std::vector<
      std::unique_ptr<
          BoundedQueue<FrameSlot*>>>
      dpu_q;

  dpu_q.reserve(
      static_cast<size_t>(
          o.runners));

  for (int r = 0;
       r < o.runners;
       ++r) {
    dpu_q.push_back(
        std::make_unique<
            BoundedQueue<FrameSlot*>>(
                static_cast<size_t>(
                    o.slots_per_runner)));
  }

  BoundedQueue<FrameSlot*> post_q(
      total_slots);

  // Interleave lane slots.
  for (int s = 0;
       s < o.slots_per_runner;
       ++s) {
    for (int r = 0;
         r < o.runners;
         ++r) {
      free_q.push(
          lanes[
              static_cast<size_t>(r)]
          ->slots()[
              static_cast<size_t>(s)]
          .get());
    }
  }

  const int participants =
      o.pre_workers
      + o.runners
      + o.post_workers;

  StartGate gate(
      participants);

  std::atomic<size_t> next_job{0};

  Clock::time_point run_start{};

  std::vector<std::thread> pre_threads;
  std::vector<std::thread> dpu_threads;
  std::vector<std::thread> post_threads;

  // ------------------------------------------------------------------------
  // PRE
  // ------------------------------------------------------------------------

  for (int pw = 0;
       pw < o.pre_workers;
       ++pw) {
    pre_threads.emplace_back(
        [&, pw] {
          if (o.pin) {
            pin_current_thread(pw);
          }

          PreprocessWorkspace ws;

          gate.worker_ready_and_wait();

          for (;;) {
            const size_t j =
                next_job.fetch_add(1);

            if (j >=
                static_cast<size_t>(
                    o.iterations)) {
              break;
            }

            const size_t di =
                j % dataset.size();

            auto arrival =
                Clock::now();

            FrameSlot* slot =
                free_q.pop();

            auto got_slot =
                Clock::now();

            StageTimes& t =
                result.samples[j];

            t.job = j;
            t.dataset_index = di;
            t.lane = slot->lane;
            t.pre_worker = pw;
            t.slot_wait_ms =
                elapsed_ms(
                    arrival,
                    got_slot);

            slot->job = j;
            slot->dataset_index = di;
            slot->arrival = arrival;

            preprocess_into(
                fs::path(o.dataset)
                / dataset[di].id,
                lanes[
                    static_cast<size_t>(
                        slot->lane)]
                ->input_scale(),
                slot->input_data(),
                ws,
                t);

            slot->queued_dpu =
                Clock::now();

            dpu_q[
                static_cast<size_t>(
                    slot->lane)]
            ->push(slot);
          }
        });
  }

  // ------------------------------------------------------------------------
  // DPU
  // ------------------------------------------------------------------------

  for (int r = 0;
       r < o.runners;
       ++r) {
    dpu_threads.emplace_back(
        [&, r] {
          if (o.pin) {
            pin_current_thread(
                o.pre_workers
                + o.post_workers
                + r);
          }

          auto& lane =
              *lanes[
                  static_cast<size_t>(r)];

          gate.worker_ready_and_wait();

          for (;;) {
            FrameSlot* slot =
                dpu_q[
                    static_cast<size_t>(r)]
                ->pop();

            if (slot == nullptr) {
              break;
            }

            StageTimes& t =
                result.samples[
                    slot->job];

            t.pre_dpu_queue_ms =
                elapsed_ms(
                    slot->queued_dpu,
                    Clock::now());

            lane.run_full(
                *slot,
                t);

            slot->queued_post =
                Clock::now();

            post_q.push(slot);
          }
        });
  }

  // ------------------------------------------------------------------------
  // POST
  // ------------------------------------------------------------------------

  for (int pw = 0;
       pw < o.post_workers;
       ++pw) {
    post_threads.emplace_back(
        [&, pw] {
          if (o.pin) {
            pin_current_thread(
                o.pre_workers
                + pw);
          }

          gate.worker_ready_and_wait();

          for (;;) {
            FrameSlot* slot =
                post_q.pop();

            if (slot == nullptr) {
              break;
            }

            StageTimes& t =
                result.samples[
                    slot->job];

            t.post_worker = pw;

            t.dpu_post_queue_ms =
                elapsed_ms(
                    slot->queued_post,
                    Clock::now());

            auto p0 =
                Clock::now();

            postprocess_into_mask(
                slot->output_data(),
                lanes[
                    static_cast<size_t>(
                        slot->lane)]
                ->output_scale(),
                slot->mask.data());

            auto p1 =
                Clock::now();

            t.postprocess_ms =
                elapsed_ms(p0, p1);

            t.e2e_ms =
                elapsed_ms(
                    slot->arrival,
                    p1);

            t.completion_s =
                std::chrono::duration<double>(
                    p1 - run_start)
                .count();

            // return reusable aligned slot
            free_q.push(slot);
          }
        });
  }

  gate.wait_ready();

  run_start =
      Clock::now();

  gate.release();

  for (auto& t : pre_threads) {
    t.join();
  }

  // stop DPU workers
  for (int r = 0;
       r < o.runners;
       ++r) {
    dpu_q[
        static_cast<size_t>(r)]
    ->push(nullptr);
  }

  for (auto& t : dpu_threads) {
    t.join();
  }

  // stop post workers
  for (int p = 0;
       p < o.post_workers;
       ++p) {
    post_q.push(nullptr);
  }

  for (auto& t : post_threads) {
    t.join();
  }

  auto run_end =
      Clock::now();

  result.wall_s =
      std::chrono::duration<double>(
          run_end - run_start)
      .count();

  result.completed =
      result.samples.size();

  result.throughput_fps =
      static_cast<double>(
          result.completed)
      /
      result.wall_s;

  return finalize_result(
      std::move(result));
}

// ============================================================================
// WRITE RESULTS
// ============================================================================

static void write_samples_csv(
    const fs::path& path,
    const BenchmarkResult& r) {
  std::ofstream f(
      path,
      std::ios::app);

  if (f.tellp() == 0) {
    f
        << "mode,job,dataset_index,lane,pre_worker,post_worker,"
        << "slot_wait_ms,io_ms,preprocess_ms,pre_dpu_queue_ms,"
        << "input_sync_ms,dpu_ms,output_sync_ms,dpu_post_queue_ms,"
        << "postprocess_ms,e2e_ms,completion_s,instantaneous_fps\n";
  }

  f << std::setprecision(12);

  for (const auto& t : r.samples) {
    f
        << r.mode << ","
        << t.job << ","
        << t.dataset_index << ","
        << t.lane << ","
        << t.pre_worker << ","
        << t.post_worker << ","
        << t.slot_wait_ms << ","
        << t.io_ms << ","
        << t.preprocess_ms << ","
        << t.pre_dpu_queue_ms << ","
        << t.input_sync_ms << ","
        << t.dpu_ms << ","
        << t.output_sync_ms << ","
        << t.dpu_post_queue_ms << ","
        << t.postprocess_ms << ","
        << t.e2e_ms << ","
        << t.completion_s << ","
        << (
            t.e2e_ms > 0.0
            ? 1000.0 / t.e2e_ms
            : 0.0)
        << "\n";
  }
}

static void write_summary_header(
    std::ofstream& f) {
  f
      << "mode,batch,runners,pre_workers,post_workers,slots_per_runner,"
      << "completed,wall_s,throughput_fps,"
      << "equiv_fps_avg,equiv_fps_min,equiv_fps_max,equiv_fps_p95,equiv_fps_p99,"
      << "completion_fps_avg,completion_fps_min,completion_fps_max,completion_fps_p95,completion_fps_p99,"
      << "latency_mean_ms,latency_median_ms,latency_min_ms,latency_max_ms,"
      << "latency_p90_ms,latency_p95_ms,latency_p99_ms,"
      << "latency_stddev_ms,latency_cv,latency_p99_minus_p50_ms,"
      << "dpu_mean_ms,dpu_p95_ms,dpu_p99_ms,"
      << "preprocess_mean_ms,postprocess_mean_ms,"
      << "inter_completion_mean_ms,inter_completion_p95_ms,inter_completion_p99_ms\n";
}

static void write_summary_row(
    std::ofstream& f,
    const BenchmarkResult& r) {
  f
      << std::setprecision(12)
      << r.mode << ","
      << r.batch << ","
      << r.runners << ","
      << r.pre_workers << ","
      << r.post_workers << ","
      << r.slots_per_runner << ","
      << r.completed << ","
      << r.wall_s << ","
      << r.throughput_fps << ","
      << (r.latency.mean > 0.0 ? 1000.0 / r.latency.mean : 0.0) << ","
      << (r.latency.max > 0.0 ? 1000.0 / r.latency.max : 0.0) << ","
      << (r.latency.min > 0.0 ? 1000.0 / r.latency.min : 0.0) << ","
      << (r.latency.p95 > 0.0 ? 1000.0 / r.latency.p95 : 0.0) << ","
      << (r.latency.p99 > 0.0 ? 1000.0 / r.latency.p99 : 0.0) << ","
      << r.throughput_fps << ","
      << (r.inter_completion.max > 0.0 ? 1000.0 / r.inter_completion.max : r.throughput_fps) << ","
      << (r.inter_completion.min > 0.0 ? 1000.0 / r.inter_completion.min : r.throughput_fps) << ","
      << (r.inter_completion.p95 > 0.0 ? 1000.0 / r.inter_completion.p95 : r.throughput_fps) << ","
      << (r.inter_completion.p99 > 0.0 ? 1000.0 / r.inter_completion.p99 : r.throughput_fps) << ","
      << r.latency.mean << ","
      << r.latency.median << ","
      << r.latency.min << ","
      << r.latency.max << ","
      << r.latency.p90 << ","
      << r.latency.p95 << ","
      << r.latency.p99 << ","
      << r.latency.stddev << ","
      << r.latency.cv << ","
      << (r.latency.p99 - r.latency.median) << ","
      << r.dpu.mean << ","
      << r.dpu.p95 << ","
      << r.dpu.p99 << ","
      << r.preprocess.mean << ","
      << r.postprocess.mean << ","
      << r.inter_completion.mean << ","
      << r.inter_completion.p95 << ","
      << r.inter_completion.p99
      << "\n";
}

static void print_result(
    const BenchmarkResult& r) {
  std::cout
      << "\n============================================================\n"
      << r.mode
      << "\n============================================================\n";

  std::cout
      << "batch              = "
      << r.batch
      << "\n"
      << "runners            = "
      << r.runners
      << "\n"
      << "pre_workers        = "
      << r.pre_workers
      << "\n"
      << "post_workers       = "
      << r.post_workers
      << "\n"
      << "slots_per_runner   = "
      << r.slots_per_runner
      << "\n"
      << "completed          = "
      << r.completed
      << "\n"
      << "wall_s             = "
      << std::fixed
      << std::setprecision(6)
      << r.wall_s
      << "\n"
      << "THROUGHPUT FPS     = "
      << r.throughput_fps
      << "\n";

  std::cout
      << "\nFPS EQUIVALENT FROM PER-JOB LATENCY\n"
      << " average = "
      << (r.latency.mean > 0.0 ? 1000.0 / r.latency.mean : 0.0)
      << "\n"
      << " min     = "
      << (r.latency.max > 0.0 ? 1000.0 / r.latency.max : 0.0)
      << "\n"
      << " max     = "
      << (r.latency.min > 0.0 ? 1000.0 / r.latency.min : 0.0)
      << "\n"
      << " p95     = "
      << (r.latency.p95 > 0.0 ? 1000.0 / r.latency.p95 : 0.0)
      << "\n"
      << " p99     = "
      << (r.latency.p99 > 0.0 ? 1000.0 / r.latency.p99 : 0.0)
      << "\n";

  std::cout
      << "\nLATENCY ms\n"
      << " mean   = "
      << r.latency.mean
      << "\n"
      << " median = "
      << r.latency.median
      << "\n"
      << " min    = "
      << r.latency.min
      << "\n"
      << " max    = "
      << r.latency.max
      << "\n"
      << " p90    = "
      << r.latency.p90
      << "\n"
      << " p95    = "
      << r.latency.p95
      << "\n"
      << " p99    = "
      << r.latency.p99
      << "\n"
      << " stddev = "
      << r.latency.stddev
      << "\n"
      << " CV     = "
      << r.latency.cv
      << "\n"
      << " jitter p99-p50 = "
      << (
          r.latency.p99
          - r.latency.median)
      << " ms\n";

  if (r.dpu.count) {
    std::cout
        << "\nDPU STAGE ms\n"
        << " mean = "
        << r.dpu.mean
        << "\n"
        << " p95  = "
        << r.dpu.p95
        << "\n"
        << " p99  = "
        << r.dpu.p99
        << "\n";
  }

  if (r.preprocess.count) {
    std::cout
        << "\nPREPROCESS mean = "
        << r.preprocess.mean
        << " ms\n";
  }

  if (r.postprocess.count) {
    std::cout
        << "POSTPROCESS mean = "
        << r.postprocess.mean
        << " ms\n";
  }

  if (r.inter_completion.count) {
    std::cout
        << "\nINTER-COMPLETION ms\n"
        << " mean = "
        << r.inter_completion.mean
        << "\n"
        << " p95  = "
        << r.inter_completion.p95
        << "\n"
        << " p99  = "
        << r.inter_completion.p99
        << "\n"
        << "OUTPUT-CADENCE FPS\n"
        << " average(system wall) = "
        << r.throughput_fps
        << "\n"
        << " min = "
        << (r.inter_completion.max > 0.0 ? 1000.0 / r.inter_completion.max : 0.0)
        << "\n"
        << " max = "
        << (r.inter_completion.min > 0.0 ? 1000.0 / r.inter_completion.min : 0.0)
        << "\n"
        << " p95 = "
        << (r.inter_completion.p95 > 0.0 ? 1000.0 / r.inter_completion.p95 : 0.0)
        << "\n"
        << " p99 = "
        << (r.inter_completion.p99 > 0.0 ? 1000.0 / r.inter_completion.p99 : 0.0)
        << "\n";
  }
}

static void write_config(
    const Options& o) {
  std::ofstream f(
      fs::path(o.out)
      / "config.txt");

  f
      << "platform=ZCU104\n"
      << "precision=INT8\n"
      << "batch=1\n"
      << "profile="
      << o.profile
      << "\n"
      << "runners="
      << o.runners
      << "\n"
      << "pre_workers="
      << o.pre_workers
      << "\n"
      << "post_workers="
      << o.post_workers
      << "\n"
      << "slots_per_runner="
      << o.slots_per_runner
      << "\n"
      << "iterations="
      << o.iterations
      << "\n"
      << "warmup="
      << o.warmup
      << "\n"
      << "pin="
      << (o.pin ? "true" : "false")
      << "\n"
      << "model="
      << o.model
      << "\n"
      << "dataset="
      << o.dataset
      << "\n"
      << "csv="
      << o.csv
      << "\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
  try {
    cv::setNumThreads(1);
    cv::setUseOptimized(true);

    Options o =
        parse_options(
            argc,
            argv);

    if (!fs::exists(o.model)) {
      throw std::runtime_error(
          "Model not found: "
          + o.model);
    }

    if (!fs::exists(o.dataset)) {
      throw std::runtime_error(
          "Dataset not found: "
          + o.dataset);
    }

    if (!fs::exists(o.csv)) {
      throw std::runtime_error(
          "CSV not found: "
          + o.csv);
    }

    ensure_dir(o.out);

    auto dataset =
        load_dataset_csv(o.csv);

    std::cout
        << "============================================================\n"
        << "HYPERSTARCOP ZCU104 OPTIMIZED BENCHMARK\n"
        << "============================================================\n"
        << "Batch: 1 ALWAYS\n"
        << "Profile: "
        << o.profile
        << "\n"
        << "Dataset images: "
        << dataset.size()
        << "\n"
        << "Runners: "
        << o.runners
        << "\n"
        << "Pre workers: "
        << o.pre_workers
        << "\n"
        << "Post workers: "
        << o.post_workers
        << "\n"
        << "Slots/runner: "
        << o.slots_per_runner
        << "\n"
        << "Pin: "
        << (o.pin ? "yes" : "no")
        << "\n";

    ModelContext model(
        o.model);

    write_config(o);

    if (o.validate) {
      run_validation(
          o,
          dataset,
          model);
    }

    std::vector<BenchmarkResult> results;

    if (o.profile == "all" ||
        o.profile == "baseline") {
      auto baseline =
          run_baseline(
              o,
              dataset,
              model);

      for (auto& r : baseline) {
        results.push_back(
            std::move(r));
      }
    }

    if (o.profile == "all" ||
        o.profile == "max-model-only") {
      results.push_back(
          run_max_model_only(
              o,
              dataset,
              model));
    }

    if (o.profile == "all" ||
        o.profile == "max-e2e") {
      results.push_back(
          run_max_e2e(
              o,
              dataset,
              model));
    }

    std::ofstream summary(
        fs::path(o.out)
        / "benchmark_summary.csv");

    write_summary_header(
        summary);

    const fs::path samples_path =
        fs::path(o.out)
        / "benchmark_samples.csv";

    if (fs::exists(samples_path)) {
      fs::remove(samples_path);
    }

    for (const auto& r : results) {
      print_result(r);

      write_summary_row(
          summary,
          r);

      write_samples_csv(
          samples_path,
          r);
    }

    std::cout
        << "\nResults saved to:\n"
        << o.out
        << "\n";

    return 0;
  }
  catch (const std::exception& e) {
    std::cerr
        << "FATAL: "
        << e.what()
        << "\n";

    return 2;
  }
}
