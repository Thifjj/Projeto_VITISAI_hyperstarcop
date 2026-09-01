// HyperSTARCOP - ZCU104 - VART C++ benchmark/validation
// Vitis AI 3.5 / DPUCZDX8G B4096
//
// MODEL ONLY:
//   measures only runner->execute_async() + runner->wait()
//
// END TO END:
//   4 TIFF reads + normalization + NHWC INT8 quantization
//   + sync input + DPU + sync output
//   + dequantization + sigmoid + threshold
//
// NOT included in E2E:
//   ground-truth read, metrics, CSV, PNG/plots
//
// Metrics match the original HyperSTARCOP validation:
//   TP, FP, FN, TN, Precision, Recall, F1, IoU, Accuracy
//
// Build:
//   g++ -O3 -DNDEBUG -std=c++17 hyperstarcop_zcu104_benchmark.cpp \
//       -o hyperstarcop_benchmark \
//       $(pkg-config --cflags --libs opencv4) \
//       -lvart-runner -lxir -lpthread

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/opencv.hpp>

#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

using Clock = std::chrono::steady_clock;

// ============================================================
// CONFIGURATION
// ============================================================

static const std::string XMODEL =
    "/home/root/hyperstarcop.xmodel";

static const std::string DATASET =
    "/home/root/STARCOP_mini";

static const std::string CSV_TEST =
    "/home/root/STARCOP_mini/test_mini10.csv";

static const std::string OUTPUT_DIR =
    "/home/root/hyperstarcop_cpp_results";

static const std::string FIG_DIR =
    "/home/root/hyperstarcop_cpp_results/figuras_por_imagem";

static constexpr int H = 512;
static constexpr int W = 512;
static constexpr int C = 4;

static constexpr int MODEL_ONLY_WARMUP = 10;
static constexpr int MODEL_ONLY_REPEATS = 100;

static constexpr int E2E_WARMUP_PASSES = 1;
static constexpr int E2E_MEASURE_PASSES = 5;

// Documented original FP32 reference for the same mini validation flow.
static constexpr uint64_t REF_TP = 40310;
static constexpr uint64_t REF_FP = 4847;
static constexpr uint64_t REF_FN = 3467;
static constexpr uint64_t REF_TN = 2310672;


// ============================================================
// SMALL HELPERS
// ============================================================

static void make_dir(const std::string& p) {
    if (::mkdir(p.c_str(), 0755) != 0) {
        // Ignore if it already exists.
    }
}

static bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

static double elapsed_ms(
    const Clock::time_point& a,
    const Clock::time_point& b
) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::string join_path(
    const std::string& a,
    const std::string& b
) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}


// ============================================================
// CPU FLAT TENSOR BUFFER
// Same concept used by AMD Vitis AI C++ examples.
// ============================================================

class CpuFlatTensorBuffer : public vart::TensorBuffer {
public:
    CpuFlatTensorBuffer(
        void* data,
        const xir::Tensor* tensor
    )
        : vart::TensorBuffer(tensor),
          data_(data) {}

    std::pair<uint64_t, size_t> data(
        const std::vector<int> idx = {}
    ) override {
        const uint32_t elem_bytes =
            static_cast<uint32_t>(
                std::ceil(
                    tensor_->get_data_type().bit_width / 8.0
                )
            );

        if (idx.empty()) {
            return {
                reinterpret_cast<uint64_t>(data_),
                tensor_->get_element_num() * elem_bytes
            };
        }

        const auto dims = tensor_->get_shape();

        size_t offset = 0;

        for (size_t k = 0; k < idx.size(); ++k) {
            size_t stride = 1;

            for (
                size_t m = k + 1;
                m < dims.size();
                ++m
            ) {
                stride *= static_cast<size_t>(dims[m]);
            }

            offset +=
                static_cast<size_t>(idx[k]) * stride;
        }

        const size_t total =
            tensor_->get_element_num();

        return {
            reinterpret_cast<uint64_t>(data_)
                + offset * elem_bytes,
            (total - offset) * elem_bytes
        };
    }

private:
    void* data_;
};


// ============================================================
// XIR / VART
// ============================================================

static std::vector<const xir::Subgraph*>
get_dpu_subgraphs(const xir::Graph* graph) {
    std::vector<const xir::Subgraph*> result;

    auto root = graph->get_root_subgraph();
    auto children = root->children_topological_sort();

    for (auto* sg : children) {
        if (!sg->has_attr("device")) {
            continue;
        }

        const auto device =
            sg->get_attr<std::string>("device");

        if (device == "DPU") {
            result.push_back(sg);
        }
    }

    return result;
}

static int get_fix_point(
    const xir::Tensor* tensor,
    int fallback
) {
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

    std::cerr
        << "WARNING: fix_point missing for tensor "
        << tensor->get_name()
        << "; using fallback="
        << fallback
        << "\n";

    return fallback;
}

static std::string shape_to_string(
    const std::vector<int32_t>& s
) {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < s.size(); ++i) {
        if (i) oss << ",";
        oss << s[i];
    }

    oss << "]";
    return oss.str();
}


// ============================================================
// CSV
// ============================================================

static std::vector<std::string>
parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];

        if (ch == '"') {
            if (
                quoted &&
                i + 1 < line.size() &&
                line[i + 1] == '"'
            ) {
                current += '"';
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(current);
            current.clear();
        } else {
            current += ch;
        }
    }

    fields.push_back(current);
    return fields;
}

struct SampleRow {
    std::string id;
    std::string has_plume;
};

static std::vector<SampleRow>
load_test_csv(const std::string& path) {
    std::ifstream f(path);

    if (!f) {
        throw std::runtime_error(
            "Cannot open CSV: " + path
        );
    }

    std::string line;

    if (!std::getline(f, line)) {
        throw std::runtime_error(
            "Empty CSV: " + path
        );
    }

    auto header = parse_csv_line(line);

    int id_col = -1;
    int plume_col = -1;

    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == "id") {
            id_col = static_cast<int>(i);
        }

        if (header[i] == "has_plume") {
            plume_col = static_cast<int>(i);
        }
    }

    if (id_col < 0) {
        throw std::runtime_error(
            "CSV does not have an 'id' column."
        );
    }

    std::vector<SampleRow> rows;

    while (std::getline(f, line)) {
        if (line.empty()) continue;

        auto fields = parse_csv_line(line);

        if (
            static_cast<size_t>(id_col)
            >= fields.size()
        ) {
            continue;
        }

        SampleRow r;
        r.id = fields[id_col];

        if (
            plume_col >= 0 &&
            static_cast<size_t>(plume_col)
                < fields.size()
        ) {
            r.has_plume = fields[plume_col];
        }

        rows.push_back(r);
    }

    return rows;
}


// ============================================================
// TIFF
// ============================================================

static cv::Mat read_tif_float(
    const std::string& path
) {
    cv::Mat img =
        cv::imread(
            path,
            cv::IMREAD_UNCHANGED
        );

    if (img.empty()) {
        throw std::runtime_error(
            "OpenCV could not read TIFF: " + path
        );
    }

    if (img.channels() != 1) {
        throw std::runtime_error(
            "Expected single-channel TIFF: " + path
        );
    }

    cv::Mat out;

    img.convertTo(
        out,
        CV_32F
    );

    if (
        out.rows != H ||
        out.cols != W
    ) {
        std::ostringstream oss;
        oss
            << "Expected 512x512, got "
            << out.cols
            << "x"
            << out.rows
            << ": "
            << path;

        throw std::runtime_error(
            oss.str()
        );
    }

    return out;
}

struct RawChannels {
    cv::Mat mag1c;
    cv::Mat red;
    cv::Mat green;
    cv::Mat blue;
};

static RawChannels
read_input_channels(
    const std::string& folder
) {
    RawChannels c;

    c.mag1c = read_tif_float(
        join_path(folder, "mag1c.tif")
    );

    c.red = read_tif_float(
        join_path(
            folder,
            "TOA_AVIRIS_640nm.tif"
        )
    );

    c.green = read_tif_float(
        join_path(
            folder,
            "TOA_AVIRIS_550nm.tif"
        )
    );

    c.blue = read_tif_float(
        join_path(
            folder,
            "TOA_AVIRIS_460nm.tif"
        )
    );

    return c;
}

static cv::Mat read_label(
    const std::string& folder
) {
    cv::Mat label_f =
        read_tif_float(
            join_path(
                folder,
                "labelbinary.tif"
            )
        );

    cv::Mat label_u8(
        H,
        W,
        CV_8U
    );

    for (int y = 0; y < H; ++y) {
        const float* src =
            label_f.ptr<float>(y);

        uint8_t* dst =
            label_u8.ptr<uint8_t>(y);

        for (int x = 0; x < W; ++x) {
            dst[x] =
                src[x] > 0.0f ? 1 : 0;
        }
    }

    return label_u8;
}


// ============================================================
// PREPROCESSING
// ============================================================

static inline float clip02(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 2.0f) return 2.0f;
    return x;
}

static void preprocess_to_int8_nhwc(
    const RawChannels& c,
    float input_scale,
    std::vector<int8_t>& input
) {
    input.resize(
        static_cast<size_t>(H)
        * W
        * C
    );

    for (int y = 0; y < H; ++y) {
        const float* pm =
            c.mag1c.ptr<float>(y);

        const float* pr =
            c.red.ptr<float>(y);

        const float* pg =
            c.green.ptr<float>(y);

        const float* pb =
            c.blue.ptr<float>(y);

        for (int x = 0; x < W; ++x) {
            const float v[4] = {
                clip02(pm[x] / 1750.0f),
                clip02(pr[x] / 60.0f),
                clip02(pg[x] / 60.0f),
                clip02(pb[x] / 60.0f)
            };

            const size_t base =
                (
                    static_cast<size_t>(y)
                    * W
                    + x
                ) * C;

            for (int ch = 0; ch < 4; ++ch) {
                int q =
                    static_cast<int>(
                        std::lrint(
                            v[ch] * input_scale
                        )
                    );

                q = std::max(
                    -128,
                    std::min(
                        127,
                        q
                    )
                );

                input[base + ch] =
                    static_cast<int8_t>(q);
            }
        }
    }
}


// ============================================================
// POSTPROCESSING
// ============================================================

struct Prediction {
    cv::Mat logits; // CV_32F
    cv::Mat prob;   // CV_32F
    cv::Mat pred;   // CV_8U, values 0/1
    float prob_min = 0.0f;
    float prob_max = 0.0f;
};

static Prediction postprocess(
    const std::vector<int8_t>& output,
    float output_scale
) {
    Prediction p;

    p.logits = cv::Mat(
        H,
        W,
        CV_32F
    );

    p.prob = cv::Mat(
        H,
        W,
        CV_32F
    );

    p.pred = cv::Mat(
        H,
        W,
        CV_8U
    );

    p.prob_min =
        std::numeric_limits<float>::infinity();

    p.prob_max =
        -std::numeric_limits<float>::infinity();

    for (int y = 0; y < H; ++y) {
        float* logit_row =
            p.logits.ptr<float>(y);

        float* prob_row =
            p.prob.ptr<float>(y);

        uint8_t* pred_row =
            p.pred.ptr<uint8_t>(y);

        for (int x = 0; x < W; ++x) {
            const int8_t q =
                output[
                    static_cast<size_t>(y)
                    * W
                    + x
                ];

            const float logit =
                static_cast<float>(q)
                * output_scale;

            // Stable enough for the quantized range.
            const float z =
                std::max(
                    -80.0f,
                    std::min(
                        80.0f,
                        logit
                    )
                );

            const float prob =
                1.0f /
                (
                    1.0f
                    + std::exp(-z)
                );

            logit_row[x] = logit;
            prob_row[x] = prob;

            // Same as sigmoid(logit) > 0.5
            // and exactly equivalent to q > 0.
            pred_row[x] =
                q > 0 ? 1 : 0;

            p.prob_min =
                std::min(
                    p.prob_min,
                    prob
                );

            p.prob_max =
                std::max(
                    p.prob_max,
                    prob
                );
        }
    }

    return p;
}


// ============================================================
// SEGMENTATION METRICS
// ============================================================

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
    const cv::Mat& pred,
    const cv::Mat& label
) {
    Counts c;

    for (int y = 0; y < H; ++y) {
        const uint8_t* p =
            pred.ptr<uint8_t>(y);

        const uint8_t* l =
            label.ptr<uint8_t>(y);

        for (int x = 0; x < W; ++x) {
            const bool pv = p[x] != 0;
            const bool lv = l[x] != 0;

            if (pv && lv) ++c.tp;
            else if (pv && !lv) ++c.fp;
            else if (!pv && lv) ++c.fn;
            else ++c.tn;
        }
    }

    return c;
}

static Metrics calc_metrics(
    const Counts& c
) {
    Metrics m;

    if (c.tp + c.fp > 0) {
        m.precision =
            static_cast<double>(c.tp)
            /
            static_cast<double>(
                c.tp + c.fp
            );
    }

    if (c.tp + c.fn > 0) {
        m.recall =
            static_cast<double>(c.tp)
            /
            static_cast<double>(
                c.tp + c.fn
            );
    }

    if (
        m.precision + m.recall
        > 0.0
    ) {
        m.f1 =
            2.0
            * m.precision
            * m.recall
            /
            (
                m.precision
                + m.recall
            );
    }

    if (
        c.tp + c.fp + c.fn
        > 0
    ) {
        m.iou =
            static_cast<double>(c.tp)
            /
            static_cast<double>(
                c.tp
                + c.fp
                + c.fn
            );
    }

    const uint64_t total =
        c.tp
        + c.fp
        + c.fn
        + c.tn;

    if (total > 0) {
        m.accuracy =
            static_cast<double>(
                c.tp + c.tn
            )
            /
            static_cast<double>(
                total
            );
    }

    return m;
}


// ============================================================
// BENCHMARK STATS
// ============================================================

static double percentile(
    std::vector<double> values,
    double p
) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    if (values.size() == 1) {
        return values[0];
    }

    const double pos =
        (p / 100.0)
        * static_cast<double>(
            values.size() - 1
        );

    const size_t lo =
        static_cast<size_t>(
            std::floor(pos)
        );

    const size_t hi =
        static_cast<size_t>(
            std::ceil(pos)
        );

    const double frac =
        pos - static_cast<double>(lo);

    return
        values[lo]
        * (1.0 - frac)
        +
        values[hi]
        * frac;
}

struct PerfStats {
    size_t n = 0;

    double latency_avg_ms = 0.0;
    double latency_min_ms = 0.0;
    double latency_max_ms = 0.0;
    double latency_p95_ms = 0.0;
    double latency_p99_ms = 0.0;

    double fps_avg = 0.0;
    double fps_min = 0.0;
    double fps_max = 0.0;
    double fps_p95 = 0.0;
    double fps_p99 = 0.0;
};

static PerfStats perf_stats(
    const std::vector<double>& ms
) {
    if (ms.empty()) {
        throw std::runtime_error(
            "No benchmark samples."
        );
    }

    PerfStats s;

    s.n = ms.size();

    const double sum_ms =
        std::accumulate(
            ms.begin(),
            ms.end(),
            0.0
        );

    s.latency_avg_ms =
        sum_ms
        /
        static_cast<double>(
            ms.size()
        );

    auto mm =
        std::minmax_element(
            ms.begin(),
            ms.end()
        );

    s.latency_min_ms = *mm.first;
    s.latency_max_ms = *mm.second;

    s.latency_p95_ms =
        percentile(ms, 95.0);

    s.latency_p99_ms =
        percentile(ms, 99.0);

    // Average throughput over total measured time.
    s.fps_avg =
        static_cast<double>(
            ms.size()
        )
        /
        (sum_ms / 1000.0);

    // Best/worst instantaneous rate.
    s.fps_max =
        1000.0
        /
        s.latency_min_ms;

    s.fps_min =
        1000.0
        /
        s.latency_max_ms;

    // Tail-FPS derived from tail latency.
    // This is more useful than percentile(inst_fps)
    // because it preserves the latency tail meaning.
    s.fps_p95 =
        1000.0
        /
        s.latency_p95_ms;

    s.fps_p99 =
        1000.0
        /
        s.latency_p99_ms;

    return s;
}

static void print_perf(
    const std::string& name,
    const PerfStats& s
) {
    std::cout
        << "\n============================================================\n"
        << name
        << "\n============================================================\n";

    std::cout
        << "Samples: "
        << s.n
        << "\n\n";

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "LATENCY (ms)\n"
        << "  Average : "
        << s.latency_avg_ms
        << "\n"
        << "  Min     : "
        << s.latency_min_ms
        << "\n"
        << "  Max     : "
        << s.latency_max_ms
        << "\n"
        << "  P95     : "
        << s.latency_p95_ms
        << "\n"
        << "  P99     : "
        << s.latency_p99_ms
        << "\n\n";

    std::cout
        << "FPS\n"
        << "  Average : "
        << s.fps_avg
        << "\n"
        << "  Min     : "
        << s.fps_min
        << "\n"
        << "  Max     : "
        << s.fps_max
        << "\n"
        << "  P95(*)  : "
        << s.fps_p95
        << "\n"
        << "  P99(*)  : "
        << s.fps_p99
        << "\n"
        << "  (*) derived from P95/P99 latency\n";
}


// ============================================================
// VISUALIZATION
// All of this happens OUTSIDE benchmark timing.
// ============================================================

static cv::Mat percentile_normalized_u8(
    const cv::Mat& src,
    double low_p,
    double high_p
) {
    std::vector<float> vals;
    vals.reserve(
        static_cast<size_t>(
            src.rows * src.cols
        )
    );

    for (int y = 0; y < src.rows; ++y) {
        const float* row =
            src.ptr<float>(y);

        for (int x = 0; x < src.cols; ++x) {
            vals.push_back(row[x]);
        }
    }

    const double low =
        percentile(
            std::vector<double>(
                vals.begin(),
                vals.end()
            ),
            low_p
        );

    const double high =
        percentile(
            std::vector<double>(
                vals.begin(),
                vals.end()
            ),
            high_p
        );

    const double denom =
        std::max(
            1e-12,
            high - low
        );

    cv::Mat out(
        src.size(),
        CV_8U
    );

    for (int y = 0; y < src.rows; ++y) {
        const float* s =
            src.ptr<float>(y);

        uint8_t* d =
            out.ptr<uint8_t>(y);

        for (int x = 0; x < src.cols; ++x) {
            double v =
                (
                    static_cast<double>(s[x])
                    - low
                )
                /
                denom;

            v = std::max(
                0.0,
                std::min(
                    1.0,
                    v
                )
            );

            d[x] =
                static_cast<uint8_t>(
                    std::lrint(
                        v * 255.0
                    )
                );
        }
    }

    return out;
}

static cv::Mat mask_to_bgr(
    const cv::Mat& mask
) {
    cv::Mat u8;

    mask.convertTo(
        u8,
        CV_8U,
        255.0
    );

    cv::Mat bgr;

    cv::cvtColor(
        u8,
        bgr,
        cv::COLOR_GRAY2BGR
    );

    return bgr;
}

static cv::Mat add_title(
    const cv::Mat& image,
    const std::string& title
) {
    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(500, 500),
        0,
        0,
        cv::INTER_NEAREST
    );

    cv::Mat panel(
        550,
        500,
        CV_8UC3,
        cv::Scalar(255,255,255)
    );

    resized.copyTo(
        panel(
            cv::Rect(
                0,
                50,
                500,
                500
            )
        )
    );

    cv::putText(
        panel,
        title,
        cv::Point(10, 32),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0,0,0),
        1,
        cv::LINE_AA
    );

    return panel;
}

static void save_validation_figure(
    const std::string& sample_id,
    const RawChannels& raw,
    const cv::Mat& label,
    const Prediction& prediction,
    const Counts& counts,
    const Metrics& metrics
) {
    // RGB visual.
    cv::Mat r =
        percentile_normalized_u8(
            raw.red,
            2.0,
            98.0
        );

    cv::Mat g =
        percentile_normalized_u8(
            raw.green,
            2.0,
            98.0
        );

    cv::Mat b =
        percentile_normalized_u8(
            raw.blue,
            2.0,
            98.0
        );

    cv::Mat rgb;

    std::vector<cv::Mat> bgr_channels = {
        b, g, r
    };

    cv::merge(
        bgr_channels,
        rgb
    );

    // MAG1C.
    cv::Mat mag_u8 =
        percentile_normalized_u8(
            raw.mag1c,
            1.0,
            99.0
        );

    cv::Mat mag_color;

    cv::applyColorMap(
        mag_u8,
        mag_color,
        cv::COLORMAP_MAGMA
    );

    // GT.
    cv::Mat gt =
        mask_to_bgr(label);

    // Probability.
    cv::Mat prob_u8;

    prediction.prob.convertTo(
        prob_u8,
        CV_8U,
        255.0
    );

    cv::Mat prob_color;

    cv::applyColorMap(
        prob_u8,
        prob_color,
        cv::COLORMAP_INFERNO
    );

    // Prediction.
    cv::Mat pred =
        mask_to_bgr(
            prediction.pred
        );

    // Difference map:
    // TN black, TP green, FP red, FN blue.
    cv::Mat diff(
        H,
        W,
        CV_8UC3,
        cv::Scalar(0,0,0)
    );

    for (int y = 0; y < H; ++y) {
        const uint8_t* p =
            prediction.pred.ptr<uint8_t>(y);

        const uint8_t* l =
            label.ptr<uint8_t>(y);

        cv::Vec3b* d =
            diff.ptr<cv::Vec3b>(y);

        for (int x = 0; x < W; ++x) {
            if (p[x] && l[x]) {
                d[x] = cv::Vec3b(
                    0,255,0
                ); // TP green
            } else if (p[x] && !l[x]) {
                d[x] = cv::Vec3b(
                    0,0,255
                ); // FP red
            } else if (!p[x] && l[x]) {
                d[x] = cv::Vec3b(
                    255,0,0
                ); // FN blue
            }
        }
    }

    cv::Mat p00 =
        add_title(
            rgb,
            "AVIRIS RGB"
        );

    cv::Mat p01 =
        add_title(
            mag_color,
            "MAG1C"
        );

    cv::Mat p02 =
        add_title(
            gt,
            "Ground Truth"
        );

    cv::Mat p10 =
        add_title(
            prob_color,
            "Methane probability"
        );

    cv::Mat p11 =
        add_title(
            pred,
            "Prediction threshold=0.5"
        );

    cv::Mat p12 =
        add_title(
            diff,
            "TP green | FP red | FN blue"
        );

    cv::Mat top;
    cv::Mat bottom;

    cv::hconcat(
        std::vector<cv::Mat>{
            p00, p01, p02
        },
        top
    );

    cv::hconcat(
        std::vector<cv::Mat>{
            p10, p11, p12
        },
        bottom
    );

    cv::Mat body;

    cv::vconcat(
        std::vector<cv::Mat>{
            top, bottom
        },
        body
    );

    cv::Mat canvas(
        body.rows + 100,
        body.cols,
        CV_8UC3,
        cv::Scalar(255,255,255)
    );

    std::ostringstream title;

    title
        << "HyperSTARCOP ZCU104 - "
        << sample_id
        << " | Precision="
        << std::fixed
        << std::setprecision(4)
        << metrics.precision
        << " Recall="
        << metrics.recall
        << " F1="
        << metrics.f1
        << " IoU="
        << metrics.iou
        << " Accuracy="
        << metrics.accuracy;

    cv::putText(
        canvas,
        title.str(),
        cv::Point(15, 38),
        cv::FONT_HERSHEY_SIMPLEX,
        0.70,
        cv::Scalar(0,0,0),
        2,
        cv::LINE_AA
    );

    std::ostringstream line2;

    line2
        << "TP="
        << counts.tp
        << " FP="
        << counts.fp
        << " FN="
        << counts.fn
        << " TN="
        << counts.tn;

    cv::putText(
        canvas,
        line2.str(),
        cv::Point(15, 72),
        cv::FONT_HERSHEY_SIMPLEX,
        0.70,
        cv::Scalar(0,0,0),
        1,
        cv::LINE_AA
    );

    body.copyTo(
        canvas(
            cv::Rect(
                0,
                100,
                body.cols,
                body.rows
            )
        )
    );

    cv::imwrite(
        join_path(
            FIG_DIR,
            "validacao_"
            + sample_id
            + ".png"
        ),
        canvas
    );
}

static cv::Mat make_bar_chart(
    const std::string& title,
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const std::string& y_label
) {
    const int width = 1200;
    const int height = 700;

    cv::Mat img(
        height,
        width,
        CV_8UC3,
        cv::Scalar(255,255,255)
    );

    const int left = 100;
    const int right = 60;
    const int top = 100;
    const int bottom = 120;

    const int plot_w =
        width - left - right;

    const int plot_h =
        height - top - bottom;

    cv::line(
        img,
        cv::Point(left, top),
        cv::Point(left, top + plot_h),
        cv::Scalar(0,0,0),
        2
    );

    cv::line(
        img,
        cv::Point(left, top + plot_h),
        cv::Point(left + plot_w, top + plot_h),
        cv::Scalar(0,0,0),
        2
    );

    cv::putText(
        img,
        title,
        cv::Point(60, 50),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0,0,0),
        2,
        cv::LINE_AA
    );

    cv::putText(
        img,
        y_label,
        cv::Point(10, 80),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0,0,0),
        1,
        cv::LINE_AA
    );

    double vmax = 0.0;

    for (double v : values) {
        vmax = std::max(
            vmax,
            v
        );
    }

    if (vmax <= 0.0) {
        vmax = 1.0;
    }

    vmax *= 1.15;

    const double slot =
        static_cast<double>(plot_w)
        /
        static_cast<double>(
            values.size()
        );

    const int bar_w =
        static_cast<int>(
            slot * 0.55
        );

    for (
        size_t i = 0;
        i < values.size();
        ++i
    ) {
        const int x_center =
            left
            +
            static_cast<int>(
                slot
                * (
                    static_cast<double>(i)
                    + 0.5
                )
            );

        const int bh =
            static_cast<int>(
                values[i]
                / vmax
                * plot_h
            );

        cv::rectangle(
            img,
            cv::Rect(
                x_center - bar_w / 2,
                top + plot_h - bh,
                bar_w,
                bh
            ),
            cv::Scalar(
                180,
                120,
                40
            ),
            cv::FILLED
        );

        std::ostringstream val;

        val
            << std::fixed
            << std::setprecision(3)
            << values[i];

        cv::putText(
            img,
            val.str(),
            cv::Point(
                x_center - bar_w / 2,
                top + plot_h - bh - 10
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0,0,0),
            1,
            cv::LINE_AA
        );

        cv::putText(
            img,
            labels[i],
            cv::Point(
                x_center - bar_w / 2,
                top + plot_h + 35
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0,0,0),
            1,
            cv::LINE_AA
        );
    }

    return img;
}

static void save_perf_charts(
    const std::string& prefix,
    const std::string& pretty_name,
    const PerfStats& s
) {
    const std::vector<std::string> names = {
        "Average",
        "Min",
        "Max",
        "P95",
        "P99"
    };

    cv::Mat latency =
        make_bar_chart(
            pretty_name + " - Latency",
            names,
            {
                s.latency_avg_ms,
                s.latency_min_ms,
                s.latency_max_ms,
                s.latency_p95_ms,
                s.latency_p99_ms
            },
            "milliseconds"
        );

    cv::imwrite(
        join_path(
            OUTPUT_DIR,
            prefix
            + "_latency.png"
        ),
        latency
    );

    cv::Mat fps =
        make_bar_chart(
            pretty_name + " - FPS",
            names,
            {
                s.fps_avg,
                s.fps_min,
                s.fps_max,
                s.fps_p95,
                s.fps_p99
            },
            "frames/second"
        );

    cv::imwrite(
        join_path(
            OUTPUT_DIR,
            prefix
            + "_fps.png"
        ),
        fps
    );
}


// ============================================================
// RESULT ROWS
// ============================================================

struct ValidationResult {
    std::string id;
    std::string has_plume;

    uint64_t real_pixels = 0;
    uint64_t predicted_pixels = 0;

    Counts counts;
    Metrics metrics;

    float prob_min = 0.0f;
    float prob_max = 0.0f;
};

static uint64_t count_positive(
    const cv::Mat& m
) {
    uint64_t n = 0;

    for (int y = 0; y < m.rows; ++y) {
        const uint8_t* row =
            m.ptr<uint8_t>(y);

        for (int x = 0; x < m.cols; ++x) {
            if (row[x]) ++n;
        }
    }

    return n;
}


// ============================================================
// CSV OUTPUT
// ============================================================

static void save_validation_csv(
    const std::vector<ValidationResult>& r
) {
    std::ofstream f(
        join_path(
            OUTPUT_DIR,
            "metricas_por_imagem.csv"
        )
    );

    f
        << "id,has_plume,"
        << "pixels_reais,pixels_previstos,"
        << "TP,FP,FN,TN,"
        << "precision,recall,f1,iou,accuracy,"
        << "prob_min,prob_max\n";

    f
        << std::setprecision(12);

    for (const auto& x : r) {
        f
            << x.id << ","
            << x.has_plume << ","
            << x.real_pixels << ","
            << x.predicted_pixels << ","
            << x.counts.tp << ","
            << x.counts.fp << ","
            << x.counts.fn << ","
            << x.counts.tn << ","
            << x.metrics.precision << ","
            << x.metrics.recall << ","
            << x.metrics.f1 << ","
            << x.metrics.iou << ","
            << x.metrics.accuracy << ","
            << x.prob_min << ","
            << x.prob_max
            << "\n";
    }
}

static void save_global_csv(
    const Counts& c,
    const Metrics& m,
    const Metrics& mean_m,
    size_t num_images
) {
    std::ofstream f(
        join_path(
            OUTPUT_DIR,
            "metricas_globais.csv"
        )
    );

    f
        << "num_imagens,"
        << "TP,FP,FN,TN,"
        << "precision_global,"
        << "recall_global,"
        << "f1_global,"
        << "iou_global,"
        << "accuracy_global,"
        << "precision_media,"
        << "recall_media,"
        << "f1_media,"
        << "iou_media,"
        << "accuracy_media\n";

    f
        << std::setprecision(12)
        << num_images << ","
        << c.tp << ","
        << c.fp << ","
        << c.fn << ","
        << c.tn << ","
        << m.precision << ","
        << m.recall << ","
        << m.f1 << ","
        << m.iou << ","
        << m.accuracy << ","
        << mean_m.precision << ","
        << mean_m.recall << ","
        << mean_m.f1 << ","
        << mean_m.iou << ","
        << mean_m.accuracy
        << "\n";
}

static void save_reference_csv(
    const Metrics& measured
) {
    Counts ref_c;
    ref_c.tp = REF_TP;
    ref_c.fp = REF_FP;
    ref_c.fn = REF_FN;
    ref_c.tn = REF_TN;

    const Metrics ref =
        calc_metrics(ref_c);

    std::ofstream f(
        join_path(
            OUTPUT_DIR,
            "comparacao_referencia_original.csv"
        )
    );

    f
        << "metrica,original_fp32_documentado,zcu104,delta\n";

    f
        << std::setprecision(12);

    f
        << "precision,"
        << ref.precision << ","
        << measured.precision << ","
        << measured.precision - ref.precision
        << "\n";

    f
        << "recall,"
        << ref.recall << ","
        << measured.recall << ","
        << measured.recall - ref.recall
        << "\n";

    f
        << "f1,"
        << ref.f1 << ","
        << measured.f1 << ","
        << measured.f1 - ref.f1
        << "\n";

    f
        << "iou,"
        << ref.iou << ","
        << measured.iou << ","
        << measured.iou - ref.iou
        << "\n";

    f
        << "accuracy,"
        << ref.accuracy << ","
        << measured.accuracy << ","
        << measured.accuracy - ref.accuracy
        << "\n";
}

static void save_benchmark_summary_csv(
    const PerfStats& mo,
    const PerfStats& e2e
) {
    std::ofstream f(
        join_path(
            OUTPUT_DIR,
            "benchmark_summary.csv"
        )
    );

    f
        << "mode,samples,"
        << "fps_average,fps_min,fps_max,fps_p95,fps_p99,"
        << "latency_avg_ms,latency_min_ms,latency_max_ms,"
        << "latency_p95_ms,latency_p99_ms\n";

    auto write_one =
        [&f](
            const std::string& name,
            const PerfStats& s
        ) {
            f
                << std::setprecision(12)
                << name << ","
                << s.n << ","
                << s.fps_avg << ","
                << s.fps_min << ","
                << s.fps_max << ","
                << s.fps_p95 << ","
                << s.fps_p99 << ","
                << s.latency_avg_ms << ","
                << s.latency_min_ms << ","
                << s.latency_max_ms << ","
                << s.latency_p95_ms << ","
                << s.latency_p99_ms
                << "\n";
        };

    write_one(
        "model_only",
        mo
    );

    write_one(
        "end_to_end",
        e2e
    );
}

static void save_raw_benchmark_csv(
    const std::vector<double>& mo,
    const std::vector<double>& e2e
) {
    std::ofstream f(
        join_path(
            OUTPUT_DIR,
            "benchmark_samples.csv"
        )
    );

    f
        << "mode,index,latency_ms,instantaneous_fps\n";

    f << std::setprecision(12);

    for (size_t i = 0; i < mo.size(); ++i) {
        f
            << "model_only,"
            << i << ","
            << mo[i] << ","
            << 1000.0 / mo[i]
            << "\n";
    }

    for (size_t i = 0; i < e2e.size(); ++i) {
        f
            << "end_to_end,"
            << i << ","
            << e2e[i] << ","
            << 1000.0 / e2e[i]
            << "\n";
    }
}


// ============================================================
// GLOBAL METRIC BAR CHART
// ============================================================

static void save_metric_charts(
    const Metrics& global,
    const Metrics& mean
) {
    const std::vector<std::string> labels = {
        "Precision",
        "Recall",
        "F1",
        "IoU",
        "Accuracy"
    };

    cv::Mat global_chart =
        make_bar_chart(
            "HyperSTARCOP ZCU104 - Global Metrics",
            labels,
            {
                global.precision,
                global.recall,
                global.f1,
                global.iou,
                global.accuracy
            },
            "metric value"
        );

    cv::imwrite(
        join_path(
            OUTPUT_DIR,
            "metricas_globais.png"
        ),
        global_chart
    );

    cv::Mat mean_chart =
        make_bar_chart(
            "HyperSTARCOP ZCU104 - Mean Per-Image Metrics",
            labels,
            {
                mean.precision,
                mean.recall,
                mean.f1,
                mean.iou,
                mean.accuracy
            },
            "metric value"
        );

    cv::imwrite(
        join_path(
            OUTPUT_DIR,
            "metricas_media_imagens.png"
        ),
        mean_chart
    );
}


// ============================================================
// MAIN
// ============================================================

int main() {
    try {
        make_dir(
            OUTPUT_DIR
        );

        make_dir(
            FIG_DIR
        );

        if (!file_exists(XMODEL)) {
            throw std::runtime_error(
                "XModel not found: "
                + XMODEL
            );
        }

        if (!file_exists(CSV_TEST)) {
            throw std::runtime_error(
                "CSV not found: "
                + CSV_TEST
            );
        }

        std::cout
            << "\n============================================================\n"
            << "HYPERSTARCOP - ZCU104 - VART C++\n"
            << "============================================================\n";

        std::cout
            << "XModel : "
            << XMODEL
            << "\n"
            << "Dataset: "
            << DATASET
            << "\n";

        auto graph =
            xir::Graph::deserialize(
                XMODEL
            );

        auto dpu_subgraphs =
            get_dpu_subgraphs(
                graph.get()
            );

        if (dpu_subgraphs.size() != 1) {
            throw std::runtime_error(
                "Expected exactly one DPU subgraph; got "
                + std::to_string(
                    dpu_subgraphs.size()
                )
            );
        }

        auto* dpu =
            dpu_subgraphs[0];

        std::cout
            << "DPU subgraph: "
            << dpu->get_name()
            << "\n";

        auto runner =
            vart::Runner::create_runner(
                dpu,
                "run"
            );

        auto input_tensors =
            runner->get_input_tensors();

        auto output_tensors =
            runner->get_output_tensors();

        if (
            input_tensors.size() != 1 ||
            output_tensors.size() != 1
        ) {
            throw std::runtime_error(
                "Expected exactly one input and one output tensor."
            );
        }

        const xir::Tensor* input_tensor =
            input_tensors[0];

        const xir::Tensor* output_tensor =
            output_tensors[0];

        const auto in_shape =
            input_tensor->get_shape();

        const auto out_shape =
            output_tensor->get_shape();

        std::cout
            << "\nInput : "
            << input_tensor->get_name()
            << " "
            << shape_to_string(in_shape)
            << "\n";

        std::cout
            << "Output: "
            << output_tensor->get_name()
            << " "
            << shape_to_string(out_shape)
            << "\n";

        if (
            in_shape.size() != 4 ||
            in_shape[0] != 1 ||
            in_shape[1] != H ||
            in_shape[2] != W ||
            in_shape[3] != C
        ) {
            throw std::runtime_error(
                "Unexpected DPU input shape."
            );
        }

        if (
            out_shape.size() != 4 ||
            out_shape[0] != 1 ||
            out_shape[1] != H ||
            out_shape[2] != W ||
            out_shape[3] != 1
        ) {
            throw std::runtime_error(
                "Unexpected DPU output shape."
            );
        }

        const int input_fix =
            get_fix_point(
                input_tensor,
                5
            );

        const int output_fix =
            get_fix_point(
                output_tensor,
                2
            );

        const float input_scale =
            std::exp2(
                static_cast<float>(
                    input_fix
                )
            );

        const float output_scale =
            std::exp2(
                -static_cast<float>(
                    output_fix
                )
            );

        std::cout
            << "Input fix  : "
            << input_fix
            << " scale="
            << input_scale
            << "\n";

        std::cout
            << "Output fix : "
            << output_fix
            << " scale="
            << output_scale
            << "\n";

        auto rows =
            load_test_csv(
                CSV_TEST
            );

        if (rows.empty()) {
            throw std::runtime_error(
                "No test images in CSV."
            );
        }

        std::cout
            << "Test images: "
            << rows.size()
            << "\n";

        // ----------------------------------------------------
        // Shared VART buffers
        // ----------------------------------------------------

        std::vector<int8_t> input_data(
            static_cast<size_t>(
                H * W * C
            )
        );

        std::vector<int8_t> output_data(
            static_cast<size_t>(
                H * W
            )
        );

        CpuFlatTensorBuffer input_tb(
            input_data.data(),
            input_tensor
        );

        CpuFlatTensorBuffer output_tb(
            output_data.data(),
            output_tensor
        );

        std::vector<vart::TensorBuffer*> input_ptrs = {
            &input_tb
        };

        std::vector<vart::TensorBuffer*> output_ptrs = {
            &output_tb
        };

        const size_t input_bytes =
            input_tensor->get_element_num();

        const size_t output_bytes =
            output_tensor->get_element_num();

        // ----------------------------------------------------
        // VALIDATION
        // ----------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "VALIDATION - SAME METRICS AS ORIGINAL MODEL\n"
            << "============================================================\n";

        std::vector<ValidationResult> validation;

        Counts global_counts;

        Metrics mean_metrics;

        // Store already-quantized inputs for model-only benchmark.
        std::vector<std::vector<int8_t>>
            prepared_inputs;

        prepared_inputs.reserve(
            rows.size()
        );

        for (
            size_t i = 0;
            i < rows.size();
            ++i
        ) {
            const auto& row =
                rows[i];

            const std::string folder =
                join_path(
                    DATASET,
                    row.id
                );

            std::cout
                << "\n["
                << i + 1
                << "/"
                << rows.size()
                << "] "
                << row.id
                << "\n";

            RawChannels raw =
                read_input_channels(
                    folder
                );

            preprocess_to_int8_nhwc(
                raw,
                input_scale,
                input_data
            );

            prepared_inputs.push_back(
                input_data
            );

            input_tb.sync_for_write(
                0,
                input_bytes
            );

            auto job =
                runner->execute_async(
                    input_ptrs,
                    output_ptrs
                );

            const int status =
                runner->wait(
                    static_cast<int>(
                        job.first
                    ),
                    -1
                );

            if (status != 0) {
                throw std::runtime_error(
                    "VART wait failed."
                );
            }

            output_tb.sync_for_read(
                0,
                output_bytes
            );

            Prediction prediction =
                postprocess(
                    output_data,
                    output_scale
                );

            cv::Mat label =
                read_label(
                    folder
                );

            Counts counts =
                confusion(
                    prediction.pred,
                    label
                );

            Metrics metrics =
                calc_metrics(
                    counts
                );

            global_counts.tp += counts.tp;
            global_counts.fp += counts.fp;
            global_counts.fn += counts.fn;
            global_counts.tn += counts.tn;

            mean_metrics.precision +=
                metrics.precision;

            mean_metrics.recall +=
                metrics.recall;

            mean_metrics.f1 +=
                metrics.f1;

            mean_metrics.iou +=
                metrics.iou;

            mean_metrics.accuracy +=
                metrics.accuracy;

            ValidationResult vr;
            vr.id = row.id;
            vr.has_plume = row.has_plume;
            vr.real_pixels =
                count_positive(
                    label
                );
            vr.predicted_pixels =
                count_positive(
                    prediction.pred
                );
            vr.counts = counts;
            vr.metrics = metrics;
            vr.prob_min =
                prediction.prob_min;
            vr.prob_max =
                prediction.prob_max;

            validation.push_back(
                vr
            );

            std::cout
                << std::fixed
                << std::setprecision(4)
                << "  TP="
                << counts.tp
                << " FP="
                << counts.fp
                << " FN="
                << counts.fn
                << " TN="
                << counts.tn
                << "\n"
                << "  Precision="
                << metrics.precision
                << " Recall="
                << metrics.recall
                << " F1="
                << metrics.f1
                << " IoU="
                << metrics.iou
                << " Accuracy="
                << metrics.accuracy
                << "\n";

            // Save prediction/probability outside benchmark timing.
            cv::Mat pred_save;
            prediction.pred.convertTo(
                pred_save,
                CV_8U,
                255.0
            );

            cv::imwrite(
                join_path(
                    OUTPUT_DIR,
                    row.id
                    + "_prediction.png"
                ),
                pred_save
            );

            save_validation_figure(
                row.id,
                raw,
                label,
                prediction,
                counts,
                metrics
            );
        }

        const double n_images =
            static_cast<double>(
                validation.size()
            );

        mean_metrics.precision /= n_images;
        mean_metrics.recall /= n_images;
        mean_metrics.f1 /= n_images;
        mean_metrics.iou /= n_images;
        mean_metrics.accuracy /= n_images;

        const Metrics global_metrics =
            calc_metrics(
                global_counts
            );

        std::cout
            << "\n============================================================\n"
            << "GLOBAL RESULT - ALL PIXELS\n"
            << "============================================================\n";

        std::cout
            << "TP="
            << global_counts.tp
            << " FP="
            << global_counts.fp
            << " FN="
            << global_counts.fn
            << " TN="
            << global_counts.tn
            << "\n";

        std::cout
            << std::fixed
            << std::setprecision(6)
            << "Precision: "
            << global_metrics.precision
            << "\n"
            << "Recall   : "
            << global_metrics.recall
            << "\n"
            << "F1       : "
            << global_metrics.f1
            << "\n"
            << "IoU      : "
            << global_metrics.iou
            << "\n"
            << "Accuracy : "
            << global_metrics.accuracy
            << "\n";

        std::cout
            << "\nMEAN PER IMAGE\n"
            << "Precision: "
            << mean_metrics.precision
            << "\n"
            << "Recall   : "
            << mean_metrics.recall
            << "\n"
            << "F1       : "
            << mean_metrics.f1
            << "\n"
            << "IoU      : "
            << mean_metrics.iou
            << "\n"
            << "Accuracy : "
            << mean_metrics.accuracy
            << "\n";

        // Documented original reference.
        Counts ref_counts;
        ref_counts.tp = REF_TP;
        ref_counts.fp = REF_FP;
        ref_counts.fn = REF_FN;
        ref_counts.tn = REF_TN;

        const Metrics ref_metrics =
            calc_metrics(
                ref_counts
            );

        std::cout
            << "\n============================================================\n"
            << "DOCUMENTED ORIGINAL FP32 REFERENCE\n"
            << "============================================================\n"
            << "TP="
            << REF_TP
            << " FP="
            << REF_FP
            << " FN="
            << REF_FN
            << " TN="
            << REF_TN
            << "\n"
            << "Precision="
            << ref_metrics.precision
            << " Recall="
            << ref_metrics.recall
            << " F1="
            << ref_metrics.f1
            << " IoU="
            << ref_metrics.iou
            << " Accuracy="
            << ref_metrics.accuracy
            << "\n";

        save_validation_csv(
            validation
        );

        save_global_csv(
            global_counts,
            global_metrics,
            mean_metrics,
            validation.size()
        );

        save_reference_csv(
            global_metrics
        );

        save_metric_charts(
            global_metrics,
            mean_metrics
        );

        // ----------------------------------------------------
        // MODEL ONLY BENCHMARK
        // Only execute_async + wait is timed.
        // ----------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "MODEL ONLY BENCHMARK\n"
            << "Timed region: execute_async + wait ONLY\n"
            << "============================================================\n";

        for (
            int i = 0;
            i < MODEL_ONLY_WARMUP;
            ++i
        ) {
            const auto& src =
                prepared_inputs[
                    static_cast<size_t>(i)
                    % prepared_inputs.size()
                ];

            std::memcpy(
                input_data.data(),
                src.data(),
                input_data.size()
            );

            // Required sync is intentionally outside
            // the model-only timed region.
            input_tb.sync_for_write(
                0,
                input_bytes
            );

            auto job =
                runner->execute_async(
                    input_ptrs,
                    output_ptrs
                );

            runner->wait(
                static_cast<int>(
                    job.first
                ),
                -1
            );

            output_tb.sync_for_read(
                0,
                output_bytes
            );
        }

        std::vector<double>
            model_only_ms;

        model_only_ms.reserve(
            MODEL_ONLY_REPEATS
        );

        for (
            int i = 0;
            i < MODEL_ONLY_REPEATS;
            ++i
        ) {
            const auto& src =
                prepared_inputs[
                    static_cast<size_t>(i)
                    % prepared_inputs.size()
                ];

            std::memcpy(
                input_data.data(),
                src.data(),
                input_data.size()
            );

            input_tb.sync_for_write(
                0,
                input_bytes
            );

            const auto t0 =
                Clock::now();

            auto job =
                runner->execute_async(
                    input_ptrs,
                    output_ptrs
                );

            const int status =
                runner->wait(
                    static_cast<int>(
                        job.first
                    ),
                    -1
                );

            const auto t1 =
                Clock::now();

            if (status != 0) {
                throw std::runtime_error(
                    "VART model-only benchmark failed."
                );
            }

            model_only_ms.push_back(
                elapsed_ms(
                    t0,
                    t1
                )
            );

            // Also intentionally outside model-only timing.
            output_tb.sync_for_read(
                0,
                output_bytes
            );
        }

        const PerfStats model_only =
            perf_stats(
                model_only_ms
            );

        print_perf(
            "MODEL ONLY",
            model_only
        );

        // ----------------------------------------------------
        // END-TO-END BENCHMARK
        //
        // Timer includes:
        // read 4 TIFFs
        // normalization + NHWC + quantization
        // sync write
        // DPU
        // sync read
        // dequantization + sigmoid + threshold
        //
        // Excludes:
        // label, metrics, PNG, CSV
        // ----------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "END-TO-END BENCHMARK\n"
            << "TIFF -> preprocess -> INT8 -> DPU -> probability -> mask\n"
            << "============================================================\n";

        // Warmup.
        for (
            int pass = 0;
            pass < E2E_WARMUP_PASSES;
            ++pass
        ) {
            for (const auto& row : rows) {
                const std::string folder =
                    join_path(
                        DATASET,
                        row.id
                    );

                RawChannels raw =
                    read_input_channels(
                        folder
                    );

                preprocess_to_int8_nhwc(
                    raw,
                    input_scale,
                    input_data
                );

                input_tb.sync_for_write(
                    0,
                    input_bytes
                );

                auto job =
                    runner->execute_async(
                        input_ptrs,
                        output_ptrs
                    );

                runner->wait(
                    static_cast<int>(
                        job.first
                    ),
                    -1
                );

                output_tb.sync_for_read(
                    0,
                    output_bytes
                );

                Prediction tmp =
                    postprocess(
                        output_data,
                        output_scale
                    );

                volatile uint8_t warmup_sink =
                    tmp.pred.at<uint8_t>(0, 0);

                (void)warmup_sink;
            }
        }

        std::vector<double> e2e_ms;

        e2e_ms.reserve(
            static_cast<size_t>(
                E2E_MEASURE_PASSES
            )
            * rows.size()
        );

        for (
            int pass = 0;
            pass < E2E_MEASURE_PASSES;
            ++pass
        ) {
            for (const auto& row : rows) {
                const std::string folder =
                    join_path(
                        DATASET,
                        row.id
                    );

                const auto t0 =
                    Clock::now();

                RawChannels raw =
                    read_input_channels(
                        folder
                    );

                preprocess_to_int8_nhwc(
                    raw,
                    input_scale,
                    input_data
                );

                input_tb.sync_for_write(
                    0,
                    input_bytes
                );

                auto job =
                    runner->execute_async(
                        input_ptrs,
                        output_ptrs
                    );

                const int status =
                    runner->wait(
                        static_cast<int>(
                            job.first
                        ),
                        -1
                    );

                if (status != 0) {
                    throw std::runtime_error(
                        "VART E2E benchmark failed."
                    );
                }

                output_tb.sync_for_read(
                    0,
                    output_bytes
                );

                Prediction prediction =
                    postprocess(
                        output_data,
                        output_scale
                    );

                // Prevent optimizer from considering
                // postprocess result unused.
                volatile uint8_t sink =
                    prediction.pred.at<uint8_t>(
                        0,
                        0
                    );

                (void)sink;

                const auto t1 =
                    Clock::now();

                e2e_ms.push_back(
                    elapsed_ms(
                        t0,
                        t1
                    )
                );
            }
        }

        const PerfStats e2e =
            perf_stats(
                e2e_ms
            );

        print_perf(
            "END TO END",
            e2e
        );

        // ----------------------------------------------------
        // SAVE PERFORMANCE OUTPUTS
        // ----------------------------------------------------

        save_benchmark_summary_csv(
            model_only,
            e2e
        );

        save_raw_benchmark_csv(
            model_only_ms,
            e2e_ms
        );

        save_perf_charts(
            "model_only",
            "Model Only",
            model_only
        );

        save_perf_charts(
            "end_to_end",
            "End-to-End",
            e2e
        );

        cv::Mat avg_fps_chart =
            make_bar_chart(
                "Average Throughput - HyperSTARCOP ZCU104",
                {
                    "Model Only",
                    "End-to-End"
                },
                {
                    model_only.fps_avg,
                    e2e.fps_avg
                },
                "FPS"
            );

        cv::imwrite(
            join_path(
                OUTPUT_DIR,
                "throughput_average_fps.png"
            ),
            avg_fps_chart
        );

        // ----------------------------------------------------
        // FINAL SUMMARY
        // ----------------------------------------------------

        std::cout
            << "\n\n============================================================\n"
            << "FINAL SUMMARY\n"
            << "============================================================\n";

        std::cout
            << std::fixed
            << std::setprecision(6)
            << "\nSEGMENTATION\n"
            << "Precision = "
            << global_metrics.precision
            << "\nRecall    = "
            << global_metrics.recall
            << "\nF1        = "
            << global_metrics.f1
            << "\nIoU       = "
            << global_metrics.iou
            << "\nAccuracy  = "
            << global_metrics.accuracy
            << "\n";

        std::cout
            << std::setprecision(4)
            << "\nMODEL ONLY\n"
            << "Average FPS = "
            << model_only.fps_avg
            << "\nMin FPS     = "
            << model_only.fps_min
            << "\nMax FPS     = "
            << model_only.fps_max
            << "\nP95 FPS     = "
            << model_only.fps_p95
            << "\nP99 FPS     = "
            << model_only.fps_p99
            << "\nAvg latency = "
            << model_only.latency_avg_ms
            << " ms\n"
            << "Min latency = "
            << model_only.latency_min_ms
            << " ms\n"
            << "Max latency = "
            << model_only.latency_max_ms
            << " ms\n"
            << "P95 latency = "
            << model_only.latency_p95_ms
            << " ms\n"
            << "P99 latency = "
            << model_only.latency_p99_ms
            << " ms\n";

        std::cout
            << "\nEND TO END\n"
            << "Average FPS = "
            << e2e.fps_avg
            << "\nMin FPS     = "
            << e2e.fps_min
            << "\nMax FPS     = "
            << e2e.fps_max
            << "\nP95 FPS     = "
            << e2e.fps_p95
            << "\nP99 FPS     = "
            << e2e.fps_p99
            << "\nAvg latency = "
            << e2e.latency_avg_ms
            << " ms\n"
            << "Min latency = "
            << e2e.latency_min_ms
            << " ms\n"
            << "Max latency = "
            << e2e.latency_max_ms
            << " ms\n"
            << "P95 latency = "
            << e2e.latency_p95_ms
            << " ms\n"
            << "P99 latency = "
            << e2e.latency_p99_ms
            << " ms\n";

        std::cout
            << "\nResults saved at:\n"
            << OUTPUT_DIR
            << "\n";

        std::cout
            << "\nMain files:\n"
            << "  metricas_por_imagem.csv\n"
            << "  metricas_globais.csv\n"
            << "  comparacao_referencia_original.csv\n"
            << "  benchmark_summary.csv\n"
            << "  benchmark_samples.csv\n"
            << "  metricas_globais.png\n"
            << "  metricas_media_imagens.png\n"
            << "  model_only_latency.png\n"
            << "  model_only_fps.png\n"
            << "  end_to_end_latency.png\n"
            << "  end_to_end_fps.png\n"
            << "  throughput_average_fps.png\n"
            << "  figuras_por_imagem/*.png\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\nERROR: "
            << e.what()
            << "\n";

        return 1;
    }
}
