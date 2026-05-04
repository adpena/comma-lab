#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint8_t SENTINEL = 5;
constexpr int CTX9_COUNT = 10077696;
constexpr int CLASS_SYMS = 5;
constexpr uint32_t TOP = 0xFFFFFFFFu;
constexpr uint32_t HALF = 0x80000000u;
constexpr uint32_t FIRST_QTR = 0x40000000u;
constexpr uint32_t THIRD_QTR = 0xC0000000u;
constexpr uint32_t SCALE_TOTAL = 65535;

struct AdaptiveModel9Binary {
  std::vector<std::array<uint16_t, 2>> prev_freq;
  std::vector<std::array<uint16_t, 2>> left_freq;
  std::vector<std::array<uint16_t, 2>> up_freq;
  std::vector<std::array<uint16_t, CLASS_SYMS>> class_freq;

  AdaptiveModel9Binary()
      : prev_freq(CTX9_COUNT), left_freq(CTX9_COUNT), up_freq(CTX9_COUNT), class_freq(CTX9_COUNT) {
    for (int ctx = 0; ctx < CTX9_COUNT; ctx++) {
      int v = ctx;
      uint8_t left2 = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t up2 = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t pd = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t pr = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t ur = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t ul = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t up = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t left = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t prev = static_cast<uint8_t>(v % 6);
      (void)left2; (void)up2; (void)pd; (void)pr; (void)ur; (void)ul;

      prev_freq[ctx] = {1, 3};
      left_freq[ctx] = {1, 4};
      up_freq[ctx] = {1, 3};
      class_freq[ctx].fill(1);
      if (up == SENTINEL) up_freq[ctx] = {60000, 1};
      if (left == SENTINEL || left == up) left_freq[ctx] = {60000, 1};
      if (prev == SENTINEL || prev == up || prev == left) prev_freq[ctx] = {60000, 1};
      for (uint8_t cls = 0; cls < CLASS_SYMS; cls++) {
        if (cls != up && cls != left && cls != prev) class_freq[ctx][cls] = 3;
      }
    }
  }
};

struct BitReader {
  const std::vector<uint8_t>& bytes;
  size_t pos = 0;
  int left = 0;
  uint8_t cur = 0;
  explicit BitReader(const std::vector<uint8_t>& data) : bytes(data) {}
  int bit() {
    if (left == 0) {
      cur = pos < bytes.size() ? bytes[pos++] : 0;
      left = 8;
    }
    int b = (cur >> 7) & 1;
    cur <<= 1;
    left--;
    return b;
  }
  uint64_t bits_consumed() const { return static_cast<uint64_t>(pos) * 8u - static_cast<uint64_t>(left); }
};

struct ArithmeticDecoder {
  uint32_t low = 0;
  uint32_t high = TOP;
  uint32_t value = 0;
  BitReader reader;
  explicit ArithmeticDecoder(const std::vector<uint8_t>& data) : reader(data) {
    for (int i = 0; i < 32; i++) value = (value << 1) | reader.bit();
  }
  uint32_t scaled(uint32_t total) const {
    uint64_t range = static_cast<uint64_t>(high) - low + 1ull;
    return static_cast<uint32_t>((((static_cast<uint64_t>(value) - low + 1ull) * total) - 1ull) / range);
  }
  void update(uint32_t cum_low, uint32_t cum_high, uint32_t total) {
    uint64_t range = static_cast<uint64_t>(high) - low + 1ull;
    uint32_t old_low = low;
    high = static_cast<uint32_t>(old_low + (range * cum_high) / total - 1ull);
    low = static_cast<uint32_t>(old_low + (range * cum_low) / total);
    while (true) {
      if (high < HALF) {
      } else if (low >= HALF) {
        value -= HALF; low -= HALF; high -= HALF;
      } else if (low >= FIRST_QTR && high < THIRD_QTR) {
        value -= FIRST_QTR; low -= FIRST_QTR; high -= FIRST_QTR;
      } else {
        break;
      }
      low <<= 1;
      high = (high << 1) | 1u;
      value = (value << 1) | static_cast<uint32_t>(reader.bit());
    }
  }
};

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("failed to open input: " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

uint32_t read_u32(const std::vector<uint8_t>& data, size_t& off) {
  if (off + 4 > data.size()) throw std::runtime_error("truncated u32");
  uint32_t v = static_cast<uint32_t>(data[off]) |
               (static_cast<uint32_t>(data[off + 1]) << 8) |
               (static_cast<uint32_t>(data[off + 2]) << 16) |
               (static_cast<uint32_t>(data[off + 3]) << 24);
  off += 4;
  return v;
}

uint8_t get_prev(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int w, int xcoord) {
  if (t == 0) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y) * w + xcoord];
}
uint8_t get_left(const std::vector<uint8_t>& decoded, size_t base, int xcoord) {
  if (xcoord == 0) return SENTINEL;
  return decoded[base + static_cast<size_t>(xcoord - 1)];
}
uint8_t get_up(const std::vector<uint8_t>& decoded, size_t base, int y, int w, int xcoord) {
  if (y == 0) return SENTINEL;
  return decoded[base - static_cast<size_t>(w) + xcoord];
}
uint8_t get_up_left(const std::vector<uint8_t>& decoded, size_t base, int y, int w, int xcoord) {
  if (y == 0 || xcoord == 0) return SENTINEL;
  return decoded[base - static_cast<size_t>(w) + xcoord - 1];
}
uint8_t get_up_right(const std::vector<uint8_t>& decoded, size_t base, int y, int w, int xcoord) {
  if (y == 0 || xcoord + 1 >= w) return SENTINEL;
  return decoded[base - static_cast<size_t>(w) + xcoord + 1];
}
uint8_t get_left2(const std::vector<uint8_t>& decoded, size_t base, int xcoord) {
  if (xcoord < 2) return SENTINEL;
  return decoded[base + static_cast<size_t>(xcoord - 2)];
}
uint8_t get_up2(const std::vector<uint8_t>& decoded, size_t base, int y, int w, int xcoord) {
  if (y < 2) return SENTINEL;
  return decoded[base - static_cast<size_t>(2 * w) + xcoord];
}
uint8_t get_prev_right(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int w, int xcoord) {
  if (t == 0 || xcoord + 1 >= w) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y) * w + xcoord + 1];
}
uint8_t get_prev_down(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int h, int w, int xcoord) {
  if (t == 0 || y + 1 >= h) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y + 1) * w + xcoord];
}

int ctx9_id(uint8_t prev, uint8_t left, uint8_t up, uint8_t up_left, uint8_t up_right, uint8_t prev_right, uint8_t prev_down, uint8_t up2, uint8_t left2) {
  int ctx = prev;
  for (uint8_t value : {left, up, up_left, up_right, prev_right, prev_down, up2, left2}) {
    ctx = ctx * 6 + static_cast<int>(value);
  }
  return ctx;
}

std::array<uint8_t, 9> decode_ctx(int ctx) {
  std::array<uint8_t, 9> out{};
  for (int i = 8; i >= 0; --i) {
    out[static_cast<size_t>(i)] = static_cast<uint8_t>(ctx % 6);
    ctx /= 6;
  }
  return out;
}

template <size_t N>
void update_adaptive(std::array<uint16_t, N>& freq, uint32_t sym) {
  uint32_t total = 0;
  for (uint32_t i = 0; i < N; i++) total += freq[i];
  if (total >= SCALE_TOTAL) {
    for (uint32_t i = 0; i < N; i++) freq[i] = static_cast<uint16_t>(std::max<uint32_t>(1, (freq[i] + 1) >> 1));
  }
  freq[sym] = static_cast<uint16_t>(std::min<uint32_t>(65535, static_cast<uint32_t>(freq[sym]) + 20));
}

template <size_t N>
uint32_t decode_symbol(ArithmeticDecoder& dec, const std::array<uint16_t, N>& freq, double* model_bits) {
  uint32_t total = 0;
  for (uint32_t i = 0; i < N; i++) total += freq[i];
  uint32_t v = dec.scaled(total);
  uint32_t cum = 0;
  for (uint32_t sym = 0; sym < N; sym++) {
    uint32_t next = cum + freq[sym];
    if (v < next) {
      dec.update(cum, next, total);
      if (model_bits) *model_bits = -std::log2(static_cast<double>(freq[sym]) / static_cast<double>(total));
      return sym;
    }
    cum = next;
  }
  throw std::runtime_error("decode_symbol out of range");
}

void json_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (char c : value) {
    if (c == '\\' || c == '"') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else out << c;
  }
  out << '"';
}

struct TopContext {
  uint32_t hits;
  float bits;
  int ctx;
};

void decode_profile(const std::vector<uint8_t>& packed, const std::string& out_path) {
  if (packed.size() < 20 || packed[0] != 'Q' || packed[1] != 'M' || packed[2] != 'A' || packed[3] != '9') {
    throw std::runtime_error("not a QMA9 range-mask payload");
  }
  size_t off = 4;
  int t_count = static_cast<int>(read_u32(packed, off));
  int h = static_cast<int>(read_u32(packed, off));
  int w = static_cast<int>(read_u32(packed, off));
  uint32_t bit_bytes = read_u32(packed, off);
  if (off + bit_bytes > packed.size()) throw std::runtime_error("truncated bitstream");
  std::vector<uint8_t> bits(packed.begin() + static_cast<std::ptrdiff_t>(off), packed.begin() + static_cast<std::ptrdiff_t>(off + bit_bytes));

  AdaptiveModel9Binary model;
  ArithmeticDecoder dec(bits);
  const size_t frame_size = static_cast<size_t>(h) * w;
  const uint64_t pixels = static_cast<uint64_t>(t_count) * frame_size;
  std::vector<uint8_t> decoded(static_cast<size_t>(pixels), 0);
  std::vector<uint32_t> ctx_hits(CTX9_COUNT, 0);
  std::vector<float> ctx_bits(CTX9_COUNT, 0.0f);
  std::vector<double> frame_bits(static_cast<size_t>(t_count), 0.0);
  std::vector<double> row_bits(static_cast<size_t>(h), 0.0);
  std::array<uint64_t, CLASS_SYMS> class_counts{};
  std::array<double, CLASS_SYMS> class_bits{};
  std::array<uint64_t, 4> predictor_counts{};
  std::array<uint64_t, 4> stage_counts{};
  std::array<double, 4> stage_bits{};
  std::array<uint64_t, 8> run_buckets{};
  std::array<uint64_t, 6> fallback_extra_matches{};
  uint64_t fallback_events = 0;
  uint64_t horizontal_runs = 0;
  uint64_t horizontal_run_pixels = 0;
  uint32_t max_horizontal_run = 0;

  auto add_stage = [&](int stage, int ctx, int cls, double bits_value, int frame, int row) {
    stage_counts[static_cast<size_t>(stage)]++;
    stage_bits[static_cast<size_t>(stage)] += bits_value;
    frame_bits[static_cast<size_t>(frame)] += bits_value;
    row_bits[static_cast<size_t>(row)] += bits_value;
    class_bits[static_cast<size_t>(cls)] += bits_value;
    ctx_bits[static_cast<size_t>(ctx)] += static_cast<float>(bits_value);
  };

  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      uint8_t run_cls = 255;
      uint32_t run_len = 0;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(decoded, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        uint8_t ul = get_up_left(decoded, base, y, w, xcoord);
        uint8_t ur = get_up_right(decoded, base, y, w, xcoord);
        uint8_t pr = get_prev_right(decoded, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(decoded, frame_size, t, y, h, w, xcoord);
        uint8_t u2 = get_up2(decoded, base, y, w, xcoord);
        uint8_t l2 = get_left2(decoded, base, xcoord);
        int ctx = ctx9_id(prev, left, up, ul, ur, pr, pd, u2, l2);
        ctx_hits[static_cast<size_t>(ctx)]++;
        uint8_t cls = 0;
        int predictor = 3;
        double mb = 0.0;
        uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.up_freq[ctx], &mb));
        if (b) cls = up;
        add_stage(0, ctx, cls, mb, t, y);
        update_adaptive<2>(model.up_freq[ctx], b);
        if (b) {
          predictor = 0;
        } else {
          b = static_cast<uint8_t>(decode_symbol<2>(dec, model.left_freq[ctx], &mb));
          if (b) cls = left;
          add_stage(1, ctx, cls, mb, t, y);
          update_adaptive<2>(model.left_freq[ctx], b);
          if (b) {
            predictor = 1;
          } else {
            b = static_cast<uint8_t>(decode_symbol<2>(dec, model.prev_freq[ctx], &mb));
            if (b) cls = prev;
            add_stage(2, ctx, cls, mb, t, y);
            update_adaptive<2>(model.prev_freq[ctx], b);
            if (b) {
              predictor = 2;
            } else {
              cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx], &mb));
              add_stage(3, ctx, cls, mb, t, y);
              update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
              predictor = 3;
              fallback_events++;
              std::array<uint8_t, 6> extras = {ul, ur, pr, pd, u2, l2};
              for (size_t i = 0; i < extras.size(); ++i) {
                if (extras[i] < CLASS_SYMS && extras[i] == cls && extras[i] != up && extras[i] != left && extras[i] != prev) {
                  fallback_extra_matches[i]++;
                }
              }
            }
          }
        }
        decoded[base + static_cast<size_t>(xcoord)] = cls;
        class_counts[static_cast<size_t>(cls)]++;
        predictor_counts[static_cast<size_t>(predictor)]++;
        if (run_len == 0 || cls == run_cls) {
          run_cls = cls;
          run_len++;
        } else {
          horizontal_runs++;
          horizontal_run_pixels += run_len;
          max_horizontal_run = std::max(max_horizontal_run, run_len);
          size_t bucket = run_len == 1 ? 0 : run_len == 2 ? 1 : run_len == 3 ? 2 : run_len < 8 ? 3 : run_len < 16 ? 4 : run_len < 32 ? 5 : run_len < 64 ? 6 : 7;
          run_buckets[bucket]++;
          run_cls = cls;
          run_len = 1;
        }
      }
      if (run_len) {
        horizontal_runs++;
        horizontal_run_pixels += run_len;
        max_horizontal_run = std::max(max_horizontal_run, run_len);
        size_t bucket = run_len == 1 ? 0 : run_len == 2 ? 1 : run_len == 3 ? 2 : run_len < 8 ? 3 : run_len < 16 ? 4 : run_len < 32 ? 5 : run_len < 64 ? 6 : 7;
        run_buckets[bucket]++;
      }
    }
  }

  std::vector<TopContext> top;
  top.reserve(CTX9_COUNT / 16);
  for (int ctx = 0; ctx < CTX9_COUNT; ++ctx) {
    if (ctx_hits[static_cast<size_t>(ctx)] > 0) {
      top.push_back({ctx_hits[static_cast<size_t>(ctx)], ctx_bits[static_cast<size_t>(ctx)], ctx});
    }
  }
  std::sort(top.begin(), top.end(), [](const TopContext& a, const TopContext& b) {
    if (a.hits != b.hits) return a.hits > b.hits;
    return a.bits > b.bits;
  });

  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("failed to open output: " + out_path);
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"schema\": \"qma9_range_mask_cpp_full_profile_v1\",\n";
  out << "  \"implementation\": \"experiments/qma9_range_mask_cpp_profiler.cpp\",\n";
  out << "  \"header\": {\"frame_count\": " << t_count << ", \"height\": " << h << ", \"width\": " << w << ", \"decoded_pixels\": " << pixels << ", \"bitstream_bytes\": " << bit_bytes << "},\n";
  out << "  \"arithmetic_final_state\": {\"low\": " << dec.low << ", \"high\": " << dec.high << ", \"value\": " << dec.value << ", \"bits_consumed\": " << dec.reader.bits_consumed() << "},\n";
  out << "  \"estimated_model_bits\": " << (stage_bits[0] + stage_bits[1] + stage_bits[2] + stage_bits[3]) << ",\n";
  out << "  \"actual_bitstream_bits\": " << (static_cast<uint64_t>(bit_bytes) * 8ull) << ",\n";
  out << "  \"stage_counts\": {\"up_gate\": " << stage_counts[0] << ", \"left_gate\": " << stage_counts[1] << ", \"prev_gate\": " << stage_counts[2] << ", \"class_fallback\": " << stage_counts[3] << "},\n";
  out << "  \"stage_estimated_bits\": {\"up_gate\": " << stage_bits[0] << ", \"left_gate\": " << stage_bits[1] << ", \"prev_gate\": " << stage_bits[2] << ", \"class_fallback\": " << stage_bits[3] << "},\n";
  out << "  \"predictor_counts\": {\"up\": " << predictor_counts[0] << ", \"left\": " << predictor_counts[1] << ", \"prev\": " << predictor_counts[2] << ", \"class_fallback\": " << predictor_counts[3] << "},\n";
  out << "  \"class_counts\": {\"0\": " << class_counts[0] << ", \"1\": " << class_counts[1] << ", \"2\": " << class_counts[2] << ", \"3\": " << class_counts[3] << ", \"4\": " << class_counts[4] << "},\n";
  out << "  \"class_estimated_bits\": {\"0\": " << class_bits[0] << ", \"1\": " << class_bits[1] << ", \"2\": " << class_bits[2] << ", \"3\": " << class_bits[3] << ", \"4\": " << class_bits[4] << "},\n";
  out << "  \"run_structure\": {\"horizontal_runs\": " << horizontal_runs << ", \"horizontal_run_pixels\": " << horizontal_run_pixels << ", \"max_horizontal_run\": " << max_horizontal_run << ", \"buckets\": {\"1\": " << run_buckets[0] << ", \"2\": " << run_buckets[1] << ", \"3\": " << run_buckets[2] << ", \"4_7\": " << run_buckets[3] << ", \"8_15\": " << run_buckets[4] << ", \"16_31\": " << run_buckets[5] << ", \"32_63\": " << run_buckets[6] << ", \"64_plus\": " << run_buckets[7] << "}},\n";
  out << "  \"fallback_extra_predictor_matches\": {\"up_left\": " << fallback_extra_matches[0] << ", \"up_right\": " << fallback_extra_matches[1] << ", \"prev_right\": " << fallback_extra_matches[2] << ", \"prev_down\": " << fallback_extra_matches[3] << ", \"up2\": " << fallback_extra_matches[4] << ", \"left2\": " << fallback_extra_matches[5] << ", \"fallback_events\": " << fallback_events << "},\n";
  out << "  \"per_frame_estimated_bytes\": [";
  for (size_t i = 0; i < frame_bits.size(); ++i) {
    if (i) out << ", ";
    out << (frame_bits[i] / 8.0);
  }
  out << "],\n";
  out << "  \"per_row_index_estimated_bytes\": [";
  for (size_t i = 0; i < row_bits.size(); ++i) {
    if (i) out << ", ";
    out << (row_bits[i] / 8.0);
  }
  out << "],\n";
  out << "  \"top_contexts\": [";
  const size_t limit = std::min<size_t>(32, top.size());
  for (size_t i = 0; i < limit; ++i) {
    if (i) out << ", ";
    auto digits = decode_ctx(top[i].ctx);
    out << "{\"context_id\": " << top[i].ctx << ", \"pixels\": " << top[i].hits << ", \"estimated_bits\": " << top[i].bits << ", \"digits\": [";
    for (size_t j = 0; j < digits.size(); ++j) {
      if (j) out << ", ";
      out << static_cast<int>(digits[j]);
    }
    out << "]}";
  }
  out << "]\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: qma9_range_mask_cpp_profiler <range_mask.qma9> <out.json>\n";
    return 2;
  }
  try {
    decode_profile(read_file(argv[1]), argv[2]);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 7;
  }
}
