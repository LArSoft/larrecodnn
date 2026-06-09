#include "art/Utilities/ToolMacros.h"
#include "fhiclcpp/ParameterSet.h"
#include "larrecodnn/ImagePatternAlgs/ToolInterfaces/IWireframeRecog.h"
#include "messagefacility/MessageLogger/MessageLogger.h"
#include "cetlib_except/exception.h"

#include "grpc_client.h"
#include "common.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace wframerec_tool {

  namespace tc = triton::client;

  class WireframeRecogTriton : public IWireframeRecog {
  public:
    explicit WireframeRecogTriton(const fhicl::ParameterSet& pset);

    std::vector<std::vector<float>> predictWireframeType(
							 const std::vector<std::vector<std::vector<short>>>& wireframes) const override;

  private:
    std::unique_ptr<tc::InferenceServerGrpcClient> makeClient() const;

    static void throwIfError(const tc::Error& err, const std::string& where)
    {
      if (!err.IsOk()) {
        throw cet::exception("WireframeRecogTriton")
          << where << " failed: " << err.Message();
      }
    }

    static long elapsedMs(const std::chrono::steady_clock::time_point& t0)
    {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
								   std::chrono::steady_clock::now() - t0).count();
    }

  private:
    std::string fTritonModelName;
    std::string fTritonURL;
    bool fTritonVerbose{false};
    std::string fTritonModelVersion;
    unsigned fTritonTimeout{10000};

    std::string fInputName;
    std::vector<std::string> fOutputNames;

    std::string fAuthTokenEnvVar{"DUNE_TRITON_TOKEN"};
    bool fRequireAuthToken{false};

    bool fTritonSSL{false};
    std::string fTritonSSLRootCertificates;
    std::string fTritonSSLPrivateKey;
    std::string fTritonSSLCertificateChain;

    bool fLogTiming{false};

    mutable std::unique_ptr<tc::InferenceServerGrpcClient> fClient;
    mutable std::string fAuthHeaderValue;
  };

  std::unique_ptr<tc::InferenceServerGrpcClient>
  WireframeRecogTriton::makeClient() const
  {
    std::unique_ptr<tc::InferenceServerGrpcClient> client;

    tc::SslOptions ssl_options;
    ssl_options.root_certificates = fTritonSSLRootCertificates;
    ssl_options.private_key = fTritonSSLPrivateKey;
    ssl_options.certificate_chain = fTritonSSLCertificateChain;

    tc::KeepAliveOptions keepalive_options;

    throwIfError(
		 tc::InferenceServerGrpcClient::Create(
						       &client,
						       fTritonURL,
						       fTritonVerbose,
						       fTritonSSL,
						       ssl_options,
						       keepalive_options),
		 "InferenceServerGrpcClient::Create");

    return client;
  }

  WireframeRecogTriton::WireframeRecogTriton(const fhicl::ParameterSet& pset)
  {
    fOutputNames = pset.get<std::vector<std::string>>("OutputNames");
    fInputName = pset.get<std::string>("InputName", "");
    fTritonModelName = pset.get<std::string>("TritonModelName");
    fTritonURL = pset.get<std::string>("TritonURL", "localhost:8001");
    fTritonVerbose = pset.get<bool>("TritonVerbose", false);
    fTritonModelVersion = pset.get<std::string>("TritonModelVersion", "");
    fTritonTimeout = pset.get<unsigned>("TritonTimeout", 10000);

    fAuthTokenEnvVar = pset.get<std::string>("AuthTokenEnvVar", "DUNE_TRITON_TOKEN");
    fRequireAuthToken = pset.get<bool>("RequireAuthToken", false);

    fTritonSSL = pset.get<bool>("TritonSSL", false);
    fTritonSSLRootCertificates = pset.get<std::string>("TritonSSLRootCertificates", "");
    fTritonSSLPrivateKey = pset.get<std::string>("TritonSSLPrivateKey", "");
    fTritonSSLCertificateChain = pset.get<std::string>("TritonSSLCertificateChain", "");

    fLogTiming = pset.get<bool>("LogTiming", false);

    if (fTritonModelName.empty()) {
      throw cet::exception("WireframeRecogTriton")
        << "TritonModelName must be provided.";
    }

    if (fOutputNames.empty()) {
      throw cet::exception("WireframeRecogTriton")
        << "OutputNames must be non-empty.";
    }

    if (!fAuthTokenEnvVar.empty()) {
      const char* token_env = std::getenv(fAuthTokenEnvVar.c_str());
      if (token_env != nullptr) {
        const std::string token(token_env);
        if (!token.empty()) {
          fAuthHeaderValue = "Bearer " + token;
        }
        else if (fRequireAuthToken) {
          throw cet::exception("WireframeRecogTriton")
            << "Required auth token environment variable '"
            << fAuthTokenEnvVar << "' is set but empty.";
        }
      }
      else if (fRequireAuthToken) {
        throw cet::exception("WireframeRecogTriton")
          << "Required auth token environment variable '"
          << fAuthTokenEnvVar << "' is not set.";
      }
    }

    fClient = makeClient();

    bool live = false;
    throwIfError(fClient->IsServerLive(&live), "IsServerLive");
    if (!live) {
      throw cet::exception("WireframeRecogTriton")
        << "Triton server at " << fTritonURL << " is not live.";
    }

    mf::LogInfo("WireframeRecogTriton")
      << "Bare Triton gRPC inference client created."
      << "\n url: " << fTritonURL
      << "\n model: " << fTritonModelName
      << "\n ssl: " << (fTritonSSL ? "true" : "false");

    setupWframeRecRoiParams(pset);
  }

  std::vector<std::vector<float>>
  WireframeRecogTriton::predictWireframeType(
    const std::vector<std::vector<std::vector<short>>>& wireframes) const
  {
    if (wireframes.empty() ||
        wireframes.front().empty() ||
        wireframes.front().front().empty()) {
      return {};
    }

    const auto t_total = std::chrono::steady_clock::now();

    const std::size_t samples = wireframes.size();
    const std::size_t rows = wireframes.front().size();
    const std::size_t cols = wireframes.front().front().size();

    for (std::size_t s = 0; s < samples; ++s) {
      if (wireframes[s].size() != rows) {
        throw cet::exception("WireframeRecogTriton")
          << "Inconsistent rows across samples.";
      }

      for (std::size_t r = 0; r < rows; ++r) {
        if (wireframes[s][r].size() != cols) {
          throw cet::exception("WireframeRecogTriton")
            << "Inconsistent cols across samples.";
        }
      }
    }

    std::vector<float> input_buffer;
    input_buffer.reserve(samples * rows * cols);

    for (std::size_t s = 0; s < samples; ++s) {
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
          input_buffer.push_back(static_cast<float>(wireframes[s][r][c]));
        }
      }
    }

    const size_t input_bytes = input_buffer.size() * sizeof(float);
    const double input_MB = input_bytes / 1024.0 / 1024.0;

    std::vector<int64_t> input_shape{
      static_cast<int64_t>(samples),
      static_cast<int64_t>(rows),
      static_cast<int64_t>(cols),
      1
    };

    const std::string& input_name = fInputName;

    tc::InferInput* input_raw = nullptr;
    throwIfError(
      tc::InferInput::Create(&input_raw, input_name, input_shape, "FP32"),
      "InferInput::Create");

    std::unique_ptr<tc::InferInput> input(input_raw);

    throwIfError(input->Reset(), "InferInput::Reset");

    throwIfError(
      input->AppendRaw(
        reinterpret_cast<const uint8_t*>(input_buffer.data()),
        input_bytes),
      "InferInput::AppendRaw");

    std::vector<std::unique_ptr<tc::InferRequestedOutput>> output_ptrs;
    std::vector<const tc::InferRequestedOutput*> outputs;
    output_ptrs.reserve(fOutputNames.size());
    outputs.reserve(fOutputNames.size());

    for (const auto& oname : fOutputNames) {
      tc::InferRequestedOutput* output_raw = nullptr;
      throwIfError(
        tc::InferRequestedOutput::Create(&output_raw, oname),
        "InferRequestedOutput::Create(" + oname + ")");

      output_ptrs.emplace_back(output_raw);
      outputs.push_back(output_ptrs.back().get());
    }

    tc::InferOptions options(fTritonModelName);
    options.model_version_ = fTritonModelVersion;

    // Triton C++ client expects timeout in microseconds.
    // FHiCL TritonTimeout is treated as milliseconds.
    options.client_timeout_ = static_cast<uint64_t>(fTritonTimeout) * 1000;

    tc::Headers headers;
    if (!fAuthHeaderValue.empty()) {
      headers["Authorization"] = fAuthHeaderValue;
    }

    std::vector<tc::InferInput*> inputs{input.get()};

    tc::InferResult* raw_result = nullptr;
    const auto t_infer = std::chrono::steady_clock::now();

    throwIfError(
      fClient->Infer(&raw_result, options, inputs, outputs, headers),
      "Infer");

    const long infer_ms = elapsedMs(t_infer);

    std::unique_ptr<tc::InferResult> result(raw_result);

    if (fLogTiming) {
      mf::LogInfo("WireframeRecogTriton")
        << "Triton inference timing"
        << "\n model: " << fTritonModelName
        << "\n samples: " << samples
        << "\n rows: " << rows
        << "\n cols: " << cols
        << "\n request_MB: " << input_MB
        << "\n infer_wall_ms: " << infer_ms;
    }

    struct OutputView {
      std::vector<std::vector<float>> values;
    };

    std::vector<OutputView> views;
    views.reserve(fOutputNames.size());

    size_t total_response_bytes = 0;

    for (const auto& oname : fOutputNames) {
      std::vector<int64_t> out_shape;
      throwIfError(
        result->Shape(oname, &out_shape),
        "InferResult::Shape(" + oname + ")");

      const uint8_t* buf = nullptr;
      size_t byte_size = 0;
      throwIfError(
        result->RawData(oname, &buf, &byte_size),
        "InferResult::RawData(" + oname + ")");

      total_response_bytes += byte_size;

      if ((byte_size % sizeof(float)) != 0) {
        throw cet::exception("WireframeRecogTriton")
          << "Output " << oname << " byte size " << byte_size
          << " is not divisible by sizeof(float).";
      }

      const size_t total_floats = byte_size / sizeof(float);
      const float* fbuf = reinterpret_cast<const float*>(buf);

      if (out_shape.empty()) {
        throw cet::exception("WireframeRecogTriton")
          << "Output " << oname << " returned empty shape.";
      }

      if (static_cast<std::size_t>(out_shape[0]) != samples) {
        throw cet::exception("WireframeRecogTriton")
          << "Output " << oname << " returned batch dimension "
          << out_shape[0] << "; expected " << samples;
      }

      size_t per_sample = 1;
      for (std::size_t i = 1; i < out_shape.size(); ++i) {
        per_sample *= static_cast<size_t>(out_shape[i]);
      }

      if (per_sample * samples != total_floats) {
        std::ostringstream os;
        os << "Output " << oname
           << " shape/product mismatch: total_floats=" << total_floats
           << " batch=" << samples
           << " per_sample=" << per_sample;
        throw cet::exception("WireframeRecogTriton") << os.str();
      }

      OutputView view;
      view.values.resize(samples);

      for (std::size_t s = 0; s < samples; ++s) {
        const float* begin = fbuf + s * per_sample;
        const float* end = begin + per_sample;
        view.values[s].assign(begin, end);
      }

      views.push_back(std::move(view));
    }

    std::vector<std::vector<float>> out;
    out.reserve(samples);

    for (std::size_t s = 0; s < samples; ++s) {
      std::size_t total = 0;
      for (const auto& vw : views) {
        total += vw.values[s].size();
      }

      out.emplace_back();
      auto& v = out.back();
      v.reserve(total);

      for (const auto& vw : views) {
        v.insert(v.end(), vw.values[s].begin(), vw.values[s].end());
      }
    }

    if (fLogTiming) {
      mf::LogInfo("WireframeRecogTriton")
        << "Triton inference completed"
        << "\n model: " << fTritonModelName
        << "\n response_MB: "
        << (total_response_bytes / 1024.0 / 1024.0)
        << "\n total_wall_ms: " << elapsedMs(t_total);
    }

    return out;
  }

} // namespace wframerec_tool

DEFINE_ART_CLASS_TOOL(wframerec_tool::WireframeRecogTriton)
