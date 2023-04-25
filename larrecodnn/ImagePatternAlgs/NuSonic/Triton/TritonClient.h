#ifndef NuSonic_Triton_TritonClient
#define NuSonic_Triton_TritonClient

#include "larrecodnn/ImagePatternAlgs/NuSonic/Triton/TritonData.h"
#include "fhiclcpp/fwd.h"

#include <memory>
#include <string>
#include <vector>

#include "grpc_client.h"

namespace lartriton {

  class TritonClient {
  public:
    struct ServerSideStats {
      uint64_t inference_count_;
      uint64_t execution_count_;
      uint64_t success_count_;
      uint64_t cumm_time_ns_;
      uint64_t queue_time_ns_;
      uint64_t compute_input_time_ns_;
      uint64_t compute_infer_time_ns_;
      uint64_t compute_output_time_ns_;
    };

    //constructor
    TritonClient(const fhicl::ParameterSet& params);

    //accessors
    TritonInputMap& input() { return input_; }
    const TritonOutputMap& output() const { return output_; }
    unsigned batchSize() const { return batchSize_; }
    bool verbose() const { return verbose_; }
    bool setBatchSize(unsigned bsize);

    //main operation
    void dispatch()
    {
      start();
      evaluate();
    }

    //helper
    void reset();

  protected:
    //helper
    bool getResults(std::shared_ptr<nvidia::inferenceserver::client::InferResult> results);

    void start();
    void evaluate();
    void finish(bool success);

    void reportServerSideStats(const ServerSideStats& stats) const;
    ServerSideStats summarizeServerStats(const inference::ModelStatistics& start_status,
                                         const inference::ModelStatistics& end_status) const;

    inference::ModelStatistics getServerSideStatus() const;

    //members
    TritonInputMap input_;
    TritonOutputMap output_;
    unsigned allowedTries_, tries_;
    std::string serverURL_;
    unsigned maxBatchSize_;
    unsigned batchSize_;
    bool noBatch_;
    bool verbose_;

    //IO pointers for triton
    std::vector<nvidia::inferenceserver::client::InferInput*> inputsTriton_;
    std::vector<const nvidia::inferenceserver::client::InferRequestedOutput*> outputsTriton_;

    std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> client_;
    //stores timeout, model name and version
    nvidia::inferenceserver::client::InferOptions options_;
  };

}
#endif
