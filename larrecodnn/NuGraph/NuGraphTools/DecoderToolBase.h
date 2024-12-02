#ifndef DECODERTOOLBASE_H
#define DECODERTOOLBASE_H

// art TOOL
#include "art/Utilities/ToolMacros.h"
#include "art/Utilities/make_tool.h"

#include "fhiclcpp/ParameterSet.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Core/EDProducer.h"

#include "larrecodnn/ImagePatternAlgs/NuSonic/Triton/TritonData.h"

#include <vector>
#include <string>
#include <utility>

using std::vector;
using std::string;

class DecoderToolBase {

public:

    /**
     *  @brief  Virtual Destructor
     */
    virtual ~DecoderToolBase() noexcept = default;
    
    /**
     *  @brief Interface for configuring the particular algorithm tool
     *
     *  @param ParameterSet  The input set of parameters for configuration
     */
    void configure(const fhicl::ParameterSet&){};

    /**
     *  @brief Type of decoder, used for declaring what is produced
     */
    enum DecoderType { unset=0, Filter=1, Semantic5=2, Vertex=3};

    /**
     * @brief declareProducts function
     *
     */
    virtual std::pair<DecoderType,std::string> declareProducts() = 0;

    /**
     * @brief writeEmptyToEvent function
     *
     * @param art::Event event record
     */
    virtual void writeEmptyToEvent(art::Event& e, vector<vector<size_t> >& idsmap) = 0;

    /**
     * @brief writeToEvent function
     *
     * @param art::Event event record
     * @param TritonOutputMap
     */
    virtual void writeToEvent(art::Event& e, vector<vector<size_t> >& idsmap, const lartriton::TritonOutputMap& infer_result) = 0;

    // Function to print elements of a vector<float>
    void printVector(const std::vector<float>& vec)
    {
      for (size_t i = 0; i < vec.size(); ++i) {
	std::cout << vec[i];
	// Print space unless it's the last element
	if (i != vec.size() - 1) { std::cout << " "; }
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }

    template <typename T, size_t N>
      void softmax(std::array<T, N>& arr)
    {
      T m = -std::numeric_limits<T>::max();
      for (size_t i = 0; i < arr.size(); i++) {
	if (arr[i] > m) { m = arr[i]; }
      }
      T sum = 0.0;
      for (size_t i = 0; i < arr.size(); i++) {
	sum += expf(arr[i] - m);
      }
      T offset = m + logf(sum);
      for (size_t i = 0; i < arr.size(); i++) {
	arr[i] = expf(arr[i] - offset);
      }
      return;
    }

};

#endif
