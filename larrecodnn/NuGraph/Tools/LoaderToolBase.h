#ifndef LOADERTOOLBASE_H
#define LOADERTOOLBASE_H

// art TOOL
#include "art/Utilities/ToolMacros.h"
#include "art/Utilities/make_tool.h"

#include "fhiclcpp/ParameterSet.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Core/EDProducer.h"

#include "lardataobj/RecoBase/Hit.h"

#include <vector>
#include <string>
#include <utility>

using std::vector;
using std::string;

struct NuGraphInput {
  NuGraphInput(string s, vector<int32_t> vi):
    input_name(s),isInt(true),input_int32_vec(vi),input_float_vec(vector<float>()){}
  NuGraphInput(string s, vector<float> vf):
    input_name(s),isInt(false),input_int32_vec(vector<int32_t>()),input_float_vec(vf){}
  string input_name;
  bool isInt;
  vector<int32_t> input_int32_vec;
  vector<float> input_float_vec;
};

class LoaderToolBase {

public:

  /**
   *  @brief  Virtual Destructor
   */
  virtual ~LoaderToolBase() noexcept = default;
    
  /**
   *  @brief Interface for configuring the particular algorithm tool
   *
   *  @param ParameterSet  The input set of parameters for configuration
   */
  void configure(const fhicl::ParameterSet&){};

  /**
   * @brief loadData virtual function
   *
   * @param art::Event event record, list of input, idsmap
   */
  virtual void loadData(art::Event& e, vector<art::Ptr<recob::Hit>>& hitlist, vector<NuGraphInput>& inputs, vector<vector<size_t> >& idsmap) = 0;

  void setDebugAndPlanes(bool d, vector<std::string>& p) { debug = d; planes = p;}

protected:
  bool debug;
  vector<std::string> planes;
};

#endif
