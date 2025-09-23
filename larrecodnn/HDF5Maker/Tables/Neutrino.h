#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class NeutrinoTable : public Table
  <int, int, int, int, int, int, float, float, float, float, float,
   float, float, float, float>
{
public:
  // neutrino table constructor
  NeutrinoTable(const std::string& nuLabel,
                const std::vector<Row>& data={});

  // function to fill table from event
  void Fill(const art::Event& evt);

private:
  std::string fNuLabel; ///< Label for neutrino MCTruth data product

}; // class NeutrinoTable

} // namespace ng
