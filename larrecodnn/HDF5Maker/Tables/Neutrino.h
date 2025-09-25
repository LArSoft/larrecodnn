#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class NeutrinoTable : public Table
  <int, int, int, int, int, int, float, float, float, float, float,
   float, float, float, float>
{
public:
  // neutrino table constructor
  NeutrinoTable(std::string const& nuLabel,
                std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fNuLabel; ///< Label for neutrino MCTruth data product

}; // class NeutrinoTable

} // namespace ng
