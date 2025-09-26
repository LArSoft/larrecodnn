#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class EDepTable : public Table
  <int, int, int, int, int, float, float, float, float>
{
public:
  // edep table constructor
  EDepTable(std::string const& hitLabel,
            std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fHitLabel; ///< Label for hit data product

}; // class EDepTable

} // namespace ng
