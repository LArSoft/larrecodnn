#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class HitTable : public Table
  <int, int, int, int, float, int, int, float, float, int, float, float>
{
public:
  // hit table constructor
  HitTable(std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);
}; // class HitTable

} // namespace ng
