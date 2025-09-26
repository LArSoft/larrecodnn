#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class HitTable : public Table
  <int, int, int, int, float, float, int, int, float, float, int, float, float>
{
public:
  // hit table constructor
  HitTable(std::string const& hitLabel, std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fHitLabel; ///< Label for hit data product

}; // class HitTable

} // namespace ng
