#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace nugraph {

  class OpHitTable : public Table<unsigned int /*run*/,
                                  unsigned int /*subrun*/,
                                  unsigned int /*event*/,
                                  unsigned int /*oh_id*/,
                                  int /*flash_id*/,
                                  unsigned int /*channel*/,
                                  float /*time*/,
                                  float /*area*/,
                                  float /*width*/,
                                  float /*amplitude*/
                                  > {
  public:
    // optical hit table constructor
    OpHitTable(std::string const& opHitLabel,
               std::string const& flashLabel,
               std::vector<Row> const& data = {});

    // function to fill table from event
    void Fill(art::Event const& evt);

  private:
    std::string fOpHitLabel; ///< Label for optical hit data product
    std::string fFlashLabel; ///< Label for flash data product

  }; // class OpHitTable

} // namespace nugraph
