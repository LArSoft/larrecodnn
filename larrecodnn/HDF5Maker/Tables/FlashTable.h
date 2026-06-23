#ifndef LARRECODNN_HDF5MAKER_TABLES_FLASHTABLE_H
#define LARRECODNN_HDF5MAKER_TABLES_FLASHTABLE_H

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace nugraph {

  class FlashTable : public Table<unsigned int /*run*/,
                                  unsigned int /*subrun*/,
                                  unsigned int /*event*/,
                                  unsigned int /*flash_id*/,
                                  float /*pe*/,
                                  float /*y*/,
                                  float /*z*/,
                                  float /*time*/,
                                  float /*y_width*/,
                                  float /*z_width*/,
                                  float /*time_width*/
                                  > {
  public:
    // flash table constructor
    FlashTable(std::string const& flashLabel, std::vector<Row> const& data = {});

    // function to fill table from event
    void Fill(art::Event const& evt);

  private:
    std::string fFlashLabel; ///< Label for flash data product

  }; // class FlashTable

} // namespace nugraph

#endif
