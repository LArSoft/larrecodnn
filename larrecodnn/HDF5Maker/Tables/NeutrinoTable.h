#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace nugraph {

  class NeutrinoTable : public Table<unsigned int /*run*/,
                                     unsigned int /*subrun*/,
                                     unsigned int /*event*/,
                                     unsigned int /*nu_id*/,
                                     unsigned int /*is_cc*/,
                                     int /*pdg_code*/,
                                     float /*lepton_energy*/,
                                     float /*vertex_x*/,
                                     float /*vertex_y*/,
                                     float /*vertex_z*/,
                                     float /*vertex_t*/,
                                     float /*momentum_x*/,
                                     float /*momentum_y*/,
                                     float /*momentum_z*/,
                                     float /*momentum_e*/
                                     > {
  public:
    // neutrino table constructor
    NeutrinoTable(std::string const& nuLabel, std::vector<Row> const& data = {});

    // function to fill table from event
    void Fill(art::Event const& evt);

  private:
    std::string fNuLabel; ///< Label for neutrino MCTruth data product

  }; // class NeutrinoTable

} // namespace nugraph
