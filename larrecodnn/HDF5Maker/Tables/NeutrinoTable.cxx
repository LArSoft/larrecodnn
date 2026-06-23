#include "larrecodnn/HDF5Maker/Tables/NeutrinoTable.h"

#include "nusimdata/SimulationBase/MCTruth.h"

namespace ng {

  //-----------------------------------------------------------------------------
  // names of columns in neutrino table
  std::vector<std::string> static const NeutrinoColumns{"run",
                                                        "subrun",
                                                        "event",
                                                        "nu_id",
                                                        "is_cc",
                                                        "pdg_code",
                                                        "lepton_energy",
                                                        "vertex_x",
                                                        "vertex_y",
                                                        "vertex_z",
                                                        "vertex_t",
                                                        "momentum_x",
                                                        "momentum_y",
                                                        "momentum_z",
                                                        "momentum_e"};

  //-----------------------------------------------------------------------------
  // neutrino table constructor
  NeutrinoTable::NeutrinoTable(std::string const& nuLabel, std::vector<Row> const& data)
    : Table("neutrinos", NeutrinoColumns, data), fNuLabel(nuLabel)
  {}

  //-----------------------------------------------------------------------------
  void NeutrinoTable::Fill(art::Event const& evt)
  {
    // get event ID
    art::EventID const& id = evt.id();

    // get neutrino truth information
    auto mcts = evt.getHandle<std::vector<simb::MCTruth>>(fNuLabel);
    for (unsigned int nu_id = 0; nu_id < mcts->size(); ++nu_id) {
      const simb::MCNeutrino& mcn = mcts->at(nu_id).GetNeutrino();
      const simb::MCParticle& nu = mcn.Nu();
      const TVector3& dir = nu.Momentum().Vect().Unit();

      // fill table row
      fData.push_back({
        id.run(),
        id.subRun(),
        id.event(), // event ID
        nu_id,
        (mcn.CCNC() == simb::kCC), // nu ID, is_cc
        nu.PdgCode(),
        mcn.Lepton().E(), // pdg_code, lepton_energy
        nu.EndX(),
        nu.EndY(),
        nu.EndZ(),
        nu.EndT(), // vertex
        nu.EndPx(),
        nu.EndPy(),
        nu.EndPz(),
        nu.EndE() // momentum
      });
    } // for true neutrino

  } // function NeutrinoTable::Fill

} // namespace ng
