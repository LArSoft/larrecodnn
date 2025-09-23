#include "larrecodnn/HDF5Maker/Tables/Neutrino.h"

#include "nusimdata/SimulationBase/MCTruth.h"

namespace ng {

//-----------------------------------------------------------------------------
// names of columns in neutrino table
static const Table::Columns NeutrinoColumns
{
  "run", "subrun", "event", "nu_id", "is_cc", "nu_pdg", "lep_energy",
  "nu_vtx_x", "nu_vtx_y", "nu_vtx_z", "nu_vtx_t",
  "nu_mom_x", "nu_mom_y", "nu_mom_z", "nu_mom_e"
};

//-----------------------------------------------------------------------------
// neutrino table constructor
NeutrinoTable::NeutrinoTable(const std::string& nuLabel,
                             const std::vector<Row>& data)
  : Table("neutrinos", NeutrinoColumns, data), fNuLabel(nuLabel)
{}

//-----------------------------------------------------------------------------
void NeutrinoTable::Fill(const art::Event& evt)
{
  // get event ID
  const art::EventID& id = evt.id();

  // get neutrino truth information
  auto mcts = evt.getHandle<std::vector<simb::MCTruth>>(fNuLabel);
  for (size_t i = 0; i < mcts->size(); ++i) {
    const simb::MCNeutrino& mcn = mcts->at(i).GetNeutrino();
    const simb::MCParticle& nu = mcn.Nu();
    const TVector3& dir = nu.Momentum().Vect().Unit();

    // fill table row
    fData.push_back({
      id.run(), id.subRun(), id.event(),            // event ID
      i, (mcn.CCNC() == simb::kCC),                 // nu ID, is_cc
      nu.PdgCode(), mcn.Lepton().E(),               // nu_pdg, lep_energy
      nu.EndX(), nu.EndY(), nu.EndZ(), nu.EndT(),   // nu vertex
      nu.EndPx(), nu.EndPy(), nu.EndPz(), nu.EndE() // nu momentum
    });
  } // for true neutrino

} // function NeutrinoTable::Fill

} // namespace ng
