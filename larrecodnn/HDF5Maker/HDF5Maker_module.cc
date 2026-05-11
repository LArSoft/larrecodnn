////////////////////////////////////////////////////////////////////////
// Class:       HDF5Maker
// Plugin Type: analyzer (art v3_06_03)
// File:        HDF5Maker_module.cc
//
// Generated at Wed May  5 08:23:31 2021 by V Hewes using cetskelgen
// from cetlib version v3_11_01.
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "larrecodnn/HDF5Maker/Tables/EDep.h"
#include "larrecodnn/HDF5Maker/Tables/Event.h"
#include "larrecodnn/HDF5Maker/Tables/Flash.h"
#include "larrecodnn/HDF5Maker/Tables/LDep.h"
#include "larrecodnn/HDF5Maker/Tables/Neutrino.h"
#include "larrecodnn/HDF5Maker/Tables/OpHit.h"
#include "larrecodnn/HDF5Maker/Tables/Particle.h"
#include "larrecodnn/HDF5Maker/Tables/Spacepoint.h"
#include "larrecodnn/HDF5Maker/Tables/ChargeHit.h"

#include "hep_hpc/hdf5/File.hpp"

namespace ng {

class HDF5Maker : public art::EDAnalyzer
{
public:
  explicit HDF5Maker(fhicl::ParameterSet const& p);
  ~HDF5Maker() noexcept {};

  HDF5Maker(HDF5Maker const&) = delete;
  HDF5Maker(HDF5Maker&&) = delete;
  HDF5Maker& operator=(HDF5Maker const&) = delete;
  HDF5Maker& operator=(HDF5Maker&&) = delete;

  void beginSubRun(art::SubRun const& sr) override;
  void endSubRun(art::SubRun const& /*sr*/) override;
  void analyze(art::Event const& e) override;

  void InitHDF5File(art::SubRun const& sr);
  void CloseHDF5File();

private:
  std::vector<std::unique_ptr<ITable>> fTables; ///< Tables to write to HDF5
  hep_hpc::hdf5::File fFile;  ///< output HDF5 file

  std::string fOutputName;

}; // class HDF5Maker

//-----------------------------------------------------------------------------
HDF5Maker::HDF5Maker(fhicl::ParameterSet const& p)
  : EDAnalyzer{p}, fOutputName(p.get<std::string>("OutputName"))
{
  std::string const& nu_label = p.get<std::string>("NuLabel");
  std::string const& charge_hit_label = p.get<std::string>("ChargeHitLabel");
  std::string const& op_hit_label = p.get<std::string>("OpHitLabel");
  std::string const& flash_label = p.get<std::string>("FlashLabel");
  std::string const& sp_label = p.get<std::string>("SpacepointLabel");

  fTables.push_back(std::make_unique<EventTable>());
  fTables.push_back(std::make_unique<ChargeHitTable>(charge_hit_label));
  fTables.push_back(std::make_unique<OpHitTable>(op_hit_label, flash_label));
  fTables.push_back(std::make_unique<FlashTable>(flash_label));
  fTables.push_back(std::make_unique<SpacepointTable>(sp_label));
  fTables.push_back(std::make_unique<NeutrinoTable>(nu_label));
  fTables.push_back(std::make_unique<EDepTable>(charge_hit_label));
  fTables.push_back(std::make_unique<ParticleTable>(charge_hit_label));
  fTables.push_back(std::make_unique<LDepTable>(op_hit_label));

} // HDF5Maker constructor

//-----------------------------------------------------------------------------
void HDF5Maker::beginSubRun(art::SubRun const& sr)
{
  InitHDF5File(sr);
} // function HDF5Maker::beginSubRun

//-----------------------------------------------------------------------------
void HDF5Maker::endSubRun(art::SubRun const& /*sr*/)
{
  CloseHDF5File();
} // function HDF5Maker::beginSubRun

//-----------------------------------------------------------------------------
void HDF5Maker::analyze(art::Event const& evt)
{
  // fill tables from art record and write to HDF5 file
  for (std::unique_ptr<ITable> const& table : fTables) {
    table->Fill(evt);
    table->WriteNtuple();
  } // for table

} // function HDF5Maker::analyze

//-----------------------------------------------------------------------------
void HDF5Maker::InitHDF5File(art::SubRun const& sr)
{
  // Open HDF5 output
  std::ostringstream fileName;
  fileName << fOutputName << "_r" << std::setfill('0') << std::setw(5)
           << sr.run() << "_s" << std::setfill('0') << std::setw(5)
           << sr.subRun() << ".h5";
  fFile = hep_hpc::hdf5::File(fileName.str(), H5F_ACC_TRUNC);

  // initialize ntuples for each table
  for (std::unique_ptr<ITable>& table : fTables) {
    table->InitNtuple(fFile);
  } // for table

} // function HDF5Maker::InitHDF5File

//-----------------------------------------------------------------------------
void HDF5Maker::CloseHDF5File()
{
  for (std::unique_ptr<ITable>& table : fTables) {
    table->DestroyNtuple();
  } // for table
  fFile.close();
} // function HDF5Maker::InitHDF5File

DEFINE_ART_MODULE(HDF5Maker)

} // namespace ng
