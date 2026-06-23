#ifndef LARRECODNN_HDF5MAKER_TABLES_TABLE_H
#define LARRECODNN_HDF5MAKER_TABLES_TABLE_H

#include "art/Framework/Principal/Event.h"

#include "hep_hpc/hdf5/File.hpp"
#include "hep_hpc/hdf5/Ntuple.hpp"

namespace nugraph {

  // virtual base class
  class ITable {
  public:
    virtual ~ITable(){};
    virtual void Fill(art::Event const&) = 0;
    virtual void InitNtuple(hep_hpc::hdf5::File&) = 0;
    virtual void WriteNtuple(bool clear = true) = 0;
    virtual void DestroyNtuple() = 0;
  }; // class ITable

  // template table class
  template <typename... Types>
  class Table : public ITable {
  public:
    using Row = std::tuple<Types...>;
    using Ntuple = hep_hpc::hdf5::Ntuple<Types...>;

    // no default constructor
    Table() = delete;

    // default destructor
    ~Table() override = default;

    /// Table constructor
    Table(std::string const& name,
          std::vector<std::string> const& columns,
          std::vector<Row> const& data = {})
      : fName(name), fColumns(columns), fData(data)
    {
      if (sizeof...(Types) != fColumns.size()) {
        throw std::runtime_error("Number of column names does not match number "
                                 "of types in table.");
      } // if column mismatch
    }   // Table constructor

    /// Function to empty contents of table
    void Clear() { fData.clear(); } // function Table::Clear

    /// HDF5 interface
    void InitNtuple(hep_hpc::hdf5::File& file) override
    {
      constexpr auto N = std::tuple_size_v<Row>;
      auto columns = InitColumns<Row>(std::make_index_sequence<N>{});
      fNtuple = std::make_unique<Ntuple>(file, fName, columns);
    } // function Table::InitNtuple

    void WriteNtuple(bool clear) override
    {
      constexpr auto N = std::tuple_size_v<Row>;
      for (const Row& row : fData) {
        WriteRow<Row>(row, std::make_index_sequence<N>{});
      }                       // for table row
      if (clear) { Clear(); } // if clearing data
    }                         // function Table::WriteNtuple

    void DestroyNtuple() override { fNtuple.reset(); } // function Table::DestroyNtuple

  protected:
    std::string fName;
    std::vector<std::string> fColumns;
    std::vector<Row> fData;
    std::unique_ptr<Ntuple> fNtuple;

    template <typename InputTypes, std::size_t... Is>
    void WriteRow(Row const& row, std::index_sequence<Is...>)
    {
      fNtuple->insert(std::get<Is>(row)...);
    } // function Table::WriteRow

    template <typename InputTypes, std::size_t... Is>
    auto InitColumns(std::index_sequence<Is...>)
    {
      return std::make_tuple(
        hep_hpc::hdf5::Column<std::tuple_element_t<Is, InputTypes>, 1>(fColumns[Is])...);
    } // function Table::InitColumns

  }; // template table class

} // namespace nugraph

#endif
