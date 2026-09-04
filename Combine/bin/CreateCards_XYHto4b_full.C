#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <cstdlib>
#include "TFile.h"
#include "TH1.h"
#include "CombineHarvester/CombineTools/interface/CombineHarvester.h"
#include "CombineHarvester/CombineTools/interface/Observation.h"
#include "CombineHarvester/CombineTools/interface/Process.h"
#include "CombineHarvester/CombineTools/interface/Utilities.h"
#include "CombineHarvester/CombineTools/interface/Systematics.h"
#include "CombineHarvester/CombineTools/interface/BinByBin.h"

using namespace std;

bool naturalLess(const string &left, const string &right) {
    size_t i = 0;
    size_t j = 0;
    while (i < left.size() && j < right.size()) {
        const bool left_is_digit = std::isdigit(
            static_cast<unsigned char>(left[i]));
        const bool right_is_digit = std::isdigit(
            static_cast<unsigned char>(right[j]));

        if (left_is_digit && right_is_digit) {
            size_t left_end = i;
            size_t right_end = j;
            while (left_end < left.size() && std::isdigit(
                       static_cast<unsigned char>(left[left_end]))) {
                ++left_end;
            }
            while (right_end < right.size() && std::isdigit(
                       static_cast<unsigned char>(right[right_end]))) {
                ++right_end;
            }

            string left_number = left.substr(i, left_end - i);
            string right_number = right.substr(j, right_end - j);
            const size_t left_nonzero = left_number.find_first_not_of('0');
            const size_t right_nonzero = right_number.find_first_not_of('0');
            left_number = left_nonzero == string::npos
                              ? "0" : left_number.substr(left_nonzero);
            right_number = right_nonzero == string::npos
                               ? "0" : right_number.substr(right_nonzero);

            if (left_number.size() != right_number.size()) {
                return left_number.size() < right_number.size();
            }
            if (left_number != right_number) {
                return left_number < right_number;
            }
            i = left_end;
            j = right_end;
            continue;
        }

        if (left[i] != right[j]) return left[i] < right[j];
        ++i;
        ++j;
    }
    return left.size() < right.size();
}

bool isNuisanceLine(const string &line) {
    istringstream parser(line);
    string name;
    string type;
    parser >> name >> type;
    static const set<string> nuisance_types = {
        "shape", "shapeN2", "lnN", "lnU", "gmN", "gmM", "param",
        "rateParam", "flatParam", "discrete", "constr"
    };
    return !name.empty() && nuisance_types.count(type) > 0;
}

bool sortDatacardNuisances(const string &filename) {
    ifstream input(filename);
    if (!input) {
        cerr << "Error opening datacard for natural sorting: " << filename
             << endl;
        return false;
    }

    vector<string> lines;
    string line;
    while (getline(input, line)) lines.push_back(line);
    input.close();

    vector<size_t> nuisance_indices;
    vector<string> nuisance_lines;
    for (size_t index = 0; index < lines.size(); ++index) {
        if (isNuisanceLine(lines[index])) {
            nuisance_indices.push_back(index);
            nuisance_lines.push_back(lines[index]);
        }
    }

    stable_sort(nuisance_lines.begin(), nuisance_lines.end(),
                [](const string &left, const string &right) {
                    istringstream left_parser(left);
                    istringstream right_parser(right);
                    string left_name;
                    string right_name;
                    left_parser >> left_name;
                    right_parser >> right_name;
                    return naturalLess(left_name, right_name);
                });

    for (size_t index = 0; index < nuisance_indices.size(); ++index) {
        lines[nuisance_indices[index]] = nuisance_lines[index];
    }

    ofstream output(filename, ios::trunc);
    if (!output) {
        cerr << "Error writing naturally sorted datacard: " << filename
             << endl;
        return false;
    }
    for (const auto &output_line : lines) output << output_line << '\n';
    return true;
}

struct ShapeBinning {
    int n_mx_bins = 0;
    vector<int> higgs_window_my_bins;
};

bool loadShapeBinning(const string &filename, ShapeBinning &binning) {
    const string mx_dir =
        "h_MaxScore_MX_Comb_3_3_3_2_Inclusive_mHcut";
    const string my_dir =
        "h_MaxScore_MY_Comb_3_3_3_2_Inclusive_mHcut";
    TFile input(filename.c_str(), "READ");
    if (input.IsZombie()) {
        cerr << "Error opening Combine shape file: " << filename << endl;
        return false;
    }

    auto *h_mx = dynamic_cast<TH1 *>(
        input.Get((mx_dir + "/Inclusive_Bkg").c_str()));
    auto *h_my = dynamic_cast<TH1 *>(
        input.Get((my_dir + "/Inclusive_Bkg").c_str()));
    if (!h_mx || !h_my) {
        cerr << "Missing Inclusive_Bkg MX or MY histogram in " << filename
             << endl;
        return false;
    }

    binning.n_mx_bins = h_mx->GetNbinsX();
    constexpr double higgs_window_low = 90.0;
    constexpr double higgs_window_high = 150.0;
    for (int bin = 1; bin <= h_my->GetNbinsX(); ++bin) {
        const double low = h_my->GetXaxis()->GetBinLowEdge(bin);
        const double high = h_my->GetXaxis()->GetBinUpEdge(bin);
        if (low < higgs_window_high && high > higgs_window_low) {
            binning.higgs_window_my_bins.push_back(bin);
        }
    }
    if (binning.n_mx_bins <= 0 || binning.higgs_window_my_bins.empty()) {
        cerr << "Could not derive valid MX/Higgs-window MY bins from "
             << filename << endl;
        return false;
    }

    cout << "Derived shape binning from " << filename << ": "
         << binning.n_mx_bins << " MX bins; Higgs-window MY bins";
    for (int bin : binning.higgs_window_my_bins) cout << " " << bin;
    cout << endl;
    return true;
}

vector<string> loadFilenames(const string &filename) {
    vector<string> filenames;
    ifstream file(filename);

    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        return filenames;
    }

    string line;
    while (getline(file, line)) {
        filenames.push_back(line);
    }
    file.close();
    return filenames;
}

int main(int argc, char **argv) {
	
   string aux_shapes = string(getenv("CMSSW_BASE")) + "/src/CombineHarvester/CombineTools/bin/";

       if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " <year> [mode] [BTag WP|Shape] [add_3b_higgs_nc 0|1]"
             << " [add_4b_nc 0|1] [add_higgsMW_my_uncertainty 0|1]"
             << endl;
        cerr << "Modes:" << endl;
        cerr << "  data_driven   -> Data-driven background (default)" << endl;
        cerr << "  mc   -> MC backgrounds" << endl;
        cerr << "BTag scale factor option should be consistent with Hist2Comb.py: " << endl;
        cerr << "  WP   -> Use correlated and uncorrelated WP uncertainties (default)" << endl;
        cerr << "  Shape -> Use shape uncertainties for each b-tagging source" << endl;
        return 1;
    }

    string year = argv[1];

    string mode = "data_driven";
        if (argc > 2) {
            mode = argv[2];
        }
    string btag_sf_option = "WP";

    if (argc > 3) {
        btag_sf_option = argv[3];
    }

    bool add_4b_nc = true;
    if (argc > 4) {
        string nc_option = argv[4];
        if (nc_option != "0" && nc_option != "1") {
            cerr << "add_4b_nc must be 0 or 1, received: "
                 << nc_option << endl;
            return 1;
        }
        add_4b_nc = (nc_option == "1");
    }

    bool add_higgsMW_my_uncertainty = true;
    if (argc > 5) {
        string my_option = argv[5];
        if (my_option != "0" && my_option != "1") {
            cerr << "add_higgsMW_my_uncertainty must be 0 or 1, received: "
                 << my_option << endl;
            return 1;
        }
        add_higgsMW_my_uncertainty = (my_option == "1");
    }

    bool add_3b_higgs_nc = false;
    if (argc > 6) {
        string nc_option = argv[6];
        if (nc_option != "0" && nc_option != "1") {
            cerr << "add_3b_higgs_nc must be 0 or 1, received: "
                 << nc_option << endl;
            return 1;
        }
        add_3b_higgs_nc = (nc_option == "1");
    }

   string file_path = "/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/";

  //Some of the code for this is in a nested namespace, so
  // we'll make some using declarations first to simplify things a bit.
  using ch::syst::SystMap;
  using ch::syst::era;
  using ch::syst::bin_id;
  using ch::syst::process;

  // Uncomment this next line to see a *lot* of debug information
  // cb.SetVerbosity(3);

  // Here we will just define two categories for an 8TeV analysis. Each entry in
  // the vector below specifies a bin name and corresponding bin_id.
  ch::Categories cats = {};
 
  //! [part1]
   
  cats.push_back({1, "h_MaxScore_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut"});
  cats.push_back({2, "h_MaxScore_MX_Comb_3_3_3_2_Inclusive_mHcut"});
  cats.push_back({3, "h_MaxScore_MY_Comb_3_3_3_2_Inclusive_mHcut"});


  //! [part2]
  vector<string> signals;
  if ( year == "2025") {
      signals = loadFilenames(file_path+"SIGNAL_names_2024.txt");
  } else if ( year == "2022" || year == "2022EE") {
      signals = loadFilenames(file_path+"SIGNAL_names.txt");
  } else if (year == "2023BPiX") {
    signals = loadFilenames(file_path + "SIGNAL_names_2023BPIX.txt");
  } else {
      signals = loadFilenames(file_path+"SIGNAL_names_" + year + ".txt");
  }

  // signals.push_back("NMSSM-XtoYHto4B_Par-MX-1000-MY-150_TuneCP5_13p6TeV_madgraph-pythia8"); // Signal names
  for (auto &sig : signals) {
      const string prefix = "NMSSM-";
      if (sig.rfind(prefix, 0) == 0) {  // check if it starts with "NMSSM_"
          sig.erase(0, prefix.size());  // remove the prefix
      }
  }

  cout<<"# of signal points "<<signals.size()<<endl;

  // Each DNN mass point has its own background estimate. Build each card in
  // an independent CombineHarvester instance so that its background shapes
  // come from the matching BkgEst_<signal>.root-derived input file.
  for (auto const& signal_mass : signals) {
  ch::CombineHarvester cb;
  vector<string> active_signals = {signal_mass};
  const string input_filename = "InputFiles/combine_input_XYH4b_" + year +
                                "_NMSSM-" + signal_mass + ".root";
  ShapeBinning shape_binning;
  if (mode == "data_driven" &&
      !loadShapeBinning(aux_shapes + input_filename, shape_binning)) {
      return 1;
  }
  
  //! [part2]

  //! [part3]
  //backgrounds
  vector<string> bkg_procs;
  // bkg_procs.push_back("Inclusive_Bkg");

  if (mode == "data_driven") {
      bkg_procs.push_back("Inclusive_Bkg");
  }
  else if (mode == "mc") {
      bkg_procs = {
          "TT",
          "ST",
          "QCD",
          "SingleH",
          "DoubleH",
          "Diboson",
          "Wto2Q",
          "Zto2Q",
      };
  }
  else {
      cerr << "Invalid mode: " << mode << endl;
      return 1;
  }



  //signal
  vector<string> sig_procs = {"NMSSM_"};
  if(year=="2024" || year=="2025" ) { sig_procs = {"NMSSM-"} ;}

  cb.AddObservations({"*"}, {"XYH"}, {"13p6TeV_"+year}, {"4b"}, cats);
  cb.AddProcesses({"*"}, {"XYH"}, {"13p6TeV_"+year},  {"4b"}, bkg_procs, cats, false); 
  cb.AddProcesses(active_signals, {"XYH"}, {"13p6TeV_"+year}, {"4b"}, sig_procs, cats, true);
  
  //systematic uncertainty//
  
  vector<string> SystNames = {
    //  JES and JER been decorrelated by year
	//  "JES_AbsoluteStat", 
	//  "JES_RelativeJEREC1", "JES_RelativeJEREC2", 
	//  "JES_RelativePtEC1", "JES_RelativePtEC2", 
	//  "JES_RelativeSample", "JES_RelativeStatEC", "JES_RelativeStatFSR", 
	//  "JES_TimePtEta",
	//  "JES_Total",
    // "JER",

    // Correlated uncertainties across eras
	 "PU",
    //  "LHEScale_muR","LHEScale_muF",
	 "LHEScale","LHEPDF","LHEAlphaS","PS_ISR","PS_FSR",

    // Correlated JES uncertainties across eras
    "JES_AbsoluteMPFBias", "JES_AbsoluteScale", "JES_FlavorQCD", "JES_Fragmentation",
    "JES_PileUpPtRef",
    "JES_PileUpDataMC",  "JES_PileUpPtBB", "JES_PileUpPtEC1", "JES_PileUpPtEC2", 
    "JES_RelativeFSR", "JES_RelativePtBB", "JES_RelativeBal", 
    "JES_SinglePionECAL", "JES_SinglePionHCAL",

  };

  vector<string> ErasDecoSystNames = {
    "JER",
    "JES_AbsoluteStat", "JES_RelativeJEREC1", "JES_RelativeJEREC2",
    "JES_RelativePtEC1", "JES_RelativePtEC2", 
    "JES_RelativeSample", "JES_RelativeStatEC", "JES_RelativeStatFSR",
    "JES_TimePtEta",
  };

  for (string& syst : ErasDecoSystNames) {
      syst = syst + "_" + year; 
      SystNames.push_back(syst);
  }


  if (btag_sf_option == "WP") {
      SystNames.push_back("Btag_WP_SF_correlated");
      // SystNames.push_back("Btag_WP_SF_uncorrelated");
      SystNames.push_back("Btag_WP_SF_uncorrelated" + "_" + year);
  } else if (btag_sf_option == "Shape") {
      SystNames.push_back("Btag_SF_jes");
      SystNames.push_back("Btag_SF_lf");
      SystNames.push_back("Btag_SF_lfstats1");
      SystNames.push_back("Btag_SF_lfstats2");
      SystNames.push_back("Btag_SF_hf");
      SystNames.push_back("Btag_SF_hfstats1");
      SystNames.push_back("Btag_SF_hfstats2");
      SystNames.push_back("Btag_SF_cferr1");
      SystNames.push_back("Btag_SF_cferr2");
      SystNames.push_back("Btag_SF_correction");
  }

  if (mode == "data_driven") {

      vector<string> bkgSystNames = {
        "Statistical_Uncertainty_" + year, 
        "Systematic_Uncertainty_" + year
      };

      for (auto const& syst : bkgSystNames) {
          cb.cp()
            .process(ch::JoinStr({bkg_procs}))
            .AddSyst(cb, syst, "shape", SystMap<>::init(1.00));
      }

      if (add_3b_higgs_nc) {
          // nc_3b is independent of the existing 4b non-closure nuisance. The
          // special MY bins are unchanged in its templates.
          cb.cp()
            .process({"Inclusive_Bkg"})
            .AddSyst(cb, "NonClosure3b_Uncertainty_" + year,
                     "shape", SystMap<>::init(1.00));
      }

      // Add rateParam 
      cb.cp().process({"Inclusive_Bkg"}).AddSyst(
          cb, "bkg_norm_"+year, "rateParam", ch::syst::SystMap<>::init(1.0)
      );

      // Set the range [0.1, 2.5] for the parameter
      cb.GetParameter("bkg_norm_"+year)->set_range(0.1, 2.5);

      // Add one uncertainty for every MY bin overlapping 90--150 GeV.
    if (add_higgsMW_my_uncertainty) {
        for (int my_bin : shape_binning.higgs_window_my_bins) {
            std::string my_syst_name = "MY125Bin" + std::to_string(my_bin) + "_Uncertainty_" + year;

            cb.cp()
            .bin({"h_MaxScore_MY_Comb_3_3_3_2_Inclusive_mHcut"})
            .process(ch::JoinStr({bkg_procs}))
            .AddSyst(cb, my_syst_name, "shape", SystMap<>::init(1.00));
        }
    }

    if (add_4b_nc) {
        cb.cp()
        .bin({"h_MaxScore_MY_Comb_3_3_3_2_Inclusive_mHcut"})
        .process(ch::JoinStr({bkg_procs}))
        .AddSyst(cb, "NonClosure_Uncertainty_" + year, "shape", SystMap<>::init(1.00));
    }


    // Apply one decorrelated non-closure uncertainty per active MX bin.
    if (add_4b_nc) {
        for (int i = 1; i <= shape_binning.n_mx_bins; ++i) {
            std::string nc_syst_name = "NonClosure_Bin" + std::to_string(i) + "_Uncertainty_" + year;
            cb.cp()
            .bin({"h_MaxScore_MX_Comb_3_3_3_2_Inclusive_mHcut"})
            .process(ch::JoinStr({bkg_procs}))
            .AddSyst(cb, nc_syst_name, "shape", SystMap<>::init(1.00));
        }
    }

    // Add the corresponding MX and Higgs-window MY uncertainties to unrolled bins.

    for (int i = 1; i <= shape_binning.n_mx_bins; ++i) {
        if (add_4b_nc) {
            std::string nc_syst_name = "NonClosure_MXBin" + std::to_string(i) + "_Uncertainty_" + year;

            cb.cp()
            .bin({"h_MaxScore_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut"})
            .process(ch::JoinStr({bkg_procs}))
            .AddSyst(cb, nc_syst_name, "shape", SystMap<>::init(1.00));
        }

        if (add_higgsMW_my_uncertainty) {
            for (int my_bin : shape_binning.higgs_window_my_bins) {
                std::string my_syst_name = "MY125Bin" + std::to_string(my_bin) + "_MXBin" + std::to_string(i) + "_Uncertainty_" + year;
                cb.cp()
                    .bin({"h_MaxScore_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut"})
                    .process(ch::JoinStr({bkg_procs}))
                    .AddSyst(cb, my_syst_name, "shape", SystMap<>::init(1.00));
            }
        }
    }

  }



  


  
  //! [part4]

//   //Some of the code for this is in a nested namespace, so
//   // we'll make some using declarations first to simplify things a bit.
//   using ch::syst::SystMap;
//   using ch::syst::era;
//   using ch::syst::bin_id;
//   using ch::syst::process;

  //! [part5]
//  cb.cp().signals()

    //luminosity uncertainty (affecting both signal & backgrounds)
    if(year=="2022"||year=="2022EE"){  
        cb.cp()
            .process(ch::JoinStr({sig_procs}))
            .AddSyst(cb, "lumi_1", "lnN", SystMap<era>::init
            ({"13p6TeV_2022"}, 1.0138));
    }

    if(year=="2023"||year=="2023BPiX"){
        cb.cp()
            .process(ch::JoinStr({sig_procs}))
            .AddSyst(cb, "lumi_1", "lnN", SystMap<era>::init
            ({"13p6TeV_2023"}, 1.0017));

        cb.cp()
            .process(ch::JoinStr({sig_procs}))
            .AddSyst(cb, "lumi_2", "lnN", SystMap<era>::init
            ({"13p6TeV_2023"}, 1.0127));
    }

    if(year=="2024"){ 
            cb.cp()
            .process(ch::JoinStr({sig_procs}))
            .AddSyst(cb, "lumi_1", "lnN", SystMap<era>::init
            ({"13p6TeV_2024"}, 1.0020));

        cb.cp()
            .process(ch::JoinStr({sig_procs}))
            .AddSyst(cb, "lumi_2", "lnN", SystMap<era>::init
            ({"13p6TeV_2024"}, 1.0068));

        cb.cp()
            .process(ch::JoinStr({sig_procs}))
            .AddSyst(cb, "lumi_3", "lnN", SystMap<era>::init
            ({"13p6TeV_2024"}, 1.0144));
    }
    if(year=="2025"){ // need to update the lumi uncertainty for 2025
        // cb.cp()
        //     .process(ch::JoinStr({sig_procs}))
        //     .AddSyst(cb, "lumi_1", "lnN", SystMap<era>::init
        //     ({"13p6TeV_2025"}, 1.0020));

        // cb.cp()
        //     .process(ch::JoinStr({sig_procs}))
        //     .AddSyst(cb, "lumi_2", "lnN", SystMap<era>::init
        //     ({"13p6TeV_2025"}, 1.0068));

        // cb.cp()
        //     .process(ch::JoinStr({sig_procs}))
        //     .AddSyst(cb, "lumi_3", "lnN", SystMap<era>::init
        //     ({"13p6TeV_2025"}, 1.0144));
        cb.cp()
            .process(ch::JoinStr({sig_procs}))
            .AddSyst(cb, "lumi_4", "lnN", SystMap<era>::init
            ({"13p6TeV_2025"}, 1.05));
    }


	// systematic uncertainties affecting signal
    for (auto const& syst : SystNames) {
      cb.cp()
        .process(ch::JoinStr({sig_procs})) 
        .AddSyst(cb, syst, "shape", SystMap<>::init(1.00));
  }

    if (mode == "mc") {
      for (auto const& syst : SystNames) {
          cb.cp()
            .process(ch::JoinStr({bkg_procs}))
            .AddSyst(cb, syst, "shape", SystMap<>::init(1.00));
      }
  }


      

  //! [part5]

  //! [part7]
  cb.cp().backgrounds().ExtractShapes(
      aux_shapes + input_filename,
      "$BIN/$PROCESS",
      "$BIN/$PROCESS_$SYSTEMATIC");
  cb.cp().signals().ExtractShapes(
      aux_shapes + input_filename,
      "$BIN/$PROCESS$MASS",
      "$BIN/$PROCESS$MASS_$SYSTEMATIC");
  //! [part7]


////////////
//  Drop any shape systematics where the Up or Down variation has 0 or negative yield

// print out the file name beeing processed for debugging
  cout << "Processing input file: " << aux_shapes + input_filename << endl;

  cb.FilterSysts([&](ch::Systematic *s) {
      if (s->type() == "shape") {
          if (s->shape_u() == nullptr || s->shape_d() == nullptr) {
              std::cout << "[Warning] Dropping missing shape systematic "
                        << s->name() << " for process " << s->process()
                        << " in bin " << s->bin() << ".\n";
              return true;
          }
          
          if (s->shape_u()->Integral() <= 0.0 || s->shape_d()->Integral() <= 0.0) {
              std::cout << "[Warning] Dropping shape systematic " << s->name() 
                        << " for process " << s->process() << " in bin " << s->bin() 
                        << " because an up/down variation has <= 0 yield.\n";
              return true; 
          }
      }
      return false;
  });/////////////////////////



  //! [part8]
 
//   auto bbb = ch::BinByBinFactory()
//     .SetAddThreshold(1.e-6)
//     .SetFixNorm(true);
//   bbb.AddBinByBin(cb.cp().backgrounds(), cb);
  
  //! [part8]
  //
  // This function modifies every entry to have a standardised bin name of
  // the form: {analysis}_{channel}_{bin_id}_{era}
  // which is commonly used in the htt analyses
//   cb.SetAutoMCStats(cb, 0, 1, 1);

  ch::SetStandardBinNames(cb);
  //! [part8]

  //! [part9]
  // First we generate a set of bin names:
  set<string> bins = cb.bin_set();
  // This method will produce a set of unique bin names by considering all
  // Observation, Process and Systematic entries in the CombineHarvester
  // instance.

  // We create the output root file that will contain all the shapes.
  // Finally we iterate through each bin,mass combination and write a datacard.
  const string m = signal_mass;
  const string output_filename = "NMSSM-XYHto4b_Par-" + year + "-" +
                                 m + "_.input.root";
	TFile output(output_filename.c_str(), "RECREATE");

	for (auto b : bins) {
		cout << ">> Writing datacard for bin: " << b << " and mass: " << m<< "\n";
        
		const string datacard_filename = b + "_" + m + ".txt";
		cb.cp().bin({b}).mass({m, "*"}).WriteDatacard(datacard_filename, output);

        ofstream out_file;
		out_file.open(datacard_filename, ios_base::app); // ios_base::app ensures we add to the end
		if (out_file.is_open()) {
		    out_file << "* autoMCStats 0\n";
		    out_file.close();
		}
		if (!sortDatacardNuisances(datacard_filename)) return 1;
	}
  //! [part9]

  }  // end loop over mass-specific background estimates
}
