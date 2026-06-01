#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <cstdlib>
#include "CombineHarvester/CombineTools/interface/CombineHarvester.h"
#include "CombineHarvester/CombineTools/interface/Observation.h"
#include "CombineHarvester/CombineTools/interface/Process.h"
#include "CombineHarvester/CombineTools/interface/Utilities.h"
#include "CombineHarvester/CombineTools/interface/Systematics.h"
#include "CombineHarvester/CombineTools/interface/BinByBin.h"

using namespace std;

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
        cerr << "Usage: " << argv[0] << " <year> [mode]" << endl;
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

   string file_path = "/afs/desy.de/user/c/chatterj/public/XtoYH4b/";

  // Create an empty CombineHarvester instance that will hold all of the
  // datacard configuration and histograms etc.
  ch::CombineHarvester cb;


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
   
  cats.push_back({1, "h_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut"});
  cats.push_back({2, "h_MX_Comb_3_3_3_2_Inclusive_mHcut"});
  cats.push_back({3, "h_MY_Comb_3_3_3_2_Inclusive_mHcut"});

  //! [part2]
  vector<string> signals;
  if ( year == "2025") {
      signals = loadFilenames(file_path+"SIGNAL_names_2024.txt");
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
  cb.AddProcesses(signals, {"XYH"}, {"13p6TeV_"+year}, {"4b"}, sig_procs, cats, true);
  
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
      SystNames.push_back("Btag_WP_SF_uncorrelated");
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

      // Add rateParam 
      cb.cp().process({"Inclusive_Bkg"}).AddSyst(
          cb, "bkg_norm_"+year, "rateParam", ch::syst::SystMap<>::init(1.0)
      );

      // Set the range [0.1, 2.5] for the parameter
      cb.GetParameter("bkg_norm_"+year)->set_range(0.1, 2.5);

      // Apply MY125Bin_Uncertainty exclusively to the MY bin for the background Separately bin 6 
    for (int i = 6; i <= 8; ++i) {
        std::string my_syst_name = "MY125Bin" + std::to_string(i) + "_Uncertainty_" + year;
        
        cb.cp()
        .bin({"h_MY_Comb_3_3_3_2_Inclusive_mHcut"})
        .process(ch::JoinStr({bkg_procs}))
        .AddSyst(cb, my_syst_name, "shape", SystMap<>::init(1.00));
    }

    cb.cp()
    .bin({"h_MY_Comb_3_3_3_2_Inclusive_mHcut"})
    .process(ch::JoinStr({bkg_procs}))
    .AddSyst(cb, "NonClosure_Uncertainty_" + year, "shape", SystMap<>::init(1.00));


    // Apply the 14 decorrelated Non-Closure uncertainties exclusively to the MX bin for the background
    for (int i = 1; i <= 14; ++i) {
        std::string nc_syst_name = "NonClosure_Bin" + std::to_string(i) + "_Uncertainty_" + year;
        
        cb.cp()
        .bin({"h_MX_Comb_3_3_3_2_Inclusive_mHcut"})
        .process(ch::JoinStr({bkg_procs}))
        .AddSyst(cb, nc_syst_name, "shape", SystMap<>::init(1.00));
    }

    // Add: Unrolled bins been affected by MY bin 6 7 8, MX bin 1-14 and NonClosure uncertainty

    for (int i = 1; i <= 14; ++i) {
        std::string nc_syst_name = "NonClosure_MXBin" + std::to_string(i) + "_Uncertainty_" + year;

        cb.cp()
        .bin({"h_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut"})
        .process(ch::JoinStr({bkg_procs}))
        .AddSyst(cb, nc_syst_name, "shape", SystMap<>::init(1.00));

        // then add the decorrelated to MX bins affected MY bin 6 7 8
        for (int my_bin = 6; my_bin <= 8; ++my_bin) {
            std::string my_syst_name = "MY125Bin" + std::to_string(my_bin) + "_MXBin" + std::to_string(i) + "_Uncertainty_" + year;
            cb.cp()
                .bin({"h_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut"})
                .process(ch::JoinStr({bkg_procs}))
                .AddSyst(cb, my_syst_name, "shape", SystMap<>::init(1.00));
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
  string input_filename; 
  input_filename = "InputFiles/combine_input_XYH4b_"+year+".root";

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
  cb.FilterSysts([&](ch::Systematic *s) {
      if (s->type() == "shape") {
          if (s->shape_u() == nullptr || s->shape_d() == nullptr) return true;
          
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
  char filename[100];
  for (auto m : signals) {
	sprintf(filename,"NMSSM-XYHto4b_Par-%s-%s_.input.root",year.c_str(),m.c_str()); 
	TFile output(filename, "RECREATE");

	for (auto b : bins) {
		cout << ">> Writing datacard for bin: " << b << " and mass: " << m<< "\n";
        
		cb.cp().bin({b}).mass({m, "*"}).WriteDatacard(b + "_" + m + ".txt", output);

        ofstream out_file;
        out_file.open(b + "_" + m + ".txt", ios_base::app); // ios_base::app ensures we add to the end
        if (out_file.is_open()) {
            out_file << "* autoMCStats 0\n"; 
            out_file.close();
        }
	}
  }
  //! [part9]

}
