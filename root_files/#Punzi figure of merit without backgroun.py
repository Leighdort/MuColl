#Punzi figure of merit without background
#Import statements
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import mplhep as hep
import joblib
import uproot
import ROOT
import pandas as pd
import xgboost
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.gridspec as gridspec

# 1. Mute RooFit's message service below ERROR or FATAL
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR) # Or ROOT.RooFit.FATAL
ROOT.RooMsgService.instance().setSilentMode(True)

# 2. Mute core ROOT/Minuit warning and info messages 
ROOT.gErrorIgnoreLevel = ROOT.kError # Or ROOT.kFatal

#First we load in the signal
df = pd.read_csv("/home/rdoherty/Testing_Files/mc_signal_cutsxxx.csv")
signal_bdt = df["bdt_score"]
signal_PROBNE1 = df["PROBNEE1"]
signal_PROBNE2 = df["PROBNEE2"]
signal_total = df["total_events"][0]
signal_postcuts = df["left_overevents"][0]


#Now we want to load in the upper mass sideband
df_2 = pd.read_csv("/home/rdoherty/Testing_Files/background_cutsxxx.csv")
upper_bdt = df_2["bdt_score"]
upper_PROBNE1 = df_2["PROBNEE1"]
upper_PROBNE2 = df_2["PROBNEE2"]
upper_mass = df_2["B_M"]
upper_brem = df_2["Brem_label"]

df_3 = pd.read_csv("/home/rdoherty/Testing_Files/background_lowercutsxxx.csv")
lower_bdt = df_3["bdt_score"]
lower_PROBNE1 = df_3["PROBNEE1"]
lower_PROBNE2 = df_3["PROBNEE2"]
lower_mass = df_3["B_M"]
lower_brem = df_3["Brem_label"]

#"B_M": B_mass
bdt_vals = np.arange(0, 1, 0.01)
prob_vals = np.arange(0, 1, 0.01)
fom_grid = np.zeros((len(bdt_vals), len(prob_vals)))
for i, bdt_score in enumerate(bdt_vals):
    signal_bdtmask = signal_bdt >= bdt_score
    upper_bdtmask = upper_bdt >= bdt_score
    upper_bdtmask = upper_bdt >= bdt_score
    lower_bdtmask =  lower_bdt >= bdt_score
    for j, PROBNEE_score in enumerate(prob_vals):
        print(f"{i}, {j}", flush=True)
        signal_prob1mask = signal_PROBNE1 >= PROBNEE_score
        signal_prob2mask = signal_PROBNE2 >= PROBNEE_score
        upper_prob1mask = upper_PROBNE1 >= PROBNEE_score
        upper_prob2mask = upper_PROBNE2 >= PROBNEE_score
        lower_prob1mask =lower_PROBNE1 >= PROBNEE_score
        lower_prob2mask = lower_PROBNE2 >= PROBNEE_score
        total_signalmask = signal_prob1mask & signal_prob2mask & signal_bdtmask
        total_uppermask = upper_prob1mask & upper_prob2mask & upper_bdtmask
        total_lowermask = lower_prob1mask & lower_prob2mask & lower_bdtmask
        SIGNAL_TOTAL_Eff = len(signal_PROBNE1 [total_signalmask]) / signal_postcuts

        #Now we want to cut on the mass
        upper_mass_cuts = upper_mass[total_uppermask]
        lower_mass_cuts = lower_mass[total_lowermask]

        #now we cut the brem
        upper_brem_cuts = upper_brem[total_uppermask]
        lower_brem_cuts = lower_brem[total_lowermask]


        #Now we need to get it into a crystal ball fit 
        #We are going to not print out a graph
        #We may go to graphs for debugging

        #Let's first work at getting this to its bare bones:
        brem0_lower = lower_mass_cuts[lower_brem_cuts == 0]
        brem1_lower = lower_mass_cuts[lower_brem_cuts == 1]
        brem2_lower = lower_mass_cuts[lower_brem_cuts == 2]

        brem0_upper = upper_mass_cuts[upper_brem_cuts == 0]
        brem1_upper = upper_mass_cuts[upper_brem_cuts == 1]
        brem2_upper = upper_mass_cuts[upper_brem_cuts == 2]

        brem0 = pd.concat([brem0_lower, brem0_upper], ignore_index=True)
        brem1 = pd.concat([brem1_lower, brem1_upper], ignore_index=True)
        brem2 = pd.concat([brem2_lower, brem2_upper], ignore_index=True)


        ##########################################################################################
        B_M0 = ROOT.RooRealVar("B_M0", "B_M0", 4000, 6000, "MeV")
        dataset_0 = ROOT.RooDataSet(
            "dataset_0",
            "dataset_0",
            ROOT.RooArgSet(B_M0)
        )

        for value in brem0:
            if ((value > 4000) and (value < 6000)):
                B_M0.setVal(value)
                dataset_0.add(ROOT.RooArgSet(B_M0))

        B_M0.setRange("left_sideband", 4000, 4689)
        B_M0.setRange("right_sideband", 5588, 6000)
        B_M0.setRange("signal", 4689, 5588)
        B_M0.setRange("signalL", 4689, 4994.01)
        B_M0.setRange("signalR", 4994.01, 5588)

        alpha1 = ROOT.RooRealVar("alpha1", "steep slope", -0.01, -0.5, -0.001)
        alpha2 = ROOT.RooRealVar("alpha2", "shallow", -0.0005, -0.005, -0.00001)
        exp10 = ROOT.RooExponential("exp10", "Steep ", B_M0, alpha1)
        exp20 = ROOT.RooExponential("exp20", "shallow", B_M0, alpha2)
        f_exp0 = ROOT.RooRealVar("f_exp0", "fraction of steep component", 0.4, 0.0, 1.0)
        bkg_pdf0 = ROOT.RooAddPdf("bkg_pdf0", "Bi-exponential Background", ROOT.RooArgList(exp10, exp20), ROOT.RooArgList(f_exp0))

        fit_result = bkg_pdf0.fitTo(
            dataset_0, 
            ROOT.RooFit.Range("left_sideband,right_sideband"), 
            ROOT.RooFit.SplitRange(True),
            ROOT.RooFit.Save(),
            ROOT.RooFit.Verbose(False),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.PrintEvalErrors(-1),
        )

        frame = B_M0.frame()
        # Data must be plotted first so pullHist can find it
        '''
        dataset_0.plotOn(frame, ROOT.RooFit.Name("data"))
        poly_bkg.plotOn(
            frame,
            ROOT.RooFit.Name("poly_fit"),
            ROOT.RooFit.LineColor(ROOT.kOrange + 2),
            ROOT.RooFit.NormRange("left_sideband,right_sideband")
        )

        can0 = ROOT.TCanvas("can0","can0",800,600)
        pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.3, 1.0, 1.0)
        pad1.SetBottomMargin(0.02)
        pad1.Draw()

        pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.3)
        pad2.Draw()

        pad1.cd()
        frame.Draw()

        frame.GetXaxis().SetLabelSize(0)
        frame.GetXaxis().SetTitleSize(0)

        # Pull computation referencing the accurate data/fit names
        hpull = frame.pullHist("data", "poly_fit")

        pad2.cd()
        pull_frame = B_M0.frame(ROOT.RooFit.Title("P"))
        pull_frame.addPlotable(hpull, "P")
        pull_frame.SetMinimum(-5)
        pull_frame.SetMaximum(5)
        pull_frame.GetYaxis().SetTitle("Pull")
        pull_frame.GetYaxis().SetTitleSize(0.1)
        pull_frame.GetYaxis().SetLabelSize(0.08)
        pull_frame.GetXaxis().SetTitleSize(0.12)
        pull_frame.GetXaxis().SetLabelSize(0.08)
        pull_frame.Draw()

        zero_line = ROOT.TLine(4000, 0, 6000, 0)
        zero_line.SetLineColor(ROOT.kBlack)
        zero_line.SetLineStyle(2)
        zero_line.Draw("same")

        can0.SaveAs("exp_fit0_with_pulls_test3.png")
        '''

        # 1. Count sideband entries directly from your array
        n_sideband_events = len(brem0[(brem0 < 4689) | (brem0 > 5588)])

        # 2. Integrate the polynomial across the normalized fitting zone
        int_sidebands = bkg_pdf0.createIntegral(
            ROOT.RooArgSet(B_M0), 
            ROOT.RooFit.Range("left_sideband,right_sideband")
        ).getVal()

        # 3. Integrate across your signal window
        int_signal = bkg_pdf0.createIntegral(
            ROOT.RooArgSet(B_M0), 
            ROOT.RooFit.Range("signal")
        ).getVal()

        # 4. Background scaling calculation
        background_events_in_signal_0 = n_sideband_events * (int_signal / int_sidebands)

        #########################################################################################
        B_M1 = ROOT.RooRealVar("B_M1", "B_M1", 4000, 6000, "MeV")
        dataset_1 = ROOT.RooDataSet(
            "dataset_1",
            "dataset_1",
            ROOT.RooArgSet(B_M1)
        )

        for value in brem1:
            if ((value > 4000) and (value < 6000)):
                B_M1.setVal(value)
                dataset_1.add(ROOT.RooArgSet(B_M1))
        print("now here")

        B_M1.setRange("left_sideband", 4000, 4689)
        B_M1.setRange("right_sideband", 5588, 6000)
        B_M1.setRange("signal", 4689, 5588)
        B_M1.setRange("signalL", 4689, 4994.01)
        B_M1.setRange("signalR", 4994.01, 5588)

        alpha11 = ROOT.RooRealVar("alpha11", "steep slope", -0.01, -0.5, -0.001)
        alpha21 = ROOT.RooRealVar("alpha21", "shallow", -0.0005, -0.005, -0.00001)
        exp11 = ROOT.RooExponential("exp11", "Steep ", B_M1, alpha11)
        exp21 = ROOT.RooExponential("exp21", "shallow", B_M1, alpha21)
        f_exp1 = ROOT.RooRealVar("f_exp1", "fraction of steep component", 0.4, 0.0, 1.0)
        bkg_pdf1 = ROOT.RooAddPdf("bkg_pdf1", "Bi-exponential Background", ROOT.RooArgList(exp11, exp21), ROOT.RooArgList(f_exp1))

        fit_result = bkg_pdf1.fitTo(
            dataset_1, 
            ROOT.RooFit.Range("left_sideband,right_sideband"), 
            ROOT.RooFit.SplitRange(True),
            ROOT.RooFit.Save(),
            ROOT.RooFit.Verbose(False),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.PrintEvalErrors(-1)
        )

        '''

        frame = B_M1.frame()
        # Data must be plotted first so pullHist can find it
        dataset_1.plotOn(frame, ROOT.RooFit.Name("data"))
        poly_bkg.plotOn(
            frame,
            ROOT.RooFit.Name("poly_fit"),
            ROOT.RooFit.LineColor(ROOT.kOrange + 2),
            ROOT.RooFit.NormRange("left_sideband,right_sideband")
        )

        c1 = ROOT.TCanvas("c1","c1",800,600)
        pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.3, 1.0, 1.0)
        pad1.SetBottomMargin(0.02)
        pad1.Draw()

        pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.3)
        pad2.Draw()

        pad1.cd()
        frame.Draw()

        frame.GetXaxis().SetLabelSize(0)
        frame.GetXaxis().SetTitleSize(0)

        # Pull computation referencing the accurate data/fit names
        hpull = frame.pullHist("data", "poly_fit")

        pad2.cd()
        pull_frame = B_M1.frame(ROOT.RooFit.Title("P"))
        pull_frame.addPlotable(hpull, "P")
        pull_frame.SetMinimum(-5)
        pull_frame.SetMaximum(5)
        pull_frame.GetYaxis().SetTitle("Pull")
        pull_frame.GetYaxis().SetTitleSize(0.1)
        pull_frame.GetYaxis().SetLabelSize(0.08)
        pull_frame.GetXaxis().SetTitleSize(0.12)
        pull_frame.GetXaxis().SetLabelSize(0.08)
        pull_frame.Draw()

        zero_line = ROOT.TLine(4000, 0, 6000, 0)
        zero_line.SetLineColor(ROOT.kBlack)
        zero_line.SetLineStyle(2)
        zero_line.Draw("same")

        c1.SaveAs("exp_fit1_with_pulls_test3.png")
    '''
        # 1. Count sideband entries directly from your array
        n_sideband_events = len(brem1[(brem1 < 4689) | (brem1 > 5588)])

        # 2. Integrate the polynomial across the normalized fitting zone
        int_sidebands = bkg_pdf1.createIntegral(
            ROOT.RooArgSet(B_M1), 
            ROOT.RooFit.Range("left_sideband,right_sideband")
        ).getVal()

        # 3. Integrate across your signal window
        int_signal = bkg_pdf1.createIntegral(
            ROOT.RooArgSet(B_M1), 
            ROOT.RooFit.Range("signal")
        ).getVal()

        # 4. Background scaling calculation
        background_events_in_signal_1 = n_sideband_events * (int_signal / int_sidebands)


        ##########################################################################
        B_M2 = ROOT.RooRealVar("B_M2", "B_M2", 4000, 6000, "MeV")
        dataset_2 = ROOT.RooDataSet(
            "dataset_2",
            "dataset_2",
            ROOT.RooArgSet(B_M2)
        )

        for value in brem2:
            if ((value > 4000) and (value < 6000)):
                B_M2.setVal(value)
                dataset_2.add(ROOT.RooArgSet(B_M2))
        print("now here")

        B_M2.setRange("left_sideband", 4000, 4689)
        B_M2.setRange("right_sideband", 5588, 6000)
        B_M2.setRange("signal", 4689, 5588)
        B_M2.setRange("signalL", 4689, 4994.01)
        B_M2.setRange("signalR", 4994.01, 5588)

        alpha12 = ROOT.RooRealVar("alpha12", "steep slope", -0.01, -0.5, -0.001)
        alpha22 = ROOT.RooRealVar("alpha22", "shallow", -0.0005, -0.005, -0.00001)
        exp12 = ROOT.RooExponential("exp12", "Steep ", B_M2, alpha12)
        exp22 = ROOT.RooExponential("exp22", "shallow", B_M2, alpha22)
        f_exp2 = ROOT.RooRealVar("f_exp2", "fraction of steep component", 0.4, 0.0, 1.0)
        bkg_pdf2 = ROOT.RooAddPdf("bkg_pdf2", "Bi-exponential Background", ROOT.RooArgList(exp12, exp22), ROOT.RooArgList(f_exp2))

        fit_result = bkg_pdf2.fitTo(
            dataset_2, 
            ROOT.RooFit.Range("left_sideband,right_sideband"), 
            ROOT.RooFit.SplitRange(True),
            ROOT.RooFit.Save(),
            ROOT.RooFit.Verbose(False),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.PrintEvalErrors(-1)
        )
        '''
        frame2 = B_M2.frame()
        # Data must be plotted first so pullHist can find it
        dataset_2.plotOn(frame2, ROOT.RooFit.Name("data"))
        poly_bkg.plotOn(
            frame2,
            ROOT.RooFit.Name("poly_fit"),
            ROOT.RooFit.LineColor(ROOT.kOrange + 2),
            ROOT.RooFit.NormRange("left_sideband,right_sideband")
        )

        can2 = ROOT.TCanvas("can2","can2",800,600)
        pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.3, 1.0, 1.0)
        pad1.SetBottomMargin(0.02)
        pad1.Draw()

        pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.3)
        pad2.Draw()

        pad1.cd()
        frame2.Draw()

        frame2.GetXaxis().SetLabelSize(0)
        frame2.GetXaxis().SetTitleSize(0)

        # Pull computation referencing the accurate data/fit names
        hpull = frame.pullHist("data", "poly_fit")

        pad2.cd()
        pull_frame = B_M2.frame(ROOT.RooFit.Title("P"))
        pull_frame.addPlotable(hpull, "P")
        pull_frame.SetMinimum(-5)
        pull_frame.SetMaximum(5)
        pull_frame.GetYaxis().SetTitle("Pull")
        pull_frame.GetYaxis().SetTitleSize(0.1)
        pull_frame.GetYaxis().SetLabelSize(0.08)
        pull_frame.GetXaxis().SetTitleSize(0.12)
        pull_frame.GetXaxis().SetLabelSize(0.08)
        pull_frame.Draw()

        zero_line = ROOT.TLine(4000, 0, 6000, 0)
        zero_line.SetLineColor(ROOT.kBlack)
        zero_line.SetLineStyle(2)
        zero_line.Draw("same")

        can2.SaveAs("exp_fit2_with_pulls_test3.png")
        '''
        # 1. Count sideband entries directly from your array
        n_sideband_events = len(brem2[(brem2 < 4689) | (brem2 > 5588)])

       
        int_sidebands = bkg_pdf2.createIntegral(
            ROOT.RooArgSet(B_M2), 
            ROOT.RooFit.Range("left_sideband,right_sideband")
        ).getVal()

       
        int_signal = bkg_pdf2.createIntegral(
            ROOT.RooArgSet(B_M2), 
            ROOT.RooFit.Range("signal")
        ).getVal()

       
        background_events_in_signal_2 = n_sideband_events * (int_signal / int_sidebands)

        N_Background = background_events_in_signal_0 + background_events_in_signal_1 + background_events_in_signal_2
        FOM = (SIGNAL_TOTAL_Eff / (3/2 + np.sqrt(N_Background)))
        fom_grid[i, j] = FOM

#For one we sum across keeping the ProbNNe Value constnat, then we sum across keeping 
fig = plt.figure(figsize=(10,8))  
top = np.sum(fom_grid, axis=0)
right= np.sum(fom_grid, axis=1)

gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[1,4], wspace=0.05, hspace=0.05)

#Top Plot
ax_top = fig.add_subplot(gs[0,0])
ax_top.plot(top, color='black', drawstyle='steps-mid')
ax_top.set_ylabel("Sum of FoM")
ax_top.set_xticklabels([])

#Main Plot
ax_main = fig.add_subplot(gs[1,0])
im = ax_main.imshow(fom_grid, aspect='auto', origin='lower', extent= [prob_vals.min(), prob_vals.max(), bdt_vals.min(), bdt_vals.max()], cmap ='RdPu')
ax_main.set_xlabel("ProbNNe Value")
ax_main.set_ylabel("BDT Value")

#Right plot
ax_right = fig.add_subplot(gs[1,1])
ax_right.plot(right, range(len(right)), color='black', drawstyle='steps-mid')
ax_right.set_xlabel("Sum of FoM")
ax_right.set_yticklabels([])

ax_top.tick_params(labelbottom=False)
ax_right.tick_params(labelleft=False)

#Now we should add a color bar
ax_cbar = fig.add_subplot(gs[0,1])
ax_cbar.axis('off')
cbar = fig.colorbar(im, ax=ax_cbar, fraction=0.6, pad=0.0)
cbar.ax.set_title("FoM")
plt.savefig("FOM_onlycom2.png")
plt.close()

'''
plt.imshow(
    fom_grid,
    origin="lower",
    extent=[prob_vals.min(), prob_vals.max(),
            bdt_vals.min(), bdt_vals.max()],
    aspect="auto"
)
plt.xlabel("PROBNEE cut")
plt.ylabel("BDT cut")
plt.colorbar(label="FOM")
plt.savefig(f"FOM_onlycom.png")
plt.close()
'''



#Let's now print out the max value
FOM_numpy = np.array(fom_grid)
max_value = np.max(FOM_numpy)
max_row, max_col = np.unravel_index(np.argmax(FOM_numpy), FOM_numpy.shape)
print(f"Highest Value: {max_value}")
print(f"Row: {max_row} Column {max_col}")
print("Finished")