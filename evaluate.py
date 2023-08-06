import itertools
from joblib import parallel_backend, Parallel, delayed, load, dump
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import os
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import time
from pprint import pprint

from scipy.stats import pearsonr, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from joblib import parallel_backend, Parallel, delayed, load, dump


from parameters import *
from helpers import *
from loadData import *



# cross similairty of a pattern
def crosscorrelation (fcor, fu, fv):
    cu = []
    idxU = np.where(fu == 1.0)[0]
    idxV = np.where(fv == 1.0)[0]
    for u in idxU:
        cv = []
        for v in idxV:
            cv.append(np.abs(fcor[u,v]))
        cu.append(np.max(cv))
    CS = np.mean(cu)
    return CS


# cross similairty of a pattern
def ccor (fcor, fu, fv):
    return 0.5*(crosscorrelation (fcor, fu, fv) + crosscorrelation (fcor, fv, fu))


def iou(outputs: np.array, labels: np.array):
    outputs = np.asarray(outputs).astype(bool)
    labels = np.asarray(labels).astype(bool)

    intersection = (outputs & labels).sum((0))
    union = (outputs | labels).sum((0))

    iou = (intersection) / (union)
    return iou



def checkSplits(df):
    # check that the splits were the same by checking gts
    # we have for each repeat 8 (or whatever) rescaling and these must be the same
    for r in range(nRepeats):
        z = df.query("Repeat == @r").copy()
        for f in range(kFold):
            K = [tuple(k) for k in z[f"Fold_{f}_GT"].values]
            assert(len(set(K)) == 1)
    pass



def extractDF (resultsA):
    df = []
    for r in range(len(resultsA)):
        res = {"AUC":resultsA[r]["AUC"], "Repeat": resultsA[r]["Repeat"]}
        res["Resampling"] = str(resultsA[r]["Params"][0]) # same for all
        res["Clf_method"] = str(resultsA[r]["Params"][2][0][0])
        res["FSel"] = str(resultsA[r]["Params"][1])
        res["FSel"] = str(resultsA[r]["Params"][1])
        res["FSel_method"] = str(resultsA[r]["Params"][1][0][0])
        for f in range(kFold):
            res[f"Fold_{f}_GT"] = resultsA[r]["Preds"][f][1]
            res[f"Fold_{f}_Preds"] = resultsA[r]["Preds"][f][0]
            res[f"FSel_{f}_names"] = str(resultsA[r]["Preds"][f][2])
        df.append(res)
    df = pd.DataFrame(df)
    return df



def featureSelectionAgreement (dfA, d):
    dfA['FSel_code'] = pd.factorize(dfA['FSel_method'])[0]
    dfA['Paramcode'], codeIndex = pd.factorize(dfA['Resampling'])
    nM = len(set(dfA["Paramcode"]))
    fMat = np.zeros((nM, nM))
    for i in set(dfA["Paramcode"]):
        for j in set(dfA["Paramcode"]):
            agreement = []
            wi = dfA.query("Paramcode == @i")
            wj = dfA.query("Paramcode == @j")
            assert (len(wi) == len(wj))
            vi = wi.sort_values(["Repeat"])["FSel_code"].values
            vj = wj.sort_values(["Repeat"])["FSel_code"].values
            agreement = np.sum(vi == vj)/len(vi)
            fMat[i,j] = np.mean(agreement)

    fMat = (fMat*100).round(0).astype(int)
    pMat = pd.DataFrame(fMat)
    pMat.columns = [getName(k) for k in codeIndex]
    pMat.index = pMat.columns
    drawArray(pMat, cmap = [("o", 0, 50, 100)], fsize = (10,7), fontsize = 15, fName = f"Table_Fsel_{d}")
    plt.close('all')
    plt.rc('text', usetex=False)
    return pMat



def featureAgreement (dfA, d, data, amethod = "ccor"):
    fcor = np.abs(np.corrcoef (data.T))

    dfA['Paramcode'], codeIndex = pd.factorize(dfA['Resampling'])
    nM = len(set(dfA["Paramcode"]))
    fMat = np.zeros((nM, nM))
    for i in set(dfA["Paramcode"]):
        for j in set(dfA["Paramcode"]):
            agreement = []
            for r in range(nRepeats):
                z = dfA.query("Repeat == @r")
                wi = z.query("Paramcode == @i")
                wj = z.query("Paramcode == @j")
                for f in range(kFold):
                    # expect al FSel_codes to be the same
                    vi = wi[[f'FSel_{f}_names']]
                    vj = wj[[f'FSel_{f}_names']]
                    vi = list(eval("pd."+vi[f"FSel_{f}_names"].iloc[0]))
                    vj = list(eval("pd."+vj[f"FSel_{f}_names"].iloc[0]))
                    fb = set(vi) | set(vj)
                    vi = np.array([1 if v in vi else 0 for v in fb])
                    vj = np.array([1 if v in vj else 0 for v in fb])
                    if amethod == "ccor":
                        agreement.extend( [ccor(fcor, vi, vj)] )
                    if amethod == "iou":
                        agreement.extend( [iou(vi, vj)] )
            fMat[i,j] = np.mean(agreement)

    fMat = (fMat*100).round(0).astype(int)
    pMat = pd.DataFrame(fMat)
    pMat.columns = [getName(k) for k in codeIndex]
    pMat.index = pMat.columns
    drawArray(pMat, cmap = [("o", 0, 50, 100)], fsize = (10,7),  aspect = 0.7, fontsize = 15, fName = f"Table_{amethod}_Features_{d}")
    plt.close('all')
    plt.rc('text', usetex=False)
    return pMat



def extractFeaturesForR (dfA, d, data):
    forR_i = []; forR_j = []
    dfA['Paramcode'], codeIndex = pd.factorize(dfA['Resampling'])
    nM = len(set(dfA["Paramcode"]))
    fMat = np.zeros((nM, nM))
    for i in set(dfA["Paramcode"]):
        for r in range(nRepeats):
            z = dfA.query("Repeat == @r")
            wi = z.query("Paramcode == @i")
            for f in range(kFold):
                vi = wi[[f'FSel_{f}_names']]
                vi = list(eval("pd."+vi[f"FSel_{f}_names"].iloc[0]))
                vi = np.array([1 if v in vi else 0 for v in data.keys()])
                vivec = [getName(codeIndex[i]), r, f]
                vivec.extend(vi)
                forR_i.append(vivec)

    # save for R
    pd.DataFrame(forR_i).to_csv(f"./tmp/fsel_{d}.csv")



def getAUCTable (dfA, d):
    table1 = []
    Amean = pd.DataFrame(dfA).groupby(["Resampling"])["AUC"].mean().round(3)
    Amean = Amean.rename({s:getName(s) for s in Amean.keys()})
    Astd = pd.DataFrame(dfA).groupby(["Resampling"])["AUC"].std().round(3)
    Astd = Astd.rename({s:getName(s) for s in Astd.keys()})
    ctable = pd.DataFrame(Amean)
    #ctable[d] = [str(s[0]) + " +/- " + str(s[1]) for s in list(zip(*[Amean.values, Astd.values]))]
    ctable[d] = [s[0] for s in list(zip(*[Amean.values, Astd.values]))]
    ctable = ctable.drop(["AUC"], axis = 1)
    table1.append (ctable)
    return table1




def drawArray (table3, cmap = None, clipRound = True, fsize = (9,7), aspect = None, DPI = 400, fontsize = None, fName = None, paper = False):

    def colorticks(event=None):
        locs, labels = plt.xticks()
        for k in range(len(labels)):
            labels[k].set_color(getColor(labels[k]._text))

        locs, labels = plt.yticks()
        for k in range(len(labels)):
            labels[k].set_color(getColor(labels[k]._text))


    #table3 = tO.copy()
    table3 = table3.copy()
    if clipRound == True:
        for k in table3.index:
            for l in table3.columns:
                if str(table3.loc[k,l])[-2:] == ".0":
                    table3.loc[k,l] = str(int(table3.loc[k,l]))
    # display graphically
    scMat = table3.copy()
    strMat = table3.copy()
    strMat = strMat.astype( dtype = "str")
    # replace nans in strMat
    strMat = strMat.replace("nan", "")

    if 1 == 1:
        plt.rc('text', usetex=True)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"]})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{mathtools}
            \usepackage{helvet}
            \renewcommand{\familydefault}{\sfdefault}        '''

        fig, ax = plt.subplots(figsize = fsize, dpi = DPI)
        sns.set(style='white')
        #ax = sns.heatmap(scMat, annot = cMat, cmap = "Blues", fmt = '', annot_kws={"fontsize":21}, linewidth = 2.0, linecolor = "black")
        dx = np.asarray(scMat, dtype = np.float64)

        def getPal (cmap):
            if cmap == "g":
                #np.array([0.31084112, 0.51697441, 0.22130127, 1.        ])*255
                pal = sns.light_palette("#4f8338", reverse=False, as_cmap=True)
            elif cmap == "o":
                pal = sns.light_palette("#ff4433", reverse=False, as_cmap=True)
            elif cmap == "+":
                pal  = sns.diverging_palette(20, 120, as_cmap=True)
            elif cmap == "-":
                pal  = sns.diverging_palette(120, 20, as_cmap=True)
            else:
                pal = sns.light_palette("#ffffff", reverse=False, as_cmap=True)
            return pal


        if len(cmap) > 1:
            for j, (cm, vmin, vcenter, vmax) in enumerate(cmap):
                pal = getPal(cm)
                m = np.ones_like(dx)
                m[:,j] = 0
                Adx = np.ma.masked_array(dx, m)
                tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                ax.imshow(Adx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)
                #cba = plt.colorbar(pa,shrink=0.25)
        else:
            if cmap[0][0] == "*":
                for j in range(scMat.shape[1]):
                    pal = getPal("o")
                    m = np.ones_like(dx)
                    m[:,j] = 0
                    Adx = np.ma.masked_array(dx, m)
                    vmin = np.min(scMat.values[:,j])
                    vmax = np.max(scMat.values[:,j])
                    vcenter = (vmin + vmax)/2
                    tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    ax.imshow(Adx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)
                    #cba = plt.colorbar(pa,shrink=0.25)
            else:
                cm, vmin, vcenter, vmax = cmap[0]
                pal = getPal(cm)
                tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                ax.imshow(dx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)

        # Major ticks
        mh, mw = scMat.shape
        ax.set_xticks(np.arange(0, mw, 1))
        ax.set_yticks(np.arange(0, mh, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, mw, 1), minor=True)
        ax.set_yticks(np.arange(-.5, mh, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        for i, c in enumerate(scMat.index):
            for j, f in enumerate(scMat.keys()):
                ax.text(j, i, strMat.at[c, f],    ha="center", va="center", color="k", fontsize = fontsize)
        plt.tight_layout()
        ax.xaxis.set_ticks_position('top') # the rest is the same
        ax.set_xticklabels(scMat.keys(), rotation = 45, ha = "left", fontsize = fontsize)
        ax.set_yticklabels(scMat.index, rotation = 0, ha = "right", fontsize = fontsize)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_tick_params ( labelsize= fontsize)
        colorticks()

    if fName is not None:
        if paper == True:
            fig.savefig(f"./paper/{fName}.png", facecolor = 'w', bbox_inches='tight')
        fig.savefig(f"./results/{fName}.png", facecolor = 'w', bbox_inches='tight')




def createFeatureFigures (table2, table3):
    tableFeatsOIU = pd.concat(table2).groupby(level=0).mean()
    tableFeatsOIU = tableFeatsOIU.loc[tR.index]
    print ("TableFeats IOU mean", tableFeatsOIU.mean().mean())
    drawArray(tableFeatsOIU.round(0), cmap = [("o", 0, 50, 100)], fsize = (10,7), fName = "Figure5", paper = True)

    tableFeatsCCor = pd.concat(table3).groupby(level=0).mean()
    tableFeatsCCor = tableFeatsCCor.loc[tR.index]
    print ("TableFeats CCor mean", tableFeatsCCor.mean().mean())
    drawArray(tableFeatsCCor.round(0), cmap = [("o", 0, 66, 100)], fsize = (10,7), fName = "Figure6", paper = True)
    return None



def createRankingTable (table1):
    tableAUC = pd.concat(table1, axis = 1)
    tR = tableAUC.rank(axis = 0, ascending = False).mean(axis = 1)
    tR = pd.DataFrame(tR).round(1)
    tR.columns = ["Mean rank"]
    tR = tR.sort_values(["Mean rank"])

    # how often the method performed best
    tA = tableAUC.rank(axis = 0,  ascending = False)
    tA = tA.loc[tR.index]
    rTable = tR.copy()
    tM = tableAUC.mean(axis= 1)
    tM = tM - tM["None"]
    tM = tM.round(3)
    rTable["Mean gain in AUC"] = tM
    tX = tableAUC - tableAUC.loc["None"]
    tX = tX.max(axis = 1)
    tX = tX.round(3)
    rTable["Maximum gain in AUC"] = tX

    drawArray(rTable, aspect = 0.6, fsize = (10,7), cmap = [("-", 3.5, (3.5+6.5)/2, 6.5), ("+", -0.015, 0.0, 0.015), ("+", -0.06, 0.0, 0.06)], fName = "Figure2", paper = True)
    rTable.to_csv("./results/ranking.csv")

    methods = tA.index
    tO = np.zeros((len(methods), len(methods)))
    tO = pd.DataFrame (tO)
    tO.index = methods
    tO.columns = methods
    for m in tA.index:
        for n in tA.index:
            wins = np.sum(tA.loc[m] < tA.loc[n])
            draws = np.sum(tA.loc[m] == tA.loc[n])
            score = wins + draws*0.5
            #tA.loc[m] < tA.loc[n]
            tO.loc[m,n] = score
        tO.loc[m,m] = None
    drawArray(tO, fsize = (10,7), cmap = [("+", 0, len(dList)/2, len(dList))], aspect = 0.66, fName = "Figure3", paper = True)

    tA = tableAUC.rank(axis = 0,  ascending = False)
    tA = tA.loc[tR.index]
    len(tR.index)
    drawArray(tA, cmap = [("-", 1.0, len(tR.index)/2, len(tR.index))], fsize = (17,7), aspect = 0.66, fName = "Figure4", paper = True)

    # S1
    table1 = [k.loc[tR.index] for k in table1]
    tableA = pd.concat(table1).groupby(level=0).mean()
    tableA = tableA.loc[tR.index]
    drawArray(tableA, cmap = [("*", 0, 0.50, 1.00)], fsize = (14,7), aspect = 0.6, fName = "FigS1")

    if len(tableA.columns) > 3:
        print (f"Friedman test: {friedmanchisquare(*[tableA[d] for d in tableA.columns])[1]:.3f}")
        print (posthoc_nemenyi_friedman(tableA.T))
        posthoc_nemenyi_friedman(tableA.T).to_excel("./results/Friedman_Nemenyi.xlsx")

    return tR, tableA



def getClfWinnerTable (dfA):
    fDict = {v:k for (k,v) in enumerate(fselParameters["FeatureSelection"]["Methods"].keys())}
    dfA["FSel_code"] = [fDict[k] for k in dfA["FSel_method"]]

    tDict = {v:k for (k,v) in enumerate(clfParameters["Classification"]["Methods"].keys())}
    dfA["Clf_code"] = [tDict[k] for k in dfA["Clf_method"]]

    dfA['Paramcode'], codeIndex = pd.factorize(dfA['Resampling'])

    fMat = np.zeros((nM, len(tDict.keys())))
    for i in set(dfA["Paramcode"]):
        for j in set(dfA["Clf_code"]):
            subdf = dfA.query("Paramcode == @i and Clf_code == @j")
            fMat[i,j] = len(subdf)

    codeIndex = [getName(k) for k in codeIndex]

    pMat = pd.DataFrame(fMat, index = codeIndex, columns = tDict.keys())
    pMat = pMat.astype(np.uint32)
    return pMat


def getFSelWinnerTable (dfA):
    fDict = {v:k for (k,v) in enumerate(fselParameters["FeatureSelection"]["Methods"].keys())}
    dfA["FSel_code"] = [fDict[k] for k in dfA["FSel_method"]]

    tDict = {v:k for (k,v) in enumerate(clfParameters["Classification"]["Methods"].keys())}
    dfA["Clf_code"] = [tDict[k] for k in dfA["Clf_method"]]

    dfA['Paramcode'], codeIndex = pd.factorize(dfA['Resampling'])

    fMat = np.zeros((nM, len(fDict.keys())))
    for i in set(dfA["Paramcode"]):
        for j in set(dfA["FSel_code"]):
            subdf = dfA.query("Paramcode == @i and FSel_code == @j")
            fMat[i,j] = len(subdf)

    codeIndex = [getName(k) for k in codeIndex]

    pMat = pd.DataFrame(fMat, index = codeIndex, columns = fDict.keys())
    pMat = pMat.astype(np.uint32)
    return pMat



def computeTable (d):
    from loadData import Arita2018, Carvalho2018, Hosny2018A, Hosny2018B, Hosny2018C
    from loadData import Ramella2018, Saha2018, Lu2019, Sasaki2019, Toivonen2019, Keek2020
    from loadData import Li2020, Park2020, Song2020, Veeraraghavan2020

    data = eval (d+"().getData('./data/')")
    data,_ = imputeData(data, None)
    try:
        resultsA = load(f"results/resampling_{d}.dump")
    except:
        print ("Does not exist:", d)
        return None

    # extract infos
    dfA = extractDF (resultsA)
    checkSplits(dfA)

    tableA = getAUCTable(dfA, d)
    table1 = pd.DataFrame(tableA[0])
    table2 = featureAgreement (dfA, d, data, amethod = "iou")
    table3 = featureAgreement (dfA, d, data, amethod = "ccor")

    extractFeaturesForR (dfA, d, data)
    return tableA, table1, table2, table3




if __name__ == '__main__':
    # gather data
    with parallel_backend("loky", inner_max_num_threads=1):
        cres = Parallel (n_jobs = ncpus)(delayed(computeTable)(d) for d in dList)
    print ("DONE")

    table1 = []; table2 = []; table3 = []
    for j, _ in enumerate(dList):
        tA, t1, t2, t3 = cres[j]
        table1.append(t1)
        table2.append(t2)
        table3.append(t3)

    # ranking table
    tR, tableA = createRankingTable (table1)

    # features
    table2 = [k.loc[tR.index][tR.index] for k in table2]
    table3 = [k.loc[tR.index][tR.index] for k in table3]
    createFeatureFigures (table2, table3)

#
