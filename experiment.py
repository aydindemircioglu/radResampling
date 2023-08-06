import itertools
from joblib import parallel_backend, Parallel, delayed, load, dump
import os
import pandas as pd
from scipy.stats import ttest_rel
import time

from cross_validation import *

from loadData import *
from parameters import *


#    wie CV: alle parameter gehen einmal durch
def getExperiments (experimentList, expParameters, sKey, inject = None):
    newList = []
    for exp in experimentList:
        for cmb in list(itertools.product(*expParameters.values())):
            pcmb = dict(zip(expParameters.keys(), cmb))
            if inject is not None:
                pcmb.update(inject)
            _exp = exp.copy()
            _exp.append((sKey, pcmb))
            newList.append(_exp)
    experimentList = newList.copy()
    return experimentList



# this is pretty non-generic, maybe there is a better way, for now it works.
def generateAllExperiments (experimentParameters, verbose = False):
    experimentList = [ [] ]
    for k in experimentParameters.keys():
        if verbose == True:
            print ("Adding", k)
        if k == "Resampling":
            newList = []
            for m in experimentParameters[k]["Methods"]:
                newList.extend(getExperiments (experimentList, experimentParameters[k]["Methods"][m], m))
            experimentList = newList.copy()
        elif k == "FeatureSelection":
            # this is for each N too
            print ("Adding feature selection")
            newList = []
            for n in experimentParameters[k]["N"]:
                for m in experimentParameters[k]["Methods"]:
                    fmethod = experimentParameters[k]["Methods"][m].copy()
                    fmethod["nFeatures"] = [n]
                    newList.extend(getExperiments (experimentList, fmethod, m))
            experimentList = newList.copy()
        elif k == "Classification":
            newList = []
            for m in experimentParameters[k]["Methods"]:
                newList.extend(getExperiments (experimentList, experimentParameters[k]["Methods"][m], m))
            experimentList = newList.copy()
        else:
            experimentList = getExperiments (experimentList, experimentParameters[k], k)

    return experimentList



def createHyperParameters():
    # infos about hyperparameters
    print ("Have", len(resamplingParameters["Resampling"]["Methods"]), "Resampling Methods.")
    print ("Have", len(fselParameters["FeatureSelection"]["Methods"]), "Feature Selection Methods.")
    print ("Have", len(clfParameters["Classification"]["Methods"]), "Classifiers.")

    # generate all experiments
    rsExperiments = generateAllExperiments (resamplingParameters)
    print ("Created", len(rsExperiments), "preprocessing parameter settings")
    fselExperiments = generateAllExperiments (fselParameters)
    print ("Created", len(fselExperiments), "feature selection parameter settings")
    clfExperiments = generateAllExperiments (clfParameters)
    print ("Created", len(clfExperiments), "classifier parameter settings")
    print ("Total", len(rsExperiments)*len(clfExperiments)*len(fselExperiments), "experiments")

    # generate list of experiment combinations
    hyperParameters = []
    for pr in rsExperiments:
        for fe in fselExperiments:
            for clf in clfExperiments:
                hyperParameters.append( (pr, fe, clf))

    return hyperParameters



if __name__ == '__main__':
    # iterate over datasets
    datasets = {}
    for d in dList:
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")
        data = datasets[d]

        # test transform too
        y = data["Target"]
        X = data.drop(["Target"], axis = 1)

        # impute data since a few datasets (esp. Hosny) has some NA values
        X, _ = imputeData (X, None)

        if os.path.exists(f"results/resampling_{d}.dump") == True:
            print ("Found it in cache, continue")
            continue

        hyperParameters = createHyperParameters()

        startNCV = time.time()
        # we process each scaling in a single parallel block,
        # this way we can easily apply  the scaling upfront
        uniqueResamplings = []
        for p in hyperParameters:
            if p[0][0] not in uniqueResamplings:
                uniqueResamplings.append(p[0][0])

        results = []
        for j, u in enumerate(uniqueResamplings):
            print ("\nResampling:", j, "/", len(uniqueResamplings), "---", u)
            jobs = []
            for p in hyperParameters:
                if p[0][0] == u:
                    jobs.append(p)

            X_std = X.copy()
            # for r in range(nRepeats):
            #     area_under_curve, sens, spec, final_preds = Parallel (n_jobs = ncpus)(delayed(nested_cross_validation)(X_std, y, jobs, kFold, r) for r in range(nRepeats))

            resamplingName = getName (str([u]), detox = True)
            os.makedirs("./tmp", exist_ok = True)
            resamplingCache = f"tmp/partial_{resamplingName}_{d}.dump"
            if os.path.exists(resamplingCache):
                cres = load (resamplingCache)
            else:
                # for r in range(nRepeats):
                #     cres = cross_validation(X_std, y, jobs, kFold, r)
                with parallel_backend("loky", inner_max_num_threads=1):
                    cres = Parallel (n_jobs = ncpus)(delayed(cross_validation)(X_std, y, jobs, kFold, r) for r in range(nRepeats))
                dump(cres, resamplingCache)
            results.extend(cres)
        endNCV = time.time()

        # dump results
        os.makedirs("./results", exist_ok = True)
        dump(results, f"results/resampling_{d}.dump")

#
