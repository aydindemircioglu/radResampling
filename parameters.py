from collections import OrderedDict
import numpy as np

ncpus = 30
nRepeats = 30
kFold = 5

dList =  [ "Arita2018",  "Carvalho2018", \
                "Hosny2018A", "Hosny2018B", "Hosny2018C", \
                "Ramella2018",  "Saha2018", "Lu2019", "Sasaki2019", "Toivonen2019", "Keek2020", "Li2020", \
                "Park2020",
                "Song2020", "Veeraraghavan2020" ]


resamplingParameters = OrderedDict({
    "Resampling": {
        "Methods": {
            "RUS": {},
            "ENN": {"k": [3]}, # larger than 3 lead to constant y_train_resampled, leading to crashes
            "AllKNN": {"k": [3,5,7]},
            "TomekLinks": {},

            "ROS": {},
            "SMOTE": {"k": [3,5,7]},
            "SVMSMOTE": {"k": [3,5,7]},

            "SMOTEENN": {"k": [3,5], "n": [3]},
            "SMOTETomek": {"k": [3,5,7]},
            "None": {}
        }
    }
})


fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "N": [1,2,4,8,16,32,64],
        "Methods": {
            "LASSO": {"C": [1.0]},
            "ET": {},
            "Anova": {},
            "Bhattacharyya": {},
        }
    }
})


clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "LogisticRegression": {"C": np.logspace(-10, 10, 11, base = 2.0)},
            "NaiveBayes": {},
            "RBFSVM": {"C":np.logspace(-10, 10, 11, base = 2.0), "gamma":["auto"]},
            "RandomForest": {"n_estimators": [250]},
            "kNN": {"k": [1,3,5,7,9]},
        }
    }
})


if __name__ == '__main__':
    import pandas as pd
    from helpers import getName

    nDict = {"RUS": "Random Undersampling",
       "ENN": "Edited NN",
       "AllKNN": "All k-NN",
       "TomekLinks":"Tomek links",
       "ROS": "Random Oversampling",
       "SMOTE": "SMOTE",
       "KMeansSMOTE": "k-Means-SMOTE",
       "SVMSMOTE": "SVM-SMOTE",
       "SMOTEENN": "SMOTE+Edited NN",
       "SMOTETomek": "SMOTE+Tomek links",
       "None": "None"}

    def beautify (z):
        if z == {}:
            return "-"
        zstr = ''
        for j, r in enumerate(z.keys()):
            zstr += (f"{r} in {z[r]}")
            if j < len(z.keys())-1:
                zstr += ", "
        return zstr


    methods = resamplingParameters["Resampling"]["Methods"].keys()
    pTable = [{"Method":nDict[m], "Parameters":beautify(resamplingParameters["Resampling"]["Methods"][m])} for m in methods]
    pTable = pd.DataFrame(pTable)
    pTable.to_excel("./paper/Table_2.xlsx")


#
