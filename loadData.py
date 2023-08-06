import numpy as np
import os
import pandas as pd
from scipy.io import arff
import scipy.io as sio
from pprint import pprint
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import Counter


# Define a class
class DataSet:
    def __init__(self, name):
        self.name = name

    def info (self):
        print("Dataset:", str(type(self).__name__), "\tDOI:", self.ID)

    def getData (self, folder):
        print("This octopus is " + self.color + ".")
        print(self.name + " is the octopus's name.")


#folder = "./data"
class Veeraraghavan2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1038/s41598-020-72475-9"
        self.mod = "DCE-MRI"
        self.tumor = "Breast"

    def getData (self, folder):
        dataDir = os.path.join(folder, "s41598-020-72475-9/SciRepEndometrial2020/")
        inputFile = "CERRFeatures_FINAL.csv"
        feats = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
        feats = feats.drop(["Exclude", "PID", "Histology", "FIGO", "Stage", "MolecularSubtype", "Age", "TMB", "CT_Make"], axis = 1)

        inputFile = "clinData_Nov18_2019.csv"
        targets = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
        targets = targets[["TMB", "Exclude"]]
        data = pd.concat([feats,targets], axis = 1)

        data = data[data["Exclude"] == "No"]
        data["Target"] = 1*(targets["TMB"] > 15.5)
        data = data.drop(["TMB", "Exclude"], axis = 1)
        data = data.reset_index(drop = True)
        return (data)



# take mgmt as target, but we have more features,
# paper only has 489 and we only take complete cases
class Sasaki2019 (DataSet):
    def __init__(self):
        self.ID = "data:10.1038/s41598-019-50849-y"
        self.mod = "MRI"
        self.tumor = "Brain"

    def getData (self, folder):
        dataDir = os.path.join(folder, "s41598-019-50849-y")
        inputFile = "41598_2019_50849_MOESM3_ESM.xlsx"
        data = pd.read_excel(os.path.join(dataDir, inputFile),header=1, engine='openpyxl')
        data["Target"] = data["MGMT_1Met0Unmet"]
        #data["Target"] = data["TERTp_1mt0wt"]  # also < 0.70 in auc
        data = data.drop(data.keys()[0:26], axis = 1)
        # complete cases only
        data = data.dropna()
        data = data.reset_index(drop = True)
        return data


# use R to read file,
# > load("ROCS_2018.RData")
# > write.table(x_complete, "./x_complete.csv", sep = ";")
# > write.table(x_complete_scale, "./x_complete_scale.csv", sep = ";")
# we then use as target those pats which had an PFS
# within 2 years.
class Lu2019 (DataSet):
    def __init__(self):
        self.ID = "data:10.1038/s41467-019-08718-9"
        self.mod = "CT"
        self.tumor = "Ovarian cancer"

    def getData (self, folder):
        dataDir = os.path.join(folder, "s41467-019-08718-9")
        inputFile = "./x_complete_scale.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile), sep = ";")
        inputFile = "./x_complete.csv"
        target = pd.read_csv(os.path.join(dataDir, inputFile), sep = ";")
        data["PFS"] = target["PFS.event"]
        data["PFS_time"] = target["Progression.free.survival..days."]
        data = data[data["PFS_time"].notna()]
        data["Target"] = (data["PFS_time"] < 365*2) & (data["PFS"] == 1)
        data["Target"] = 1.0*data["Target"]
        data = data.drop(["PFS", "PFS_time"], axis = 1)
        data = data.reset_index(drop = True)
        return data



class Arita2018 (DataSet):
    def __init__(self):
        self.ID = "data:10.1038/s41598-018-30273-4"
        self.mod = "MRI"
        self.tumor = "Brain"

    def getData (self, folder):
        dataDir = os.path.join(folder, "s41598-018-30273-4/")
        # inputFile = "41598_2018_30273_MOESM2_ESM.xlsx"
        # data = pd.read_excel(os.path.join(dataDir, inputFile),header=1)
        # data.shape
        # data.head()

        inputFile = "41598_2018_30273_MOESM3_ESM.csv"
        dataA = pd.read_csv(os.path.join(dataDir, inputFile), encoding = "ISO-8859-1")

        inputFile = "41598_2018_30273_MOESM4_ESM.csv"
        dataB = pd.read_csv(os.path.join(dataDir, inputFile), encoding = "ISO-8859-1")
        data = pd.concat([dataA,dataB])
        data["Target"] = data["IDH.1"]
        data = data[[z for z in data.keys()[33:]]]
        data = data[data.isnull().sum(axis=1) < 22]
        data = data.reset_index(drop = True)
        return data



class Song2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0237587"
        self.mod = "MRI"
        self.tumor = "Prostate cancer"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0237587/")
        inputFile = "numeric_feature.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
        data["Target"] = np.asarray(data["label"] > 0.5, dtype = np.uint8)
        data = data.drop(["Unnamed: 0", "label"], axis = 1)
        data = data.reset_index(drop = True)
        return (data)



class Keek2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0232639"
        self.mod = "CT"
        self.tumor = "HNSCC"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0232639/Peritumoral-HN-Radiomics/")

        inputFile = "Clinical_DESIGN.csv"
        clDESIGNdata = pd.read_csv(os.path.join(dataDir, inputFile), sep=";")
        df = clDESIGNdata.copy()
        # remove those pats who did not die and have FU time less than 3 years
        df = clDESIGNdata[(clDESIGNdata["StatusDeath"].values == 1) | (clDESIGNdata["TimeToDeathOrLastFU"].values > 3*365)]
        target = df["TimeToDeathOrLastFU"] < 3*365
        target = np.asarray(target, dtype = np.uint8)

        inputFile = "Radiomics_DESIGN.csv"
        rDESIGNdata = pd.read_csv(os.path.join(dataDir, inputFile), sep=";")
        rDESIGNdata = rDESIGNdata.drop([z for z in rDESIGNdata.keys() if "General_" in z], axis = 1)
        rDESIGNdata = rDESIGNdata.loc[df.index]
        rDESIGNdata = rDESIGNdata.reset_index(drop = True)
        rDESIGNdata["Target"] = target

        # convert strings to float
        rDESIGNdata = rDESIGNdata.applymap(lambda x: float(str(x).replace(",", ".")))
        rDESIGNdata["Target"] = target

        rDESIGNdata = rDESIGNdata.reset_index(drop = True)
        return rDESIGNdata



class Li2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0227703"
        self.mod = "MRI"
        self.tumor = "Glioma"

    def getData (self, folder):
        # clinical description not needed
        # dataDir = os.path.join(folder, "journal.pone.0227703/")
        # inputFile = "pone.0227703.s011.xlsx"
        # targets = pd.read_excel(os.path.join(dataDir, inputFile))
        dataDir = os.path.join(folder, "journal.pone.0227703/")
        inputFile = "pone.0227703.s014.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        data["Target"] = data["Label"]
        data = data.drop(["Label"], axis = 1)
        data = data.reset_index(drop = True)
        return data



class Park2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0227315"
        self.mod = "US"
        self.tumor = "Thyroid cancer"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0227315/")
        inputFile = "pone.0227315.s003.xlsx"
        data = pd.read_excel(os.path.join(dataDir, inputFile), engine='openpyxl')
        target = data["pathological lateral LNM 0=no, 1=yes"]
        data = data.drop(["Patient No.", "pathological lateral LNM 0=no, 1=yes",
            "Sex 0=female, 1=male", "pathological central LNM 0=no, 1=yes"], axis = 1)
        data["Target"] = target
        data = data.reset_index(drop = True)
        return data



class Toivonen2019 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0217702"
        self.mod = "MRI"
        self.tumor = "Prostate cancer"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0217702/")
        inputFile = "lesion_radiomics.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        data["Target"] = np.asarray(data["gleason_group"] > 0.0, dtype = np.uint8)
        data = data.drop(["gleason_group", "id"], axis = 1)
        data = data.reset_index(drop = True)
        # toivonen2019 raises overflow with powertransformer
        # we thus scale it down, at least some features
        return data



class Hosny2018A (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"
        self.mod = "CT"
        self.tumor = "NSCLC"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "HarvardRT.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)

        # fix power transform overflow bugs
        ranking = data.std().sort_values(ascending = False).keys()
        L = list(data.keys())[0:825]+list(data.keys())[845:1004]
        data = data[L+["Target"]].copy()

        data = data.reset_index(drop = True)
        return data


class Hosny2018B (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"
        self.mod = "CT"
        self.tumor = "NSCLC"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "Maastro.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)
        data = data.reset_index(drop = True)
        return data


class Hosny2018C (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"
        self.mod = "CT"
        self.tumor = "NSCLC"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "Moffitt.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)
        data = data.reset_index(drop = True)
        # toivonen2019 raises overflow with powertransformer
        # we thus scale it down, at least some features
        # t = data["Target"].copy()
        # data = data/1000
        # data["Target"] = t
        return data



class Ramella2018 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0207455"
        self.mod = "CT"
        self.tumor = "NSCLC"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0207455/")
        inputFile = "pone.0207455.s001.arff"

        data = arff.loadarff(os.path.join(dataDir, inputFile))
        data = pd.DataFrame(data[0])
        data["Target"] = np.asarray(data['adaptive'], dtype = np.uint8)

        data = data.drop(['sesso', 'fumo', 'anni', 'T', 'N', "stadio", "istologia", "mutazione_EGFR", "mutazione_ALK", "adaptive"], axis = 1)
        data = data.reset_index(drop = True)
        return data



class Carvalho2018 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0192859"
        self.mod = "FDG+PET"
        self.tumor = "NSCLC"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0192859/")
        inputFile = "Radiomics.PET.features.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        # all patients that are lost to followup were at least followed for two
        # years. that means if we just binarize the followup time using two years
        # we get those who died or did not die within 2 years as binary label
        data["Target"] = (data["Survival"] < 2.0)*1
        data = data.drop(["Survival", "Status"], axis = 1)
        data = data.reset_index(drop = True)
        return data



class Saha2018 (DataSet):
    def __init__(self):
        self.ID = "data:10.1038/s41416-018-0185-8"
        self.mod = "DCE-MRI"
        self.tumor = "Breast"

    def getData (self, folder):
        dataDir = os.path.join(folder, "s41416-018-0185-8/")
        inputFile = "Imaging_Features.xlsx"
        data = pd.read_excel(os.path.join(dataDir, inputFile))

        inputFile = "Clinical_and_Other_Features.xlsx"
        targets = pd.read_excel(os.path.join(dataDir, inputFile), header = 1)
        t = targets[["Patient ID", "Mol Subtype"]]
        data = data.merge(t, on = "Patient ID")
        data["Target"] = (data["Mol Subtype"] != 0)*1
        data = data.drop(["Mol Subtype", "Patient ID"], axis = 1)
        data = data.reset_index(drop = True)
        return data


def imputeData (X, y):
    # imputation does not make any sense for columns with nearly all NA
    removeNACols = list(X.keys()[ (X.isna().sum(axis = 0) >= X.shape[0]//4) ])
    for k in removeNACols:
        X[k] = np.random.normal(0,1,X.shape[0])
    print ("Scrambled", len(removeNACols), "rows")

    simp = SimpleImputer(strategy="mean")
    X = pd.DataFrame(simp.fit_transform(X),columns = X.columns)

    # fix constant variables to be random, which reduces problems later
    np.random.seed(471)
    dropKeys = [z for z in X.keys() if len(set(X[z].values))==1]
    for k in dropKeys:
        X[k] = np.random.normal(0,1,X.shape[0])

    # scale upfront
    scaler = StandardScaler()
    scaler.fit (X.copy())
    X = pd.DataFrame(scaler.fit_transform(X.copy()),columns = X.columns)

    return X, y


 
if __name__ == '__main__':
    from parameters import dList
    datasets = {}
    table1 = []
    for d in dList:
        #d = dList[0]
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")
        data = datasets[d]
        X,_ = imputeData(data, None)
        # save for R
        X.to_csv(f"./data/data_{d}.csv")
        b = Counter(data["Target"])
        imbalance = b[0]/b[1]
        if imbalance < 1:
            imbalance = 1/imbalance
        imbalance = np.round(imbalance, 1)
        Npos = b[1]
        Nneg = b[0]
        table1.append({"Dataset": d, "N": data.shape[0], "d": data.shape[1], "N+": Npos, "N-": Nneg, "Balance": imbalance, "Modality": eval(d+"().mod"), \
                "Tumor type": eval(d+"().tumor"), "DOI": eval(d+"().ID").replace("data:", '') })
    table1 = pd.DataFrame(table1)
    table1.to_excel("./paper/Table_1.xlsx", index = False)
    print(table1)

#
