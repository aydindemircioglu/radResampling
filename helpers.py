#
from math import sqrt
import os
import shutil


def getColor (name):
    color = "black"
    if name == "Random Undersampling":
        color = "red"
    if "Edited NN [k=" in name:
        color = "red"
    if "All k-NN [k=" in name:
        color = "red"
    if "Tomek links" in name:
        color = "red"
    if "Random Oversampling" in name:
        color = "blue"
    if "SMOTE [k=" in name:
        color = "blue"
    if "k-Means-SMOTE [k=" in name:
        color = "blue"
    if "SVM-SMOTE [k=" in name:
        color = "blue"
    if "SMOTE+Edited NN [k=" in name:
        color = "#004800"
    if "SMOTE+Tomek links [k=" in name:
        color = "#004800"
    if name == "None":
        color = "black"
    return color


def getName (s, detox = False):
    v = eval(s)[0]
    if "RUS" == v[0]:
        name = "Random Undersampling"
    if "ENN" == v[0]:
        name = f"Edited NN [k={v[1]['k']}]"
    if "AllKNN" == v[0]:
        name = f"All k-NN [k={v[1]['k']}]"
    if "TomekLinks" == v[0]:
        name = f"Tomek links"
    if "ROS" == v[0]:
        name = "Random Oversampling"
    if "SMOTE" == v[0]:
        name = f"SMOTE [k={v[1]['k']}]"
    if "KMeansSMOTE" == v[0]:
        name = f"k-Means-SMOTE [k={v[1]['k']}]"
    if "SVMSMOTE" == v[0]:
        name = f"SVM-SMOTE [k={v[1]['k']}]"
    if "SMOTEENN" == v[0]:
        name = f"SMOTE+Edited NN [k={v[1]['k']}, n={v[1]['n']}]"
    if "SMOTETomek" == v[0]:
        name = f"SMOTE+Tomek links [k={v[1]['k']}]"
    if "None" == v[0]:
        name = "None"
    if detox == True:
        name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "").replace(",", "_")
    return name



def recreatePath (path, create = True):
    print ("Recreating path ", path)
    try:
        shutil.rmtree (path)
    except:
        pass

    if create == True:
        try:
            os.makedirs (path)
        except:
            pass
    print ("Done.")



def findOptimalCutoff (fpr, tpr, threshold, verbose = False):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    fpr, tpr, threshold

    Returns
    -------
    list type, with optimal cutoff value

    """

    # own way
    minDistance = 2
    bestPoint = (2,-1)
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p

    if verbose == True:
        print ("BEST")
        print (minDistance)
        print (bestPoint)
    sensitivity = bestPoint[1]
    specificity = 1 - bestPoint[0]
    return sensitivity, specificity


#
