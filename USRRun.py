from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit import rdBase
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
import pandas as pd
from numpy.linalg import inv, det
from scipy.stats import norm
from sklearn.metrics import f1_score
import math
import pickle
import conformer_utils as cu
from glob import glob
from random import shuffle
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import simClasses as scls
from itertools import chain
import pandas as pd
import findspark

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

LOCAL_IP = "spark-master"

spark = SparkSession \
    .builder \
    .appName("Test Etienne JOB") \
    .master("spark://"+LOCAL_IP+":7077") \
    .config("spark.executor.cores", 2) \
    .config("spark.cores.max", 16) \
    .config("spark.python.worker.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executorEnv.SPARK_LOCAL_IP", LOCAL_IP) \
    .getOrCreate()

sc = spark.sparkContext
sc.addFile("simClasses.py")
sc.addFile("conformer_utils.py")

homeDir = "/home/etienne/MScAI/dissertation"
#homeDir = "/home/ubuntu/data_vol/projects/dissertation/conf_gen"
molfiles = [[homeDir+"/data3/Adenosine A2a receptor (GPCR)/", "actives_final.ism", "decoys_final.ism"],
            [homeDir+"/data3/Progesterone Receptor/", "actives_final.ism", "decoys_final.ism"],
            [homeDir+"/data3/Neuraminidase/", "actives_final.ism", "decoys_final.ism"],
            [homeDir+"/data3/Thymidine kinase/", "actives_final.ism", "decoys_final.ism"],
            [homeDir+"/data3/Leukotriene A4 hydrolase (Protease)/", "actives_final.ism", "decoys_final.ism"],
            [homeDir+"/data3/HIVPR/", "actives_final.ism", "decoys_final.ism"]]


def getEnrichmentFactor(threshold, ds, sort_by="prob", truth="truth"):
    sorted_ds = ds.sort_values(by=sort_by, ascending=False)
    # print(sorted_ds)
    top_thresh = sorted_ds.iloc[0:int(threshold * len(sorted_ds))]
    # print(top_thresh)

    num_actives = sum(top_thresh[truth])
    print("Number of actives found in top ", (threshold * 100), "%=", num_actives)
    print("Number of actives expected =", (float(sum(sorted_ds[truth])) / len(sorted_ds)) * len(top_thresh))
    ef = float(num_actives) / len(top_thresh) / (float(sum(sorted_ds[truth])) / len(sorted_ds))

    return ef

numActives=10

molNdx=1

(sim_ds, sim_paths) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="usr", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")
(sim_es_ds, sim_paths_es) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="esh", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")
(sim_es5_ds, sim_paths_es5) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="es5", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")


def plotROCCurve(truth, preds, label, fileName):
    fpr, tpr, _ = roc_curve(truth.astype(int), preds)

    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ('+label+")")
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(fileName)

def plotSimROC(mol_ds, results, fileName):
    labels = [l[2] for l in mol_ds]

    simresults = np.concatenate([np.array(list(zip(r, labels))) for r in results])
    # print(simresults.shape)
    plotROCCurve(simresults[:, 1], simresults[:, 0], molfiles[0][0], fileName)

    ef = [getEnrichmentFactor(0.01, pd.DataFrame(data=list(zip(r, labels)), columns=("sim", "truth")), sort_by="sim",
                              truth="truth") for r in results]
    print(ef)
    ef_mean = np.mean(ef)

    # sim_pd = pd.DataFrame(data=simresults, columns=("sim", "truth"))
    # print(getEnrichmentFactor(0.01, sim_pd, sort_by="sim", truth="truth"))
    print("Mean EF@1%=", ef_mean)

simobj = scls.USRMoleculeSim(sim_ds, sim_paths)
usr_results = scls.runSparkScreening(sc, simobj)

simobj_es = scls.USRMoleculeSim(sim_es_ds, sim_paths_es)
usr_results_esh = scls.runSparkScreening(sc, simobj_es)

simobj_es5 = scls.USRMoleculeSim(sim_es5_ds, sim_paths_es5)
usr_results_es5 = scls.runSparkScreening(sc, simobj_es5)

plotSimROC(sim_ds, usr_results, "usr_plot.pdf")
plotSimROC(sim_es_ds, usr_results_esh, "esh_plot.pdf")
plotSimROC(sim_es5_ds, usr_results_es5, "es5_plot.pdf")