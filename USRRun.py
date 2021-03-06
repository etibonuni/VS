import os
import numpy as np
import conformer_utils as cu
import pandas as pd
import findspark
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import evaluation as eval
import simClasses as scls

findspark.init()
from pyspark.sql import SparkSession

def initSpark():
    LOCAL_IP = "spark-master"

    spark = SparkSession \
        .builder \
        .appName("Test Etienne JOB") \
        .master("spark://"+LOCAL_IP+":7077") \
        .config("spark.cores.max", 22) \
        .config("spark.executorEnv.SPARK_LOCAL_IP", LOCAL_IP) \
        .getOrCreate()
#        .config("spark.executor.heartbeatInterval", "100s") \
#        .config("spark.network.timeout", "400s") \
    
    sc = spark.sparkContext
    sc.addFile("simClasses.py")
    sc.addFile("conformer_utils.py")
    
    return sc

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


#homeDir = "/home/etienne/MScAI/dissertation/Conformers"
homeDir = "/home/ubuntu/data_vol/projects/dissertation/Conformers"
#molfiles = get_immediate_subdirectories(homeDir)
molfiles = [[homeDir+"/"+x+"/",x] for x in get_immediate_subdirectories(homeDir)]
print(molfiles)
# molfiles = [[homeDir+"/Conformers/Adenosine A2a receptor (GPCR)/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir+"/Conformers/Progesterone Receptor/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir+"/Conformers/Neuraminidase/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir+"/Conformers/Thymidine kinase/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir+"/Conformers/Leukotriene A4 hydrolase (Protease)/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir+"/Conformers/HIVPR/", "actives_final.ism", "decoys_final.ism"]]


def getEnrichmentFactor(threshold, ds, sort_by="prob", truth="truth"):


    sorted_ds = ds.sort_values(by=sort_by, ascending=False)
    # print(sorted_ds)

    top_thresh = sorted_ds.iloc[0:int(threshold * len(sorted_ds))]
    # print(top_thresh)
    
    num_actives = sum(top_thresh[truth])
    #print("Number of actives found in top ", (threshold * 100), "%=", num_actives)
    expected = (float(sum(sorted_ds[truth])) / len(sorted_ds)) * len(top_thresh)
    #print("Number of actives expected =", expected)
    ef = float(num_actives)/expected
    #ef = float(num_actives) / len(top_thresh) / (float(sum(sorted_ds[truth])) / len(sorted_ds))

    return ef

#print(sim_ds[0][0])
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

    thresholds = [0.01, 0.05]

    for threshold in thresholds:
        ef = [getEnrichmentFactor(threshold, pd.DataFrame(data=list(zip(r, labels)), columns=("sim", "truth")), sort_by="sim",
                              truth="truth") for r in results]
    
#        print(ef)
        ef_mean = np.mean(ef)

#        ef = getEnrichmentFactor(threshold, pd.DataFrame(data=simresults, columns=("sim", "truth")), sort_by="sim", truth="truth")
        # sim_pd = pd.DataFrame(data=simresults, columns=("sim", "truth"))
        # print(getEnrichmentFactor(0.01, sim_pd, sort_by="sim", truth="truth"))
        print("Mean EF@"+str((threshold * 100))+"%=", ef_mean)

numActives=1

molNdx=0

#(sim_ds, sim_paths) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="usr", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")
#(sim_es_ds, sim_paths_es) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="esh", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")
#(sim_es5_ds, sim_paths_es5) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="es5", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")

results = []
for molNdx in range(0, len(molfiles)):

    molName = molfiles[molNdx][1]
    try:
        print("Processing "+molfiles[molNdx][0])
        print("Processing USR")
        sc = initSpark()
        (sim_ds, sim_paths) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="usr", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")
        simobj = scls.USRMoleculeSim(sim_ds, sim_paths)
        usr_results = np.array(simobj.runSparkScreening(sc)).transpose()
        sc.stop()
        #plotSimROC(sim_ds, usr_results, "usr_plot_"+molfiles[molNdx][1]+".pdf")
        auc_usr = eval.plotSimROC([l[2] for l in sim_ds], usr_results,
                                         molName + " USR Sim ROC",
                                         "usr_sim_"+molName + ".pdf")
        auc_rank_usr = eval.plotRankROC([l[2] for l in sim_ds], usr_results,
                                         molName + " USR Rank ROC",
                                         "usr_rank_"+molName + ".pdf")
        mean_ef_usr = eval.getMeanEFs([l[2] for l in sim_ds], usr_results)
    except:
        print("Error processing USR for " + molfiles[molNdx][1])
        auc_usr=0
        auc_rank_usr=0
        mean_ef_usr=0

#    try:
#        print("Processing Electroshape 4-d")
#        sc = initSpark()
#        (sim_es_ds, sim_paths_es) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="esh", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")
#        simobj_es = scls.USRMoleculeSim(sim_es_ds, sim_paths_es)
#        usr_results_esh = np.array(simobj_es.runSparkScreening(sc)).transpose()
#        sc.stop()
#        #plotSimROC(sim_es_ds, usr_results_esh, "esh_plot_"+molfiles[molNdx][1]+".pdf")
#        # (auc_esh, mean_ef_esh) = eval.plotSimROC([l[2] for l in sim_ds], usr_results_esh,
#        #                                  molName + "ElectroShape 4-d results",
#        #                                  "esh_plot_"+molName + ".pdf")
#
#        auc_esh = eval.plotSimROC([l[2] for l in sim_ds], usr_results_esh,
#                                         molName + " ElectroShape 4-d Sim ROC",
#                                         "esh_sim_"+molName + ".pdf")
#        auc_rank_esh = eval.plotRankROC([l[2] for l in sim_ds], usr_results_esh,
#                                         molName + " ElectroShape 4-d Rank ROC",
#                                         "esh_rank_"+molName + ".pdf")
#        mean_ef_esh = eval.getMeanEFs([l[2] for l in sim_ds], usr_results_esh)
#    except:
#        print("Error processing Electroshape 4-d for " + molfiles[molNdx][1])
        
    auc_esh=0
    auc_rank_esh=0
    mean_ef_esh=0

    try:
        print("Processing Electroshape 5-d")
        sc = initSpark()
        (sim_es5_ds, sim_paths_es5) = cu.loadDescriptors(molfiles[molNdx][0], numActives, dtype="es5", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")
        simobj_es5 = scls.USRMoleculeSim(sim_es5_ds, sim_paths_es5)
        usr_results_es5 = np.array(simobj_es5.runSparkScreening(sc)).transpose()
        sc.stop()
        #plotSimROC(sim_es5_ds, usr_results_es5, "es5_plot_"+molfiles[molNdx][1]+".pdf")
        # (auc_es5, mean_ef_es5) = eval.plotSimROC([l[2] for l in sim_ds], usr_results_es5,
        #                                  molName + "ElectroShape 5-d results",
        #                                  "es5_plot_"+molName + ".pdf")

        auc_es5 = eval.plotSimROC([l[2] for l in sim_ds], usr_results_es5,
                                         molName + " ElectroShape 5-d Sim ROC",
                                         "es5_sim_"+molName + ".pdf")
        auc_rank_es5 = eval.plotRankROC([l[2] for l in sim_ds], usr_results_es5,
                                         molName + " ElectroShape 5-d Rank ROC",
                                         "es5_rank_"+molName + ".pdf")
        mean_ef_es5 = eval.getMeanEFs([l[2] for l in sim_ds], usr_results_es5)
    except:
        print("Error processing Electroshape 5-d for " + molfiles[molNdx][1])
        auc_es5=0
        auc_rank_es5=0
        mean_ef_es5=0

    results.append([molName, auc_usr, auc_rank_usr, mean_ef_usr, auc_esh, auc_rank_esh, mean_ef_esh, auc_es5, auc_rank_es5, mean_ef_es5])
    print("Results:")

print(results)
