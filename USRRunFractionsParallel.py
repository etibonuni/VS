import os
import numpy as np
import conformer_utils as cu
import pandas as pd
from sklearn.metrics import roc_curve, auc
import time

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import evaluation as eval
import simClasses as scls


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


#homeDir = "/home/etienne/MScAI/dissertation/Conformers"
homeDir = "/home/ebon0023/projects/dissertation/Conformers"
#molfiles = get_immediate_subdirectories(homeDir)
#done = ["try1", "mk14", "aa2ar", "rxra"]
done = ["aces", "ace", "ada", "aldr", "ampc", "andr", "cdk2", "comt", "dyr", "egfr", "esr1", "fa10", "fgfr1", "gcr", "hivpr", "hivrt", "hmdh", "hs90a", "inha", "kith", "lkha4", "mcr", "nram", "parp1", "pde5a", "pgh1", "pgh2", "pnph", "pparg", "prgr", "pur2", "pygm", "sahh", "src", "thrb", "vgfr2"]
molfiles = [[homeDir+"/"+x+"/",x] for x in get_immediate_subdirectories(homeDir) if x not in done]
print(molfiles)



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
datasetPortion=[0.8, 0.6, 0.5, 0.3, 0.1, 0.05, 10]

results = []
for molNdx in range(0, len(molfiles)):

    molName = molfiles[molNdx][1]
    for portion in datasetPortion:
        portionResults=[]
        descTypes = ["usr", "esh", "es5"]
        descType=descTypes[2]
        if portion <= 1:
            print("Loading " + str(portion*100) +"% of " + molfiles[molNdx][1])
        else:
            print("Loading " + str(portion) + " actives from " + molfiles[molNdx][1])

        auc_esh=0
        auc_rank_esh=0
        mean_ef_esh=0

        try:
            print("Processing Electroshape 5-d - "+str(portion))
            t0=time.time()
            (sim_es5_ds, sim_paths_es5) = cu.loadDescriptors(molfiles[molNdx][0], portion, dtype="es5", active_decoy_ratio=-1, selection_policy="SEQUENTIAL", return_type="SEPARATE")
            simobj_es5 = scls.USRMoleculeSimParallel(sim_es5_ds, sim_paths_es5)
            usr_results_es5 = np.array(simobj_es5.runScreening(50)).transpose()

            #plotSimROC(sim_es5_ds, usr_results_es5, "es5_plot_"+molfiles[molNdx][1]+".pdf")
            # (auc_es5, mean_ef_es5) = eval.plotSimROC([l[2] for l in sim_ds], usr_results_es5,
            #                                  molName + "ElectroShape 5-d results",
            #                                  "es5_plot_"+molName + ".pdf")

            auc_es5 = eval.plotSimROC([l[2] for l in sim_ds], usr_results_es5,
                                             molName + " ElectroShape 5-d Sim ROC",
                                             "results/es5_sim_"+molName + ".pdf")
            auc_rank_es5 = eval.plotRankROC([l[2] for l in sim_ds], usr_results_es5,
                                             molName + " ElectroShape 5-d Rank ROC",
                                             "results/es5_rank_"+molName + ".pdf")
            mean_ef_es5 = eval.getMeanEFs([l[2] for l in sim_ds], usr_results_es5)
            t1 = time.time();
            t_es5 = t1-t0
        except:
            print("Error processing Electroshape 5-d for " + molfiles[molNdx][1])
            auc_es5=0
            auc_rank_es5=0
            mean_ef_es5=0

        resultLine = [molName, portion, 0, 0, 0, 0, 0, 0, 0, 0, auc_es5, auc_rank_es5, mean_ef_es5, t_es5]
        portionResults.append(resultLine)

    f1 = open("results/results_"+molName+".txt", "w")
    print(portionResults, file=f1)
    f1.close()

    # results.append(resultLine)
    # print("Results:")
    # print(results)
    # f1 = open('results_usr_parallel.txt', 'w')
    # print(results, file=f1)
    # f1.close()
