#from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import conformer_utils as cu
import os
import time
import evaluation as eval
from multiprocessing import Pool

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

homeDir = os.environ["DISSERTATION_HOME"]+"/Conformers/"

molfiles = [[homeDir+"/"+x+"/",x] for x in get_immediate_subdirectories(homeDir)]



def doMolGMM(molNdx):
    componentResults = []
    portionResults = []

    molName = molfiles[molNdx][1]  # [molfiles[molNdx].rfind("/", 0, -1)+1:-1]
    for portion in datasetPortion:
        t0 = time.time()
        descTypes = ["usr", "esh", "es5"]
        descType=descTypes[1]
        if portion <= 1:
            print("Loading " + str(portion*100) +"% of " + molfiles[molNdx][1])
        else:
            print("Loading " + str(portion) + " actives from " + molfiles[molNdx][1])

        (test_ds, test_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion*0.2, dtype=descType, active_decoy_ratio=-1, selection_policy="RANDOM", return_type="SEPERATE")
        numcols = test_ds[0][0].shape[1]-2

        folds = 3

        (n_fold_ds, n_fold_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion*0.8, dtype=descType, active_decoy_ratio=-1,
                                                       selection_policy="RANDOM", return_type="SEPARATE", exclusion_list=test_paths)

        (folds_list, excl_list) = cu.split(n_fold_ds, folds, policy="RANDOM")

        foldResults = []

        for fold in range(0, folds):

            val_ds = folds_list[fold]

            train_ds = None;

            for i in range(0, folds):
                if i != fold:
                    if train_ds is None:
                        train_ds = [r[0] for r in folds_list[i]]
                    else:
                        train_ds.append([r[0] for r in folds_list[i]])

            train_ds = cu.joinDataframes(train_ds)

            numcols = train_ds.shape[1]-2


            ann = MLPRegressor(max_iter=1000, early_stopping=True)

            ann.fit(train_ds.iloc[:, 0:numcols], ((train_ds["active"])).astype(int)*100)

            results = pd.DataFrame()

            results["score"] = [max(ann.predict(x[0].iloc[:, 0:numcols])) for x in val_ds]
            results["truth"] = [x[2] for x in val_ds]
            auc = eval.plotSimROC(np.array(results["truth"]), np.array([results["score"]]), "", None)
            mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["score"]]))
            foldResults.append((auc, mean_ef))


        print("X-Validation results: ")
        print(foldResults)

        if len(foldResults)>0:
            mean_auc_sim = np.mean([x[0] for x in foldResults])
            std_auc_sim = np.std(np.mean([x[0] for x in foldResults]))
            mean_mean_ef_1pc = np.mean([x[1][0.01] for x in foldResults])
            std_mean_ef_1pc = np.std([x[1][0.01] for x in foldResults])
            mean_mean_ef_5pc = np.mean([x[1][0.05] for x in foldResults])
            std_mean_ef_5pc = np.std([x[1][0.05] for x in foldResults])

            print("mean AUC=" + str(mean_auc_sim) +
                  ", std=" + str(std_auc_sim) +
                  ", mean EF(1%)=" + str(mean_mean_ef_1pc) +
                  ", std=" + str(std_mean_ef_1pc) +
                  ", mean EF(5%)=" + str(mean_mean_ef_5pc) +
                  ", std=" + str(std_mean_ef_5pc))

            componentResults.append((molName, portion, mean_auc_sim, std_auc_sim, mean_mean_ef_1pc,
                                     std_mean_ef_1pc, mean_mean_ef_5pc, std_mean_ef_5pc))
        else:
            print("X-Validation returned no results. Skipping training...")
            componentResults.append((molName, portion, 0, 0, 0, 0, 0, 0))



        train_ds = cu.lumpRecords(n_fold_ds)
        ann = MLPRegressor(max_iter=1000, early_stopping=True)
        ann.fit(train_ds.iloc[:, 0:numcols], ((train_ds["active"])).astype(int) * 100)

        results = pd.DataFrame()

        results["score"] = [max(ann.predict(x[0].iloc[:, 0:numcols])) for x in test_ds]
        results["truth"] = [x[2] for x in test_ds]#np.array(test_ds)[:, 2]

        auc_sim = eval.plotSimROC(results["truth"], [results["score"]],
                              molName+"[ANN, "+str(portion*100)+"%]",
                              "results/"+molName+"_ANN_sim_"+str(portion*100)+".pdf")
        auc_rank = eval.plotRankROC(results["truth"], [results["score"]],
                                    molName + "[ANN-" + str(portion * 100) + "%]",
                                    "results/" + molName + "_ANN_rank_" + str(portion*100) + ".pdf")

        mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["score"]]))

        print("AUC(Sim)="+str(auc))
        print("EF: ", mean_ef)
        t1 = time.time();

        portionResults.append((molName, portion, auc_sim, auc_rank, mean_ef, (t1-t0)))

        print("Time taken = "+str(t1-t0))

        print(componentResults)
        print(portionResults)

        f1 = open("results/results_ann_"+molName+".txt", 'w')
        print(componentResults, file=f1)
        print(portionResults, file=f1)
        f1.close()


datasetPortion=[1, 0.8, 0.6, 0.5, 0.3, 0.1, 0.05, 10]

componentResults = []
portionResults = []

numProcesses=4
p = Pool(numProcesses)

p.map(doMolGMM, range(0, len(molfiles)));