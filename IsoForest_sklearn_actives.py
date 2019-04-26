#from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import conformer_utils as cu
import os
import time
import evaluation as eval
from sklearn.metrics import f1_score

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

homeDir = os.environ["DISSERTATION_HOME"]+"/Conformers/"

molfiles = [[homeDir+"/"+x+"/",x] for x in get_immediate_subdirectories(homeDir)]

print(molfiles)

datasetPortion=[1, 0.8, 0.6, 0.5, 0.3, 0.1, 0.05, 10]
params=[500, 100, 10]

componentResults = []
xvalResults=[]
portionResults = []

done = ["aa2ar", "aldr", "comt", "fa10", "hivrt", "kith", "parp1", "pnph", "pygm", "thrb", "ace", "ampc", "dyr", "fgfr1", "hmdh", "lkha4", "pde5a", "pparg", "rxra", "try1", "aces", "andr", "egfr", "gcr", "hs90a", "mcr", "pgh1", "prgr", "sahh", "try1", "ada", "cdk2", "esr1", "hivpr", "inha", "mk14", "pgh2", "pur2", "src", "vgfr2", "try1_2"]
for molNdx in range(0, len(molfiles)):
    molName = molfiles[molNdx][1]  # [molfiles[molNdx].rfind("/", 0, -1)+1:-1]
    if molName in done:
        continue

    for portion in datasetPortion:
        try:
            descTypes = ["usr", "esh", "es5"]
            descType=descTypes[1]
            if portion <= 1:
                print("Loading " + str(portion*100) +"% of " + molfiles[molNdx][1])
            else:
                print("Loading " + str(portion) + " actives from " + molfiles[molNdx][1])

            (test_ds, test_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion*0.2, dtype=descType, active_decoy_ratio=-1, selection_policy="RANDOM", return_type="SEPERATE")
            numcols = test_ds[0][0].shape[1]-2

            folds = 5

            (n_fold_ds, n_fold_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion*0.8, dtype=descType, active_decoy_ratio=-1,
                                                           selection_policy="RANDOM", return_type="SEPARATE", exclusion_list=test_paths)

            (folds_list, excl_list) = cu.split(n_fold_ds, folds, policy="RANDOM")

            componentResults = []
            for param in params:
                foldResults = []


                for fold in range(0, folds):
                    try:
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

                        clf = IsolationForest(n_estimators=param, n_jobs=-1)

                        train_a = train_ds[train_ds["active"] == True]

                        clf.fit(train_a.iloc[:, 0:numcols], None)


                        results = pd.DataFrame()

                        results["score"] = [max(clf.decision_function(x[0].iloc[:, 0:numcols]).ravel()) for x in val_ds]

                        results["truth"] = [x[2] for x in val_ds]

                        auc = eval.plotSimROC(np.array(results["truth"]), np.array([results["score"]]), "", None)
                        mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["score"]]), eval_method="sim")
                        foldResults.append((auc, mean_ef))
                    except:
                        foldResults.append((0,{0.01:0, 0.05:0}))

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

                    componentResults.append((molName, portion, param, mean_auc_sim, std_auc_sim, mean_mean_ef_1pc,
                                             std_mean_ef_1pc, mean_mean_ef_5pc, std_mean_ef_5pc))
                else:
                    print("X-Validation returned no results. Skipping training...")
                    componentResults.append((molName, portion, param, 0, 0, 0, 0, 0, 0))

        except:
            componentResults.append((molName, portion, param, 0, 0, 0, 0, 0, 0))


        xvalResults.extend(componentResults)

        # Find best score
        aucs_rank = [x[5] for x in componentResults]

        best_estimators = params[np.argmax(aucs_rank)]
        print("Best-score estimators no.: "+str(best_estimators))

        train_ds = cu.lumpRecords(n_fold_ds)

        t0 = time.time()
        clf = IsolationForest(n_estimators=best_estimators, n_jobs=-1)

        train_a = train_ds[train_ds["active"] == True]

        clf.fit(train_a.iloc[:, 0:numcols], None)

        results = pd.DataFrame()

        results["score"] = [max(clf.decision_function((x[0].iloc[:, 0:numcols]))) for x in test_ds]
        results["truth"] = [x[2] for x in test_ds]#np.array(test_ds)[:, 2]

        auc = eval.plotSimROC(results["truth"], [results["score"]],
                              molName + "[IsoForest, " + str(portion * 100) + "%]",
                              molName + "_IsoForest_sim_" + str(portion * 100) + ".pdf")
        auc_rank = eval.plotRankROC(results["truth"], [results["score"]],
                                    molName + "[IsoForest, " + str(portion * 100) + "%]",
                                    molName + "_IsoForest_rank_" + str(portion * 100) + ".pdf")

        auc = eval.plotSimROC(results["truth"], [results["score"]], molName+"[IsoForest, "+str(portion*100)+"%]", molName+"_IsoForest_sim_"+str(portion*100)+".pdf")
        mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["score"]]), eval_method="sim")

        print("AUC(Sim)="+str(auc))
        print("EF: ", mean_ef)

        t1 = time.time();
        print("Time taken = "+str(t1-t0))

        portionResults.append((molName, portion, best_estimators, auc, auc_rank, mean_ef, t1-t0))

        print(xvalResults)
        print(portionResults)

        f1 = open('results_isoForest.txt', 'w')
        print(xvalResults, file=f1)
        print(portionResults, file=f1)
        f1.close()

    full_train_dss = [x[0] for x in test_ds]
    full_train_dss.append([x[0] for x in n_fold_ds])
    full_train_ds = cu.joinDataframes(full_train_dss)
    clf = IsolationForest(n_estimators=best_estimators, n_jobs=-1)

    G_a = clf.fit(full_train_ds.iloc[:, 0:numcols], full_train_ds.iloc[:, numcols])

    import pickle
    mdlf = open(molName + "_IsoForest.pkl", "wb")
    pickle.dump(G_a, mdlf)
    mdlf.close()

    print("Saved model for "+molName+" to disk")
        #except:
        #    print("Eception occurred.")
        #    pass
