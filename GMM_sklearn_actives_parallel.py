from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
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

print(molfiles)

def doMolGMM(molNdx):
    molName = molfiles[molNdx][1]  # [molfiles[molNdx].rfind("/", 0, -1)+1:-1]
    for portion in datasetPortion:
        try:
            descTypes = ["usr", "esh", "es5"]
            descType = descTypes[1]
            if portion <= 1:
                print("Loading " + str(portion * 100) + "% of " + molfiles[molNdx][1])
            else:
                print("Loading " + str(portion) + " actives from " + molfiles[molNdx][1])

            (test_ds, test_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion * 0.2, dtype=descType,
                                                       active_decoy_ratio=-1, selection_policy="RANDOM",
                                                       return_type="SEPERATE")
            numcols = test_ds[0][0].shape[1] - 2

            componentsValues = [1, 10, 50, 100, 1000]
            folds = 5

            (n_fold_ds, n_fold_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion * 0.8, dtype=descType,
                                                           active_decoy_ratio=-1,
                                                           selection_policy="RANDOM", return_type="SEPARATE",
                                                           exclusion_list=test_paths)

            (folds_list, excl_list) = cu.split(n_fold_ds, folds, policy="RANDOM")

            componentResults = []

            for components in componentsValues:
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

                    numcols = train_ds.shape[1] - 2

                    train_a = train_ds[train_ds["active"] == True]
                    # train_d = train_ds[train_ds["active"]==False]

                    if len(train_a) > components:
                        # print("Generating GMM for actives...")
                        G_a = GaussianMixture(n_components=components, covariance_type="full").fit(
                            train_a.iloc[:, 0:numcols], train_a.iloc[:, numcols])

                        results = pd.DataFrame()

                        print(numcols)
                        results["a_score"] = [G_a.score(x[0].iloc[:, 0:numcols]) for x in
                                              val_ds]  # map(lambda x: G_a.score(x[0].iloc[:, 0:12]), test_ds)
                        results["truth"] = [x[2] for x in val_ds]  # np.array(val_ds)[:,2]

                        auc = eval.plotSimROC(np.array(results["truth"]), np.array([results["a_score"]]), "", None)
                        mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["a_score"]]))

                        foldResults.append((auc, mean_ef))
                    else:
                        print("Training samples(" + str(len(train_a)) + ") < GMM components(" + str(
                            components) + ") -> cannot train.")
                        break
                        # foldResults.append(0)

                print("X-Validation results, num components = " + str(components) + ": ")
                print(foldResults)

                if len(foldResults) > 0:
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

                    componentResults.append((molName, portion, components, mean_auc_sim, std_auc_sim, mean_mean_ef_1pc,
                                             std_mean_ef_1pc, mean_mean_ef_5pc, std_mean_ef_5pc))
                else:
                    print(
                        "X-Validation returned no results for " + str(components) + " components. Skipping training...")
                    componentResults.append((molName, portion, components, 0, 0, 0, 0, 0, 0))
            # print(componentResults)

            xvalResults.extend(componentResults)
            # Find best score
            aucs_rank = [x[5] for x in componentResults]

            best_components = componentsValues[np.argmax(aucs_rank)]
            print("Best-score compnents no.: " + str(best_components))

            (train_ds, train_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion * 0.8, dtype=descType,
                                                         active_decoy_ratio=0,
                                                         selection_policy="RANDOM", return_type="LUMPED",
                                                         exclusion_list=test_paths)

            # molName = molfiles[molNdx][1]#[molfiles[molNdx].rfind("/", 0, -1)+1:-1]
            if len(train_ds) > best_components:
                t0 = time.time()
                G_a = GaussianMixture(n_components=best_components, covariance_type="full").fit(
                    train_ds.iloc[:, 0:numcols], train_ds.iloc[:, numcols])
                results = pd.DataFrame()
                results["a_score"] = [G_a.score(x[0].iloc[:, 0:numcols]) for x in test_ds]
                results["truth"] = [x[2] for x in test_ds]  # np.array(test_ds)[:, 2]
                auc = eval.plotSimROC(results["truth"], [results["a_score"]],
                                      molName + "[GMM-" + str(components) + " components(Similarity), " + str(
                                          portion * 100) + "%]",
                                      molName + "_GMM_sim_" + str(components) + "_" + str(portion * 100) + ".pdf")
                auc_rank = eval.plotRankROC(results["truth"], [results["a_score"]],
                                            molName + "[GMM-" + str(components) + " components(Similarity), " + str(
                                                portion * 100) + "%]",
                                            molName + "_GMM_sim_" + str(components) + "_" + str(portion * 100) + ".pdf")
                mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["a_score"]]))
                t1 = time.time()
            else:
                auc = 0
                mean_ef = 0

            print("Final results, num components = ", str(components) + ": ")
            print("AUC=" + str(auc))
            print("EF: ", mean_ef)

            portionResults.append((molName, portion, best_components, auc, auc_rank, mean_ef, t1 - t0))
        except:
            portionResults.append((molName, portion, 0, 0, 0, 0, 0))

        f1 = open('results/results_gmm_"+molName+".txt', 'w')
        print(xvalResults, file=f1)
        print(portionResults, file=f1)
        f1.close()

        full_train_dss = [x[0] for x in test_ds]
        full_train_dss.append([x[0] for x in n_fold_ds])
        full_train_ds = cu.joinDataframes(full_train_dss)
        G_a = GaussianMixture(n_components=best_components, covariance_type="full").fit(
            full_train_dss.iloc[:, 0:numcols],
            full_train_dss.iloc[:, numcols])
        import pickle
        mdlf = open(molName + "_GMM.pkl", "w")
        pickle.dump(G_a, mdlf)
        mdlf.close()
        print("Saved model for " + molName + " to disk")


datasetPortion=[1, 0.8, 0.6, 0.5, 0.3, 0.1, 0.05, 10]

componentResults = []
portionResults = []
xvalResults=[]

numProcesses=50
p = Pool(numProcesses)

p.map(doMolGMM, range(0, len(molfiles)));

#for molNdx in range(0, len(molfiles)):
