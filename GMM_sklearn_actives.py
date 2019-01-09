from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
import pandas as pd
import conformer_utils as cu
import os
import evaluation as eval

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

homeDir = os.environ["DISSERTATION_HOME"]
#homeDir="/media/etienne/Elements/Conformers"
#homeDir = "/home/etienne/MScAI/dissertation/Conformers"
#homeDir = "/home/ubuntu/data_vol/projects/dissertation"
molfiles = [[homeDir+"/"+x+"/",x] for x in get_immediate_subdirectories(homeDir)]

# homeDir = "/home/ubuntu/data_vol/projects/dissertation"
# molfiles = [[homeDir + "/Conformers/Adenosine A2a receptor (GPCR)/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir + "/Conformers/Progesterone Receptor/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir + "/Conformers/Neuraminidase/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir + "/Conformers/Thymidine kinase/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir + "/Conformers/Leukotriene A4 hydrolase (Protease)/", "actives_final.ism", "decoys_final.ism"],
#             [homeDir + "/Conformers/HIVPR/", "actives_final.ism", "decoys_final.ism"]]



#from sklearn.ensemble import VotingClassifier

# (test_ds, test_paths) = cu.loadDescriptors(molfiles[2], 10, dtype="usr", active_decoy_ratio=0, selection_policy="RANDOM", return_type="SEPERATE")
# ds1 = cu.split(test_ds, 2, policy="SEQUENTIAL")
# ds2 = cu.lumpRecords(test_ds)
#
# (p1, p1_sel) = cu.takePortion(test_ds, 0.3, selection_policy="RANDOM", return_type="LUMPED")
# (p2, p2_sel) = cu.takePortion(test_ds, 0.5, selection_policy="RANDOM", return_type="LUMPED", exclusion_list=p1_sel)

datasetPortion=[1, 0.8, 0.6, 0.5, 0.3, 0.1, 0.05, 10]
#datasetPortion=[0.1, 0.05, 10]

portionResults = []

for molNdx in range(0, len(molfiles)):
    for portion in datasetPortion:
        descTypes = ["usr", "esh", "es5"]
        descType=descTypes[1]
        if portion <= 1:
            print("Loading " + str(portion*100) +"% of " + molfiles[molNdx])
        else:
            print("Loading " + str(portion) + " actives from " + molfiles[molNdx])

        (test_ds, test_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion*0.2, dtype=descType, active_decoy_ratio=-1, selection_policy="RANDOM", return_type="SEPERATE")
        numcols = test_ds[0][0].shape[1]-2

        componentsValues=[1, 10, 50, 100, 1000]
        folds = 10
        componentResults=[]

        (n_fold_ds, n_fold_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion*0.8, dtype=descType, active_decoy_ratio=-1,
                                                       selection_policy="RANDOM", return_type="SEPARATE", exclusion_list=test_paths)

        (folds_list, excl_list) = cu.split(n_fold_ds, folds, policy="RANDOM")

        for components in componentsValues:
            foldResults = []

            for fold in range(0, folds):
                # (val_ds, val_paths) = cu.loadDescriptors(molfiles[molNdx], portion*0.11, dtype=descType, active_decoy_ratio=-1,
                #                                              selection_policy="RANDOM", return_type="SEPARATE",
                #                                              exclusion_list=test_paths)
                #
                # excl_paths = test_paths+val_paths
                # (train_ds, train_paths) = cu.loadDescriptors(molfiles[molNdx], portion*0.7, dtype=descType, active_decoy_ratio=0,
                #                                              selection_policy="RANDOM", return_type="LUMPED",
                #                                              exclusion_list=excl_paths)

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

                train_a = train_ds[train_ds["active"]==True]
                #train_d = train_ds[train_ds["active"]==False]

                if len(train_a) > components:
                    #print("Generating GMM for actives...")
                    G_a = GaussianMixture(n_components=components, covariance_type="full").fit(train_a.iloc[:,0:numcols], train_a.iloc[:,numcols])

                    results=pd.DataFrame()


                    print(numcols)
                    results["a_score"] = [G_a.score(x[0].iloc[:, 0:numcols]) for x in val_ds]#map(lambda x: G_a.score(x[0].iloc[:, 0:12]), test_ds)
                    results["truth"] = [x[2] for x in val_ds]#np.array(val_ds)[:,2]

                    #(auc, mean_ef) = eval.plotSimROC(np.array(results["truth"]), np.array([results["a_score"]]), "", None)

                    auc = eval.plotSimROC(np.array(results["truth"]), np.array([results["a_score"]]), "", None)
                    auc_rank = eval.plotRankROC(np.array(results["truth"]), np.array([results["a_score"]]), "", None)
                    mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["a_score"]]))

                    foldResults.append(auc)
                else:
                    print("Training samples("+str(len(train_a))+") < GMM components("+str(components)+") -> cannot train.")
                    break
                    #foldResults.append(0)


            print("X-Validation results, num components = "+str(components)+": ")
            print(foldResults)

            if len(foldResults)>0:
                print("mean AUC=" + str(np.mean(foldResults)) + ", std=" + str(np.std(foldResults)))

                componentResults.append((auc, mean_ef))
            else:
                print("X-Validation returned no results for "+str(components) + " components. Skipping training...")
                componentResults.append((0, 0))
        #print(componentResults)

        # Find best score
        aucs = [x[0] for x in componentResults]

        best_components = componentsValues[np.argmax(aucs)]
        print("Best-score compnents no.: "+str(best_components))

        (train_ds, train_paths) = cu.loadDescriptors(molfiles[molNdx], portion*0.8, dtype=descType, active_decoy_ratio=0,
                                                 selection_policy="RANDOM", return_type="LUMPED",
                                                 exclusion_list=test_paths)

        G_a = GaussianMixture(n_components=best_components, covariance_type="full").fit(train_ds.iloc[:, 0:numcols],
                                                                                   train_ds.iloc[:, numcols])

        results = pd.DataFrame()

        results["a_score"] = [G_a.score(x[0].iloc[:, 0:numcols]) for x in test_ds]
        results["truth"] = [x[2] for x in test_ds]#np.array(test_ds)[:, 2]
        molName = molfiles[molNdx][1]#[molfiles[molNdx].rfind("/", 0, -1)+1:-1]
        #(auc, mean_ef) = eval.plotSimROC(results["truth"], [results["a_score"]], molName+"[GMM-"+str(components)+" components, "+str(portion*100)+"%]", molName+"_GMM_"+str(components)+"_"+str(portion*100)+".pdf")
        auc = eval.plotSimROC(results["truth"], [results["a_score"]], molName+"[GMM-"+str(components)+" components(Similarity), "+str(portion*100)+"%]", molName+"_GMM_sim_"+str(components)+"_"+str(portion*100)+".pdf")
        auc_rank = eval.plotRankROC(results["truth"], [results["a_score"]], molName+"[GMM-"+str(components)+" components(Rank), "+str(portion*100)+"%]", molName+"_GMM_rank_"+str(components)+"_"+str(portion*100)+".pdf")
        mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["a_score"]]))


        print("Final results, num components = ", str(components)+": ")
        print("AUC="+str(auc))
        print("EF: ", mean_ef)

        portionResults.append((molName, best_components, auc, auc_rank, mean_ef))

        print(portionResults)

#        print("F1 score=", f1_score(np.array(np.array(test_ds)[:,2], dtype=bool), results["predict"]))
#print("Generating GMM for decoys...")
#G_d = GaussianMixture(n_components=100, covariance_type="full").fit(train_d.iloc[:,0:numcols], train_d.iloc[:,numcols])
#print("Done")