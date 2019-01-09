#from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import conformer_utils as cu
import os
import time
import evaluation as eval

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

homeDir = os.environ["DISSERTATION_HOME"]
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
#datasetPortion=[1]

portionResults = []

for molNdx in range(0, len(molfiles)):
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
        componentResults=[]

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

            #print(train_ds.head())

            ann = MLPRegressor()

            ann.fit(train_ds.iloc[:, 0:numcols], ((train_ds["active"])).astype(int)*100)

            results = pd.DataFrame()

            results["score"] = [max(ann.predict(x[0].iloc[:, 0:numcols])) for x in val_ds]
            results["truth"] = [x[2] for x in val_ds]
            auc = eval.plotSimROC(np.array(results["truth"]), np.array([results["score"]]), "", None)
            auc_rank = eval.plotRankROC(np.array(results["truth"]), np.array([results["score"]]), "", None)
            mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["score"]]))
            foldResults.append(auc)


        print("X-Validation results: ")
        print(foldResults)

        if len(foldResults)>0:
            print("mean AUC=" + str(np.mean(foldResults)) + ", std=" + str(np.std(foldResults)))

            componentResults.append((auc, mean_ef))
        else:
            print("X-Validation returned no results. Skipping training...")
            componentResults.append((0, 0))


        train_ds = cu.lumpRecords(n_fold_ds)
        ann = MLPRegressor(max_iter=500)
        #ann.fit(train_ds.iloc[:, 0:numcols], train_ds.iloc[:, numcols])
        ann.fit(train_ds.iloc[:, 0:numcols], ((train_ds["active"])).astype(int) * 100)
    #    G_a = GaussianMixture(n_components=best_components, covariance_type="full").fit(train_ds.iloc[:, 0:numcols],
    #                                                                               train_ds.iloc[:, numcols])

        results = pd.DataFrame()

        results["score"] = [max(ann.predict(x[0].iloc[:, 0:numcols])) for x in test_ds]
        #results["a_score"] = [G_a.score(x[0].iloc[:, 0:numcols]) for x in test_ds]
        results["truth"] = [x[2] for x in test_ds]#np.array(test_ds)[:, 2]
        molName = molfiles[molNdx][1]#[molfiles[molNdx].rfind("/", 0, -1)+1:-1]
        auc = eval.plotSimROC(results["truth"], [results["score"]], molName+"[ANN, "+str(portion*100)+"%]", molName+"_ANN_sim_"+str(portion*100)+".pdf")
        auc_rank = eval.plotRankROC(results["truth"], [results["score"]], molName+"[ANN, "+str(portion*100)+"%]", molName+"_ANN_rank_"+str(portion*100)+".pdf")
        mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["score"]]))

        #print("Final results, num components = ", str(components)+": ")
        print("AUC(Sim)="+str(auc))
        print("AUC(Rank)="+str(auc_rank))
        print("EF: ", mean_ef)

        portionResults.append((molName, auc, auc_rank, mean_ef))
        t1 = time.time();
        print("Time taken = "+str(t1-t0))

        print(portionResults)

#        print("F1 score=", f1_score(np.array(np.array(test_ds)[:,2], dtype=bool), results["predict"]))
#print("Generating GMM for decoys...")
#G_d = GaussianMixture(n_components=100, covariance_type="full").fit(train_d.iloc[:,0:numcols], train_d.iloc[:,numcols])
#print("Done")