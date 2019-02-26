# from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
import pandas as pd
import conformer_utils as cu
import os
import time
import evaluation as eval

from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras import regularizers


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


homeDir = os.environ["DISSERTATION_HOME"]+"/Conformers/"
print(homeDir)
molfiles = [[homeDir + "/" + x + "/", x] for x in get_immediate_subdirectories(homeDir)]


def getKerasNNModel(descDim, hiddenSize):
    INPUT_DIM = descDim

    model = models.Sequential()
    model.add(layers.Dense(hiddenSize, activation='relu', input_dim=INPUT_DIM))
    model.add(layers.Dense(1, activation='linear', activity_regularizer=regularizers.l2(0.0001)))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model

params = [100]
datasetPortion = [1, 0.8, 0.6, 0.5, 0.3, 0.1, 0.05, 10]

componentResults = []
xvalResults=[]
portionResults = []

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='accuracy', min_delta=0.001, patience=10, verbose=1)

done = []

for molNdx in range(0, len(molfiles)):
    molName = molfiles[molNdx][1]  # [molfiles[molNdx].rfind("/", 0, -1)+1:-1]
    if molName in done:
        continue
            
    print("Processing "+molName)

    for portion in datasetPortion:
        try:
            print("Portion "+str(portion))

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

            folds = 5
            componentResults = []

            (n_fold_ds, n_fold_paths) = cu.loadDescriptors(molfiles[molNdx][0], portion * 0.8, dtype=descType,
                                                           active_decoy_ratio=-1,
                                                           selection_policy="RANDOM", return_type="SEPARATE",
                                                           exclusion_list=test_paths)

            (folds_list, excl_list) = cu.split(n_fold_ds, folds, policy="RANDOM")
            componentResults = []

            for param in params:
                foldResults = []
                print("Param "+str(param))
                for fold in range(0, folds):
                    print("Fold "+str(fold))
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

                    ann = getKerasNNModel(numcols, param)
                    ann.fit(train_ds.iloc[:, 0:numcols], ((train_ds["active"])).astype(int) * 100,
                            batch_size=500000,
                            epochs=1000, callbacks=[early_stopping])

                    results = pd.DataFrame()


                    results["score"] = [max(ann.predict(x[0].iloc[:, 0:numcols]).ravel()) for x in val_ds]
                    results["truth"] = [x[2] for x in val_ds]
                    auc = eval.plotSimROC(np.array(results["truth"]), np.array([results["score"]]), "", None)
                    auc_rank = eval.plotRankROC(np.array(results["truth"]), np.array([results["score"]]), "", None)
                    mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["score"]]))
                    foldResults.append((auc, auc_rank, mean_ef))

                print("X-Validation results: ")
                print(foldResults)

                if len(foldResults) > 0:
                    mean_auc_sim = np.mean([x[0] for x in foldResults])
                    std_auc_sim = np.std(np.mean([x[0] for x in foldResults]))
                    mean_auc_rank = np.mean([x[1] for x in foldResults])
                    std_auc_rank = np.std(np.mean([x[1] for x in foldResults]))
                    mean_mean_ef_1pc = np.mean([x[2][0.01] for x in foldResults])
                    std_mean_ef_1pc = np.std([x[2][0.01] for x in foldResults])
                    mean_mean_ef_5pc = np.mean([x[2][0.05] for x in foldResults])

                print("mean AUC(Sim)=" + str(mean_auc_sim) +
                      ", std=" + str(std_auc_sim) +
                      ", mean AUC(Rank)=" + str(mean_auc_rank) +
                      ", std=" + str(std_auc_rank) +
                      ", mean EF(1%)=" + str(mean_mean_ef_1pc) +
                      ", std=" + str(std_mean_ef_1pc) +
                      ", mean EF(5%)=" + str(mean_mean_ef_5pc) +
                      ", std=" + str(std_mean_ef_5pc))

                componentResults.append((molName, portion, param, mean_auc_sim, std_auc_sim, mean_auc_rank, std_auc_rank, mean_mean_ef_1pc,
                                         std_mean_ef_1pc, mean_mean_ef_5pc, std_mean_ef_5pc))
            else:
                print("X-Validation returned no results. Skipping training...")
                componentResults.append((molName, portion, param, 0, 0, 0, 0, 0, 0, 0, 0))

            xvalResults.extend(componentResults)

            # Find best score
            ef_rank = [x[5] for x in componentResults]

            best_estimators = params[np.argmax(ef_rank)]
            print("Best-score estimators no.: " + str(best_estimators))

            train_ds = cu.lumpRecords(n_fold_ds)
            ann = getKerasNNModel(numcols, best_estimators)
            
            t0 = time.time()

            ann.fit(train_ds.iloc[:, 0:numcols], ((train_ds["active"])).astype(int) * 100,
                    batch_size=500000,
                    epochs=1000, callbacks=[early_stopping])

            results = pd.DataFrame()

            results["score"] = [max(ann.predict(x[0].iloc[:, 0:numcols]).ravel()) for x in test_ds]
            # results["a_score"] = [G_a.score(x[0].iloc[:, 0:numcols]) for x in test_ds]
            results["truth"] = [x[2] for x in test_ds]  # np.array(test_ds)[:, 2]
        
            auc = eval.plotSimROC(results["truth"], [results["score"]], molName + "[ANN, " + str(portion * 100) + "%]",
                              molName + "_ANN_k_" + str(portion * 100) + "_sim.pdf")
            auc_rank = eval.plotRankROC(results["truth"], [results["score"]], molName + "[ANN, " + str(portion * 100) + "%]",
                              molName + "_ANN_k_" + str(portion * 100) + "_rank.pdf")
            mean_ef = eval.getMeanEFs(np.array(results["truth"]), np.array([results["score"]]))
            t1 = time.time();

            # print("Final results, num components = ", str(components)+": ")
            print("AUC(Sim)=" + str(auc))
            print("EF: ", mean_ef)

            portionResults.append((molName, portion, auc, auc_rank, mean_ef, t1-t0))
            print("Time taken = " + str(t1 - t0))

            f1 = open('results_keras_100.txt', 'w')
            print(xvalResults, file=f1)
            print(portionResults, file=f1)
            f1.close()

        except:
            fe = open("Errors.txt", "w")
            print("Error for "+molName+", potion="+str(portion))

        # full_train_ds = test_ds
        # full_train_ds.extend(n_fold_ds)

        # full_train_dss = [x[0] for x in test_ds]
        # full_train_dss.append([x[0] for x in n_fold_ds])
        # full_train_ds = cu.joinDataframes(full_train_dss)
        # ann = getKerasNNModel(numcols)
        # ann.fit(full_train_ds.iloc[:, 0:numcols], ((full_train_ds["active"])).astype(int) * 100, batch_size=500000, epochs=1000, callbacks=[early_stopping])
        #
        # # serialize model to JSON
        # model_json = ann.to_json()
        # with open(molName + "_AMM.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # ann.save_weights(molName + "_AMM.h5")
        # print("Saved model for "+molName+" to disk")


