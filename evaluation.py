import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

def getEnrichmentFactor(threshold, ds, sort_by="prob", truth="truth"):
    sorted_ds = ds.sort_values(by=sort_by, ascending=False)
    # print(sorted_ds)

    top_thresh = sorted_ds.iloc[0:int(threshold * len(sorted_ds))]
    # print(top_thresh)

    num_actives = sum(top_thresh[truth])
    # print("Number of actives found in top ", (threshold * 100), "%=", num_actives)
    expected = (float(sum(sorted_ds[truth])) / len(sorted_ds)) * len(top_thresh)
    # print("Number of actives expected =", expected)

    if expected > 0:
        ef = float(num_actives) / expected
    else:
        ef = 0
    # ef = float(num_actives) / len(top_thresh) / (float(sum(sorted_ds[truth])) / len(sorted_ds))

    return ef


# print(sim_ds[0][0])
def plotROCCurve(truth, preds, label, fileName):
    fpr, tpr, _ = roc_curve(truth.astype(int), preds)

    roc_auc = auc(fpr, tpr)

    if fileName is not None:
        fig = plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC (' + label + ")")
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(fileName)

        plt.close()
    return roc_auc

def plotSimROC(truth, results, title, fileName):
    #labels = [l[2] for l in mol_ds]

    simresults = np.concatenate([np.array(list(zip(r, truth))) for r in results])
    # print(simresults.shape)
    auc = plotROCCurve(simresults[:, 1], simresults[:, 0], title, fileName)

    thresholds = [0.01, 0.05]
    mean_efs={}
    for threshold in thresholds:
        ef = [getEnrichmentFactor(threshold, pd.DataFrame(data=list(zip(r, truth)), columns=("sim", "truth")),
                                  sort_by="sim",
                                  truth="truth") for r in results]

        #        print(ef)
        ef_mean = np.mean(ef)
        mean_efs[threshold] = ef_mean
        #        ef = getEnrichmentFactor(threshold, pd.DataFrame(data=simresults, columns=("sim", "truth")), sort_by="sim", truth="truth")
        # sim_pd = pd.DataFrame(data=simresults, columns=("sim", "truth"))
        # print(getEnrichmentFactor(0.01, sim_pd, sort_by="sim", truth="truth"))
        print("Mean EF@" + str((threshold * 100)) + "%=", ef_mean)

    return (auc, mean_efs)