import pandas as pd
import numpy as np
from glob import glob
from random import shuffle
import random


# Returns the number of active and decoy molecules present in the given folder
def getDescriptorStats(path):
    active_filenames = glob(path + "active_*.sdf")
    decoy_filenames = glob(path + "decoy_*.sdf")

    return {"num_actives": len(active_filenames), "num_decoys": len(decoy_filenames)}
    
# Load the USR descriptors in the given file into a Pandas dataframe, and add the "active" column set to the given boolean label
def loadDescriptorFile(path, label):
    try:
        # print(path)
        f = open(path)
        numRotatableBonds = int(f.readline())
        # print("numRot=", numRotatableBonds)
        descs = pd.read_csv(f, header=None, index_col=0).fillna(0)
        descs["active"] = [label] * len(descs.index)
        descs = descs.drop(labels=len(descs.columns) - 1, axis=1)
        # print(len(descs.columns))
        descs = descs.sort_values(by=len(descs.columns) - 1, ascending=True)

        return (descs, numRotatableBonds)
    except:
        return (None, 0)

def checkMissing(path, dtype):
    missing_actives=list()
    missing_decoys=list()

    active_filenames = glob(path + "/active_mol_*." + dtype)
    decoy_filenames = glob(path + "/decoy_mol_*." + dtype)

    #print(active_filenames)
    #print(decoy_filenames)

    max_active = 0
    i=0
    while i < len(active_filenames)+len(missing_actives)+1:
        #print("a:"+str(i)+" - "+str(len(active_filenames)+len(missing_actives)+1))
        if path+"/active_mol_"+str(i)+"."+dtype not in active_filenames:
            if i <= len(active_filenames)+1:
                missing_actives.append(i)
        elif max_active < i:
            max_active=i
        i+=1


    #print(max_active+len(decoy_filenames)+len(missing_decoys)+1)

    while i < max_active+len(decoy_filenames)+len(missing_decoys)+2:
        #print("d:" + str(i)+" - "+str(max_active+len(decoy_filenames)+len(missing_decoys)+2))
        if path+"/decoy_mol_"+str(i)+"."+dtype not in decoy_filenames:
            if i <= max_active+len(decoy_filenames)+1:
               missing_decoys.append(i)
        i+=1

    return (missing_actives, missing_decoys)

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

# Selectively load the USR descriptors in the given path according to the parameters
# num_actives: the number of actives to load. If <= 1 it is taken as a percentage of the total number of actives
# active_decoy_ratio: The number of decoys to load as a fraction of the number of actives. -1 to maintain the population ratio of decoys to actives
# selection_policy: "RANDOM" | "SEQUENTIAL". If SEQUENTIAL, molecules will be loaded in order starting at the first one
# return_type: "SEPARATE" | "LUMPED". If SEPARATE, a separate Pandas DataFrame with conformer description for each molecule will be returned in a list along with the path for each descriptor file. If LUMPED then all the conformer descriptors for all the loaded molecules will be returned in a single DataFrame.
def loadDescriptors(path, num_actives, active_decoy_ratio=1, dtype="usr", selection_policy="SEQUENTIAL",
                    return_type="SEPARATE", exclusion_list=None):
    active_filenames = glob(path + "active_*." + dtype)

    decoy_filenames = glob(path + "decoy_*." + dtype)

    if num_actives <= 1:
        num_actives = len(active_filenames) * num_actives

    if active_decoy_ratio < 0:
        active_decoy_ratio = float(len(decoy_filenames) / len(active_filenames))

    num_decoys = num_actives * active_decoy_ratio

    print("Loading "+str(num_actives)+" actives and "+str(num_decoys)+" decoys.")

    paths = []
    records = []
    if selection_policy == "SEQUENTIAL":
        numToLoad = min(num_actives, len(active_filenames))
        sortedNdx_actives = argsort(active_filenames)

        i = 0

        while (len(records) < numToLoad) and (i<len(active_filenames)):
            #fpath = path + "active_mol_" + str(i) + "." + dtype
            fpath = active_filenames[sortedNdx_actives[i]]
            if (exclusion_list is not None) and (fpath in exclusion_list):
                i += 1
                continue

            paths.append(fpath)

            mol = loadDescriptorFile(fpath, True)
            if mol[0] is not None:
                records.append((mol[0], mol[1], True))

            i += 1

        numToLoad = numToLoad + min(num_decoys, len(decoy_filenames))
        #i = len(active_filenames)
        sortedNdx_decoys = argsort(decoy_filenames)
        i=0
        while (len(records) < numToLoad) and (i<len(decoy_filenames)):
            #fpath = path + "decoy_mol_" + str(i) + "." + dtype
            fpath = decoy_filenames[sortedNdx_decoys[i]]
            if (exclusion_list is not None) and (fpath in exclusion_list):
                i += 1
                continue

            paths.append(fpath)

            mol = loadDescriptorFile(fpath, False)

            if mol[0] is not None:
                records.append((mol[0], mol[1], False))

            i += 1
    else:
        if selection_policy == "RANDOM":
            shuffle(active_filenames)
            shuffle(decoy_filenames)

            numToLoad = min(num_actives, len(active_filenames))

            i = 0
            while (len(records) < numToLoad) and (i < len(active_filenames)):
                fpath = active_filenames[i]
                if (exclusion_list is not None) and (fpath in exclusion_list):
                    i += 1
                    continue

                paths.append(fpath)

                mol = loadDescriptorFile(fpath, True)

                if mol[0] is not None:
                    records.append((mol[0], mol[1], True))

                i += 1

            numToLoad = numToLoad + min(num_decoys, len(decoy_filenames))

            # i = len(active_filenames)
            i = 0
            while (len(records) < numToLoad) and (i < len(decoy_filenames)):
                fpath = decoy_filenames[i]
                if (exclusion_list is not None) and (fpath in exclusion_list):
                    i += 1
                    continue

                paths.append(fpath)

                mol = loadDescriptorFile(fpath, False)

                if mol[0] is not None:
                    records.append((mol[0], mol[1], False))

                i += 1
        else:
            return None

    if return_type == "LUMPED":
        recs = None
        for [mol, rot, label] in records:
            if recs is None:
                recs = mol
            else:
                recs = recs.append(mol)#np.concatenate((recs, mol))

        return (recs, paths)
    else:
        return (records, paths)


def takePortion(records, num_to_take, selection_policy="SEQUENTIAL", return_type="SEPARATE", exclusion_list=None):

    if num_to_take <= 1:
        num_to_take = len(records) * num_to_take

    sel_range = set(range(0, len(records)))
    if (exclusion_list is not None) and (len(exclusion_list)>0):
        #sel_range = [x for x in sel_range if x not in exclusion_list]
        sel_range = sel_range-exclusion_list

    if selection_policy=="SEQUENTIAL":
        sel_range = [sel_range[x] for x in range(0, num_to_take)]
    elif selection_policy=="RANDOM":
        sel_range = set(np.random.choice(list(sel_range), size=int(num_to_take), replace=False))

    selection = [records[x] for x in sel_range]

    if return_type=="LUMPED":
        recs = None
        for [mol, rot, label] in records:
            if recs is None:
                recs = mol
            else:
                recs = recs.append(mol)#np.concatenate((recs, mol))

        return (recs, sel_range)

    return (selection, sel_range)

def split(records, folds, policy="RANDOM"):
    recs_per_fold = int(len(records)/folds)

    if policy=="RANDOM":
        excl = []
        recs = []
        excl2 = set()
        for fold in range(0, folds):
            (r, i) = takePortion(records, recs_per_fold, selection_policy=policy, return_type="SEPARATE", exclusion_list=excl2)
            recs.append(r)
            excl.append(i)
            excl2.update(i)

        return (recs, excl)
    elif policy=="SEQUENTIAL":
        startNdx=0
        recs=[]
        excl = []

        for fold in range(0, folds):
            recs.append(records[startNdx:startNdx+recs_per_fold])
            excl.append(range(startNdx, startNdx+recs_per_fold))

            startNdx = startNdx+recs_per_fold

        return (recs, excl)

def lumpRecords(records):
    recs = None
    for [mol, rot, label] in records:
        if recs is None:
            recs = mol
        else:
            recs = recs.append(mol)  # np.concatenate((recs, mol))

    return recs

def joinDataframes(record_list):
    recs = None

    for rec in record_list:
        if recs is None:
            recs = rec
        else:
            recs = recs.append(rec)

    return recs
