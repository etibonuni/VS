import pandas as pd
import numpy as np
from glob import glob
from random import shuffle

# Returns the number of active and decoy molecules present in the given folder
def getDescriptorStats(path):
    active_filenames = glob(path + "active_mol_*.usr")
    decoy_filenames = glob(path + "decoy_mol_*.usr")

    return {"num_actives": len(active_filenames), "num_decoys": len(decoy_filenames)}
    
# Load the USR descriptors in the given file into a Pandas dataframe, and add the "active" column set to the given boolean label
def loadDescriptorFile(path, label):
    # print(path)
    f = open(path)
    numRotatableBonds = int(f.readline())
    # print("numRot=", numRotatableBonds)

    descs = pd.read_csv(f, header=None, index_col=0)
    descs["active"] = [label] * len(descs.index)

    descs = descs.drop(labels=len(descs.columns) - 1, axis=1)
    # print(len(descs.columns))
    descs = descs.sort_values(by=len(descs.columns) - 1, ascending=True)

    return (descs, numRotatableBonds)

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



# Selectively load the USR descriptors in the given path according to the parameters
# num_actives: the number of actives to load. If <= 1 it is taken as a percentage of the total number of actives
# active_decoy_ratio: The number of decoys to load as a fraction of the number of actives. -1 to maintain the population ratio of decoys to actives
# selection_policy: "RANDOM" | "SEQUENTIAL". If SEQUENTIAL, molecules will be loaded in order starting at the first one
# return_type: "SEPARATE" | "LUMPED". If SEPARATE, a separate Pandas DataFrame with conformer description for each molecule will be returned in a list along with the path for each descriptor file. If LUMPED then all the conformer descriptors for all the loaded molecules will be returned in a single DataFrame.
def loadDescriptors(path, num_actives, active_decoy_ratio=1, dtype="usr", selection_policy="SEQUENTIAL",
                    return_type="SEPARATE", exclusion_list=None):
    active_filenames = glob(path + "active_mol_*." + dtype)
    print(len(active_filenames))

    decoy_filenames = glob(path + "decoy_mol_*." + dtype)

    if num_actives <= 1:
        num_actives = len(active_filenames) * num_actives

    if active_decoy_ratio < 0:
        active_decoy_ratio = float(len(decoy_filenames) / len(active_filenames))

    num_decoys = num_actives * active_decoy_ratio

    paths = []
    records = []
    if selection_policy == "SEQUENTIAL":
        numToLoad = min(num_actives, len(active_filenames))
        i = 0
        while len(records) < numToLoad:
            fpath = path + "active_mol_" + str(i) + "." + dtype
            if (exclusion_list is not None) and (fpath in exclusion_list):
                i += 1
                continue

            paths.append(fpath)

            mol = loadDescriptorFile(fpath, True)
            records.append((mol[0], mol[1], True))
            i += 1

        numToLoad = numToLoad + min(num_decoys, len(decoy_filenames))
        i = len(active_filenames)
        while len(records) < numToLoad:
            fpath = path + "decoy_mol_" + str(i) + "." + dtype
            if (exclusion_list is not None) and (fpath in exclusion_list):
                i += 1
                continue

            paths.append(fpath)

            mol = loadDescriptorFile(fpath, False)
            records.append((mol[0], mol[1], False))

            i += 1
    else:
        if selection_policy == "RANDOM":
            shuffle(active_filenames)
            shuffle(decoy_filenames)

            numToLoad = min(num_actives, len(active_filenames))
            print("numToLoad=", numToLoad)
            i = 0
            while (len(records) < numToLoad) and (i < len(active_filenames)):
                fpath = active_filenames[i]
                if (exclusion_list is not None) and (fpath in exclusion_list):
                    i += 1
                    continue

                paths.append(fpath)

                mol = loadDescriptorFile(fpath, True)
                records.append((mol[0], mol[1], True))

                i += 1

            numToLoad = numToLoad + min(num_decoys, len(decoy_filenames))
            print("numToLoad=", numToLoad)

            # i = len(active_filenames)
            i = 0
            while (len(records) < numToLoad) and (i < len(decoy_filenames)):
                fpath = decoy_filenames[i]
                if (exclusion_list is not None) and (fpath in exclusion_list):
                    i += 1
                    continue

                paths.append(fpath)

                mol = loadDescriptorFile(fpath, False)
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
                recs = recs.append(mol)

        return (recs, paths)
    else:
        return (records, paths)