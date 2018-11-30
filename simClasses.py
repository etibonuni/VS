from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit import rdBase
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
import pandas as pd
from numpy.linalg import inv, det
from scipy.stats import norm
from sklearn.metrics import f1_score
import math
import pickle
import conformer_utils as cu
from glob import glob
from random import shuffle
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def getExpectedLikelihood(mu1, sigma1, mu2, sigma2):
    mu_1 = np.matrix(mu1)
    mu_2 = np.matrix(mu2)
    #sigma_1 = np.diag(np.matrix(sigma1).A1)
    sigma_1 = np.matrix(sigma1)
    sigma_1_inv = inv(sigma_1)
    sigma_1_det = det(sigma_1)
    #sigma_2 = np.diag(np.matrix(sigma2).A1)
    sigma_2 = np.matrix(sigma2)
    sigma_2_inv = inv(sigma_2)
    sigma_2_det = det(sigma_2)
    sigma_cross = inv(sigma_1_inv + sigma_2_inv)
    sigma_cross_det = det(sigma_cross)
    #mu_cross = np.dot(sigma_1_inv, mu_1) + np.dot(sigma_2_inv, mu_2)
    mu_cross = mu_1*sigma_1_inv + mu_2*sigma_2_inv
    #weight1 = weight1s[i]
    #weight2 = weight2s[j]

    a = ((2*math.pi) ** -6)  
    a = a* (sigma_cross_det ** 0.5) *  (sigma_1_det ** -0.5) * (sigma_2_det ** -0.5)
    #print a
    b = mu_1*sigma_1_inv*np.transpose(mu_1)
    #print("b=", b)
    b += mu_2*sigma_2_inv*np.transpose(mu_2)
    #print b
    b -= mu_cross*sigma_cross*np.transpose(mu_cross)
    
    #print (a, b)
    #print b
    b = math.exp(-0.5 * b)
    #print b
    #b = math.exp(-0.5*(np.transpose(mu_1)*sigma_1_inv*mu_1 + np.transpose(mu_2)*sigma_2_inv*mu_2 - np.transpose(mu_cross)*sigma_cross*mu_cross))
    #print (weight1 * weight2 * a * b)
    #print "----"
    #ret +=  (weight1 * weight2 * a * b)
    return  (a * b)
  
def getExpectedLikelihood_bhatt(mu1, sigma1, mu2, sigma2):
    mu_1 = np.matrix(mu1)
    mu_2 = np.matrix(mu2)
    #sigma_1 = np.diag(np.matrix(sigma1).A1)
    sigma_1 = np.matrix(sigma1)
    sigma_1_inv = inv(sigma_1)
    sigma_1_det = det(sigma_1)
    #sigma_2 = np.diag(np.matrix(sigma2).A1)
    sigma_2 = np.matrix(sigma2)
    sigma_2_inv = inv(sigma_2)
    sigma_2_det = det(sigma_2)
    sigma_cross = inv(0.5*sigma_1_inv + 0.5*sigma_2_inv)
    sigma_cross_det = det(sigma_cross)
    #mu_cross = np.dot(sigma_1_inv, mu_1) + np.dot(sigma_2_inv, mu_2)
    mu_cross = 0.5*mu_1*sigma_1_inv + 0.5*mu_2*sigma_2_inv
    
    a = (sigma_cross_det ** 0.5) * (sigma_1_det ** -0.25) * (sigma_2_det ** -0.25)
    
    #print(mu1)
    #print(np.transpose(mu_1))
    #print(sigma_1_inv)
    b = (-(0.25*(mu_1 * sigma_1_inv * np.transpose(mu_1))))
    #print (b)
    b -= (0.25* (mu_2 * sigma_2_inv*np.transpose(mu_2))) [0][0]
    #print (b)
    c = 0.5*(mu_cross*sigma_cross*np.transpose(mu_cross))[0][0]
    #print("b,c=",b,c)
    #print(0.5*(mu_cross*sigma_cross*np.transpose(mu_cross)))
    b += c
    
    #b = math.exp(b[0])
    #print(a, b)
    
    return  (a * math.exp(b))

def getExpectedLikelihood_bhatt2(mu1, sigma1, mu2, sigma2):
    mu_1 = np.matrix(mu1)
    mu_2 = np.matrix(mu2)
    #sigma_1 = np.diag(np.matrix(sigma1).A1)
    sigma_1 = np.matrix(sigma1)
    sigma_1_inv = inv(sigma_1)
    sigma_1_det = det(sigma_1)
    #sigma_2 = np.diag(np.matrix(sigma2).A1)
    sigma_2 = np.matrix(sigma2)
    sigma_2_inv = inv(sigma_2)
    sigma_2_det = det(sigma_2)
    #sigma_cross = inv(0.5*sigma_1_inv + 0.5*sigma_2_inv)
    #sigma_cross_det = det(sigma_cross)
    #mu_cross = 0.5*mu_1*sigma_1_inv + 0.5*mu_2*sigma_2_inv
    
    a = mu1-mu2
    a = a*inv((sigma_1+sigma_2)/2)
    a = a*np.transpose(np.matrix(mu1-mu2))
    a = float(a)/8
    
    b = math.log( det((sigma_1+sigma_2)/2) / math.sqrt(sigma_1_det * sigma_2_det) ) * 0.5
    

    return  (a + math.exp(b))
    
    
def getGMMSimilarity(gmm1, gmm2, simfunction):
    mu1s = gmm1.means_
    mu2s = gmm2.means_
    sigma1s = gmm1.covariances_
    sigma2s = gmm2.covariances_
    weight1s = gmm1.weights_
    weight2s = gmm2.weights_
    
    ret = 0

    ret = np.sum([weight1s[i]*weight2s[j]*simfunction(mu1s[i], sigma1s[i], mu2s[j], sigma2s[j]) for j in range(0, len(weight2s)) for i in range(0,len(weight1s))])
#    for i in range(0, len(weight1s)):
#            #j=i
#        for j in range(0, len(weight2s)):
#            l = getExpectedLikelihood(mu1s[i], sigma1s[i], mu2s[j], sigma2s[j])
#            #print("l=",l)
#            #l = l/math.sqrt(getExpectedLikelihood(mu1s[i], sigma1s[i], mu1s[i], sigma1s[i]) * 
#            #               getExpectedLikelihood(mu2s[j], sigma2s[j], mu2s[j], sigma2s[j]))
#            
#            #print("l=",l)
#            
#            #w = 1-(abs(weight1s[i]-weight2s[j])/(weight1s[i]+weight2s[j])) 
#            w = weight1s[i] * weight2s[j]
#            #print("w=", w)
#            #print("w*l=", w*l)
#            ret += w * l
#            #print("ret=", ret)
            
    return ret#/(len(weight1s)*len(weight2s))

#def getC2Similarity(gmm1, gmm2):
#    mu1s = gmm1.means_
#    mu2s = gmm2.means_
#    sigma1s = gmm1.covariances_
#    sigma2s = gmm2.covariances_
#    weight1s = gmm1.weights_
#    weight2s = gmm2.weights_
#    
#    ret = 0
#    a=0
#    b=0
#    c=0
#    
#    for i in range(0, len(weight1s)):
#        for j in range(0, len(weight2s)):
#            mu_1 = np.matrix(mu1s[i])
#            mu_2 = np.matrix(mu2s[j])
#            #sigma_1 = np.diag(np.matrix(sigma1).A1)
#            sigma_1 = np.matrix(sigma1s[i])
#            sigma_1_inv = inv(sigma_1)
#            sigma_1_det = det(sigma_1)
#            #sigma_2 = np.diag(np.matrix(sigma2).A1)
#            sigma_2 = np.matrix(sigma2s[j])
#            sigma_2_inv = inv(sigma_2)
#            sigma_2_det = det(sigma_2)
#                
#            v = inv(inv(sigma_1)+inv(sigma_2))
#            v_det = det(v)
#            k = mu_1 * sigma_1_inv * np.transpose(np.matrix(mu_1-mu_2)) +
#                mu_2 * sigma_2_inv * np.transpose(np.matrix(mu_2-mu_1))
#            
#            a += math.sqrt(v_det / (exp(k)*sigma_1_det*sigma_2_det))
#            b += weight1s[i]*weight2[j]*math.sqrt(v_det/(exp(k)*))
#    return ret#/(len(weight1s)*len(weight2s))

#def getNMP(gmm1, gmm2):
#    mu1s = gmm1.means_
#    mu2s = gmm2.means_
#    sigma1s = gmm1.covariances_
#    sigma2s = gmm2.covariances_
#    weight1s = gmm1.weights_
#    weight2s = gmm2.weights_
#    print(len(weight1s))
#    print(len(weight2s))
#    print(sigma1s)
#    rets = 0.0
#    for i in range(0, len(weight1s)):
#        for j in range(0, len(weight2s)):
#            d = 1.0
#            #print("s1s=", sigma1s[i])
#            for l in range(0, len(sigma1s[i])):
#                sigma1_sqr = sigma1s[i][l]**2
#                sigma2_sqr = sigma2s[j][l]**2
#                #print("s1_sqr=", sigma1_sqr)
#                sigma_npm = math.sqrt((sigma1_sqr/weight1s[i])+(sigma2_sqr/weight2s[j]))
#                #print("s=",sigma_npm)
#                #print("mu1_i_l=", mu1s[i][l])
#                #print("mu2_j_l=", mu2s[j][l])
#                d = d*norm.pdf(mu1s[i][l], loc=mu2s[j][l], scale=sigma_npm)
#                #print("d=", d)
#            print("prod=", d)
#            rets += d
#    print("sum=", rets)
# 
#    return rets

def NMPSim(x, y, w1, w2, mu1, mu2):
    sigma1_sqr = x ** 2
    sigma2_sqr = y ** 2
    sigma_npm = ((sigma1_sqr/w1)+(sigma2_sqr/w2))**0.5
    
    p = np.prod(norm.pdf(mu1, loc=mu2, scale=sigma_npm))
    #print("prod=", p)
    return p
    #return np.prod(list(map(lambda l: norm.pdf(mu1[l], loc=mu2[l], scale=sigma_npm[l]), range(0, len(x)))))
    
def getNMP2(gmm1, gmm2):
    mu1s = gmm1.means_
    mu2s = gmm2.means_
    sigma1s = gmm1.covariances_
    sigma2s = gmm2.covariances_
    weight1s = gmm1.weights_
    weight2s = gmm2.weights_

    s= np.sum(list(map(lambda x: np.sum(list(map(lambda y: NMPSim(x[0], y[0], x[1], y[1], x[2], y[2]),zip(sigma2s, weight2s, mu2s)))),zip(sigma1s, weight1s, mu1s))))
    #print("sum=", s)
    return s
    
def getNMPSimilarity(gmm1, gmm2):
    #print(getNMP2(gmm1, gmm1))
    #print(getNMP2(gmm2, gmm2))
    #print(getNMP2(gmm1, gmm2))
    r = getNMP2(gmm1, gmm1)+getNMP2(gmm2, gmm2) - 2*getNMP2(gmm1, gmm2)
    #print (r)
    return r**0.5



class MoleculeSimilarity:
    def __init__(self, conformers, paths):
        self.conformers = conformers
        self.paths = paths
        self.numcols = self.conformers[0][0].shape[1]-2
        
        return
        
    def getSim(self, mol1, mol2):
        return 0
   
    def getConformers(self):
        return self.conformers

    def doSim(self, templateNdx, simObj_bc):
        resultsMol = [simObj_bc.value.getSim(templateNdx, molNdx) for molNdx in range(0, len(simObj_bc.value.conformers))]
        return resultsMol

    def runSparkScreening(self, sc):
        activeRange = sc.range(0, sum([self.conformers[x][2] for x in range(0, len(self.conformers))]))
        # print(activeRange.collect())
        # activeRange=sc.range(0,3,numSlices=2)

        simObj_bc = sc.broadcast(self);

        return activeRange.map(lambda x: self.doSim(x, simObj_bc)).collect()
    
def manhattanDist(v1, v2):
    return np.sum(abs(np.array(v1)-np.array(v2)), axis=1)

def manhattanSim(v1, v2):
    #print("v1=", v1)
    #print("v2=", v2)
    dim = v1.shape[1]

    a = np.repeat(v1, v2.shape[0], axis=0)
    #b = np.repeat(v2, v1.shape[0], axis=0)
    b = np.tile(v2, (v1.shape[0], 1))
#    if len(dim)==1:
#        v1 = np.reshape(np.array(v1), (1,len(v1)))
#        v2 = np.reshape(np.array(v2), (1,len(v2)))
        
    return 1.0/(1+manhattanDist(a, b)/dim)
    
class USRMoleculeSim(MoleculeSimilarity):

    def getSim(self, mol1, mol2):
        
        mol1_confs = self.conformers[mol1][0][:,0:self.numcols]
        mol2_confs = self.conformers[mol2][0][:,0:self.numcols]
        #sims=[manhattanSim(c1, mol2_confs) for c1 in mol1_confs]

        sims = manhattanSim(mol1_confs, mol2_confs)

        return np.max(sims)

    def doSim(self, candidate, actives_bc):
        #resultsMol = [simObj_bc.value.getSim(templateNdx, molNdx) for molNdx in range(0, len(simObj_bc.value.conformers))]

        resultsMol = (candidate[0], [np.max(manhattanSim(candidate[1][:,0:self.numcols], actives_bc.value[i])) for i in range(0, len(actives_bc.value))])
        return resultsMol

    def runSparkScreening(self, sc):
        #activeRange = sc.range(0, sum([self.conformers[x][2] for x in range(0, len(self.conformers))]))

        actives = [np.array(self.conformers[i][0][:,0:self.numcols]) for i in range(0, len(self.conformers)) if self.conformers[i][2]==True]

        # print(activeRange.collect())
        # activeRange=sc.range(0,3,numSlices=2)

        actives_bc = sc.broadcast(actives)

        candidates = sc.parallelize([(i, np.array(self.conformers[i][0]) ) for i in range(0, len(self.conformers) )])

        candidates = candidates.repartition(len(actives))

        c = candidates.map(lambda x: self.doSim(x, actives_bc)).sortByKey(ascending=True).values().collect()
        return c

        #return activeRange.map(lambda x: self.doSim(x, simObj_bc)).collect()


class USR_MNPSim(MoleculeSimilarity):
    def __init__(self, conformers, paths):
        super(USR_MNPSim, self).__init__( conformers, paths)
        self.gmm_cache=dict()
        
    def getGMM(self, molNdx):
        if molNdx in self.gmm_cache:
            gmm = self.gmm_cache[molNdx]
        else:
            gmm = GaussianMixture(n_components=max(self.conformers[molNdx][1], 1)*4, covariance_type="diag").fit(self.conformers[molNdx][0].iloc[:,0:self.numcols])
            self.gmm_cache[molNdx] = gmm
        return gmm
    
    def getSim(self, mol1, mol2):
        print(mol1, ", ", mol2)
        gmm_1 = self.getGMM(mol1)
        gmm_2 = self.getGMM(mol2)

        return getNMPSimilarity(gmm_1, gmm_2)
    
class USR_EL1Sim(MoleculeSimilarity):
    def __init__(self, conformers, paths):
        super(USR_EL1Sim, self).__init__( conformers, paths)
        self.gmm_cache=dict()
        
    def getGMM(self, molNdx):
        if molNdx in self.gmm_cache:
            gmm = self.gmm_cache[molNdx]
        else:
            gmm = GaussianMixture(n_components=max(self.conformers[molNdx][1], 1)*4, covariance_type="full").fit(self.conformers[molNdx][0].iloc[:,0:self.numcols])
            self.gmm_cache[molNdx] = gmm
        return gmm
    
    def getSim(self, mol1, mol2):
        print(mol1, ", ", mol2)
        gmm_1 = self.getGMM(mol1)
        gmm_2 = self.getGMM(mol2)

        return getGMMSimilarity(gmm_1, gmm_2, getExpectedLikelihood)

class USR_Bhatt1Sim(MoleculeSimilarity):
    def __init__(self, conformers, paths):
        super(USR_Bhatt1Sim, self).__init__( conformers, paths)
        self.gmm_cache=dict()
        
    def getGMM(self, molNdx):
        if molNdx in self.gmm_cache:
            gmm = self.gmm_cache[molNdx]
        else:
            gmm = GaussianMixture(n_components=max(self.conformers[molNdx][1], 1)*4, covariance_type="full").fit(self.conformers[molNdx][0].iloc[:,0:self.numcols])
            self.gmm_cache[molNdx] = gmm
        return gmm
    
    def getSim(self, mol1, mol2):
        print(mol1, ", ", mol2)
        gmm_1 = self.getGMM(mol1)
        gmm_2 = self.getGMM(mol2)

        return getGMMSimilarity(gmm_1, gmm_2, getExpectedLikelihood_bhatt)
    
class USR_Bhatt2Sim(MoleculeSimilarity):
    def __init__(self, conformers, paths):
        super(USR_Bhatt2Sim, self).__init__( conformers, paths)
        self.gmm_cache=dict()
        
    def getGMM(self, molNdx):
        if molNdx in self.gmm_cache:
            gmm = self.gmm_cache[molNdx]
        else:
            gmm = GaussianMixture(n_components=max(self.conformers[molNdx][1], 1)*4, covariance_type="full").fit(self.conformers[molNdx][0].iloc[:,0:self.numcols])
            self.gmm_cache[molNdx] = gmm
        return gmm
    
    def getSim(self, mol1, mol2):
        print(mol1, ", ", mol2)
        gmm_1 = self.getGMM(mol1)
        gmm_2 = self.getGMM(mol2)

        return getGMMSimilarity(gmm_1, gmm_2, getExpectedLikelihood_bhatt2)


def doSim(templateNdx, simObj_bc):
    resultsMol = [simObj_bc.value.getSim(templateNdx, molNdx) for molNdx in range(0, len(simObj_bc.value.conformers))]
    return resultsMol

def runSparkScreening(sc, simObj):
    activeRange = sc.range(0, sum([simObj.conformers[x][2] for x in range(0, len(simObj.conformers))]))
    # print(activeRange.collect())
    # activeRange=sc.range(0,3,numSlices=2)

    simObj_bc = sc.broadcast(simObj);

    return activeRange.map(lambda x: doSim(x, simObj_bc)).collect()
