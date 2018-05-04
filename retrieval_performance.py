# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:14:27 2016

@author: jing
"""


import pypar
myid = pypar.rank()

from numpy import *
from pylab import *
import mdp
import scipy as sy
import inpobject as inp
import time
from funcall import *


def ANre(nstorp,networkb,nois,reflen, anL,ran,s,da):    

    storinp_L = zeros((nstorp,re,nbin**2))
    storpaL = zeros((nstorp,re,di))

    storpaaL = zeros((nstorp,re,di+da))

    refinp_L = inp.Lrw(reflen,omega2,bo,bar_l2,bar_w)

    refl = lowdi(refinp_L,reflen,nbin,bo)
    del refinp_L

    ref_aL = networkb(refl)[:,:4]
    pca_aL,D_aL = whiten(ref_aL)

    
    ns = int(nstorp*ran)
    for i in range(nstorp):

        sL = inp.Lrw(re,omega2,bo,bar_l2,bar_w)    #testing with randowm walk data
        storinp_L[i] = lowdi(sL,re,nbin,bo)        


        saL= networkb(storinp_L[i])[:,:4]

        storpaL[i] = inner(pca_aL(saL),D_aL)

    del storinp_L 

    storpnL = concatenate((storpaL, anL[:,newaxis,:]*ones((re,1))),2)
    
    xL = storpaL[:ns,0,newaxis]    
    anindL = argmin(sum((storpaL[ns::,0] - xL)**2,2),0)    

    Lan = anL[anindL]

    storpaaL[ns::] = concatenate((storpaL[ns::], Lan[:,newaxis,:]*ones((re,1))),2) 
    storpaaL[:ns] = concatenate((storpaL[:ns], anL[:ns,newaxis,:]*ones((re,1))),2)    


    dataaL = [storpaaL[r] for r in range(nstorp)]
    datanL = [storpnL[r] for r in range(nstorp)]
    

    allsto_aL = segpool(dataaL,ls,nstorp)
    allsto_nL= segpool(datanL,ls,nstorp)
    del  dataaL, datanL

    reerrL = zeros((3,s,re))
    orierrL = zeros((3,s,re))
    jins = zeros((3,s,2))
    jams = zeros((3,s,2))


    dd = range(ns,nstorp)
    shuffle(dd)
    dd = dd[:s]
    aa = range(s)
    nn = range(nstorp)
    shuffle(nn)
    nn = nn[:s]
    d = da + di

    for j in range(s):
#### retrieval error ###

        reerrL[0,j],orierrL[0,j] = distancewithan(re,storpaaL[dd[j]],allsto_aL,icue,ls,d,nois,storpaL[dd[j]])
        reerrL[1,j],orierrL[1,j] = distancewithan(re,storpaaL[aa[j]],allsto_aL,icue,ls,d,nois,storpaL[aa[j]])
        reerrL[2,j],orierrL[2,j] = distancewithan(re,storpnL[nn[j]],allsto_nL,icue,ls,d,nois,storpaL[nn[j]])
        
        jins[0,j] = jumpinseq(re,storpaaL[dd[j]],allsto_aL,icue,ls,d,nois)
        jins[1,j] = jumpinseq(re,storpaaL[aa[j]],allsto_aL,icue,ls,d,nois)
        jins[2,j] = jumpinseq(re,storpnL[nn[j]],allsto_nL,icue,ls,d,nois)

#### am seq jump ####
        jams[0,j] = jumpamseq(re,storpaaL[dd[j]],allsto_aL,icue,ls,d,nois)
        jams[1,j] = jumpamseq(re,storpaaL[aa[j]],allsto_aL,icue,ls,d,nois)
        jams[2,j] = jumpamseq(re,storpnL[nn[j]],allsto_nL,icue,ls,d,nois)

    return reerrL,orierrL,jins,jams


re = 50                                                # retrival length
icue = 0                                               # index of the cue
ls = 2                                                   # length segment
nosd  = 0.1                                        # std of the adding noise
di = 4


bo = 300
step = 45
bar_l1 = 24.5 

bar_l2 = 44.5
bar_wh = 6.5
bar_v = 3.5

bar_w = 14.5
no = 0



p = .02*pi
q = .06

omega2 =.025*e 

trainstep = 10000 
reflen = 3000
nois = [.05,.1,.2,.5] 


da = 2# dimention of AN {2,3,4}

nbin = 30
nstorp = 200 #total number of stored sequences
s = 20   # sample size
ran = 0.1 #original 10%# number of with neurogensis/number of all sequences

p_a = [.1,.3,.5,1,3]
outan = ['a01','a03','a05','a1','a3']
outni = ['_n005_','_n01_','_n02_','_n05_']


for n in range(len(nois)):
    for t in range(len(p_a)):


        btraindata = inp.Lmore(trainstep,p,q,omega2,bo,bar_l2,bar_w,no) #training with lisau trajectory
        trainb = lowdi(btraindata,trainstep,nbin,bo) 
        del btraindata

        networkb = hierachynet(30,15,3)
        networkb(trainb)

        anL = p_a[t]*normal(0,1,(nstorp,da))
        reerrL,orierrL,jins,jams= ANre(nstorp,networkb,nois[n],reflen,anL,ran,s,da)

        errL = mean(reerrL,1)
        oerrL = mean(orierrL,1)

        jin = mean(jins,1)
        jam = mean(jams,1)


        savez("AnL_di_rw"+outni[n]+outan[t]+"_%d.npz"%(myid),
              eL=errL,oL=oerrL,jin=jin,jam=jam)

pypar.finalize()
