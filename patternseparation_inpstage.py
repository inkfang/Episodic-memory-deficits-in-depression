# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:48:47 2016

@author: fangjim9
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

    
def changeInput(seq,inpno):   #adding noise to the input sequence.
    newSeq = copy(seq) 
    noind = binomial(1,inpno,size(seq[0]))
    ind = nonzero(noind)
    newSeq[:,ind] =  abs(seq[:,ind]-1)  

    return newSeq

def inpno(nstorp,networkb,nois,reflen,ran,s,inpno):    

    storinp_L = zeros((nstorp,re,nbin**2))
    storpaL = zeros((nstorp,re,di))

    refinp_L = inp.Lrw(reflen,omega2,bo,bar_l2,bar_w)

    inpref =  changeInput(refinp_L,inpno)    
    refl = lowdi(inpref,reflen,nbin,bo)
    del refinp_L

    ref_aL = networkb(refl)[:,:4]
    pca_aL,D_aL = whiten(ref_aL)
    
    ns = int(nstorp*ran)
    for i in range(nstorp):

        sL = inp.Lrw(re,omega2,bo,bar_l2,bar_w)   
        nsL =  changeInput(sL,inpno)          
        storinp_L[i] = lowdi(nsL,re,nbin,bo)        

        saL= networkb(storinp_L[i])[:,:4]
        storpaL[i] = inner(pca_aL(saL),D_aL)

    del storinp_L
        

    dataL = [storpaL[r] for r in range(nstorp)]

    allsto_aL = segpool(dataL,ls,nstorp)

    del  dataL
    reerrL = zeros((s,re))
    dd = range(ns,nstorp)
    shuffle(dd)
    dd = dd[:s]


    for j in range(s):
        reerrL[j] = distancewithoutk(re,storpaL[dd[j]],allsto_aL,icue,ls,di,nois)
    return reerrL



re = 50                                                # retrival length
icue = 0                                               # index of the cue
ls = 2                                                   # length segment
nosd  = 0.1                                        # std of the adding noise
di = 4


bo = 300
step = 45
bar_l1 = 24.5 # for shape H
#bar_l1 = 3.5
bar_l2 = 44.5# 4.5
bar_wh = 6.5
bar_v = 3.5
#bar_v = 6.5
bar_w = 14.5#6.5
no = 0
v = 1
sudturnp = 0
noro = 0


p = .02*pi
q = .06
omega2 =.025*e 

trainstep = 10000 
reflen = 3000
nois = [.05,.1,.2,.5] 

nbin = 30
nstorp = 200
s =40  
ran = 0.2


outni = ['_n005_','_n01_','_n02_','_n05_']
noinp = [0,.01,.03,.05,.1]
outan = ['i0','i001','i003','i005','i01']

for n in range(len(nois)):
    for t in range(len(noinp)):

        btraindata = inp.Lmore(trainstep,p,q,omega2,bo,bar_l2,bar_w,no)
        trainb = lowdi(btraindata,trainstep,nbin,bo)
        del btraindata

        networkb = hierachynet(30,15,3)
        networkb(trainb)

        reerrL= inpno(nstorp,networkb,nois[n],reflen,ran,s,noinp[t])

        errL = mean(reerrL,0)

        np.save("noinp_inp"+outni[n]+outan[t]+"_%d.npy"%(myid),
              errL)

pypar.finalize()
