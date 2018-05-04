# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:10:07 2016

@author: jing
"""


#squence analysis depression#
import pypar
myid = pypar.rank()

from numpy import *
from pylab import *
import mdp
import scipy as sy
import inpobject as inp
from funcall import *

    
def ancompdist(nstorp,ns,ne,seq,seqall,indseq):
    indall = range(len(seqall))
    indall.remove(indseq[ns])
    rest_seq = seqall[indall]
    disat = sum((rest_seq[:,:48]- seq[ns,ne])**2,2)
    cloind = unravel_index(disat.argmin(),disat.shape)
    lentest = min(49 - cloind[1],10,49-ne)
    seqdist = sum((seq[ns,ne:(ne+lentest)]- rest_seq[cloind[0],cloind[1]:(cloind[1]+lentest)])**2,1)
    seqdist = seqdist - seqdist[0]  
    return lentest,seqdist


def Andiv(nstorp,networka,reflen,anL,ran,s,da):
 
    storinp_L = zeros((nstorp,re,nbin**2))
    storpaL = zeros((nstorp,re,di))

    storpaaL = zeros((nstorp,re,di+da))

    refinp_L = inp.Lmore(reflen,p,q,omega2,bo,bar_l2,bar_w,no)


    refl = lowdi(refinp_L,reflen,nbin,bo)
    del refinp_L

    ref_aL = networka(refl)[:,:4]
    pca_aL,D_aL = whiten(ref_aL)

    
    ns = int(nstorp*ran)
    for i in range(nstorp):
        sL = inp.Lmore(re,p,q,omega2,bo,bar_l2,bar_w,no)   
        storinp_L[i] = lowdi(sL,re,nbin,bo)        


        saL= networka(storinp_L[i])[:,:4]


        storpaL[i] = inner(pca_aL(saL),D_aL)

    del storinp_L 

    storpnL = concatenate((storpaL, anL[:,newaxis,:]*ones((re,1))),2)
    xL = storpaL[:ns,0,newaxis]    
    anindL = argmin(sum((storpaL[ns::,0] - xL)**2,2),0)    

    Lan = anL[anindL]
    storpaaL[ns::] = concatenate((storpaL[ns::], Lan[:,newaxis,:]*ones((re,1))),2) 
    storpaaL[:ns] = concatenate((storpaL[:ns], anL[:ns,newaxis,:]*ones((re,1))),2)    


    dd = range(ns,nstorp)
    shuffle(dd)
    dd = dd[:s]
    aa = range(s)
    nn = range(nstorp)
    shuffle(nn)
    nn = nn[:s]

    for j in range(s*5):

        ne = randint(0,(re-2),3) 
        nlen = zeros((3,10))
        ctl = zeros((3,10))
        c1,dT = ancompdist(s,j%s,ne[0],storpaaL[dd],storpaaL,dd)
        c2,aT = ancompdist(s,j%s,ne[1],storpaaL[aa],storpaaL,aa)
        c3,nT = ancompdist(s,j%s,ne[2],storpnL[nn],storpnL,nn)
   
        nlen[0,:c1] += ones(c1)
        nlen[1,:c2] += ones(c2)
        nlen[2,:c3] += ones(c3)

        ctl[0,:c1] += dT
        ctl[1,:c2] += aT
        ctl[2,:c3] += nT
        nlen+=1e-20
        ctl+=1e-20
    mdist = ctl/nlen

    return mdist

thr = 1
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
bar_w = 14.5#6.5
no = 0
noro = 0


p = .02*pi
q = .06

omega2 =.025*e 

Nr = 30

trainstep = 10000
reflen = 3000

nbin = 30
nstorp = 200
s = 20   #
ran = 0.1 # number of with neurogensis/number of all sequences
# adult neurogensis #
p_a = [.1,.3,.5,1,3]
outan = ['a01','a03','a05','a1','a3']

divseq = zeros((3,10))

da = 2
for a in range(len(p_a)):
    atraindata = inp.Lmore(trainstep,p,q,omega2,bo,bar_l2,bar_w,no)
    traina = lowdi(atraindata,trainstep,nbin,bo)
    
    del atraindata

    networka = hierachynet(30,15,3)
    networka(traina)
    anT = p_a[a]*normal(0,1,(nstorp,da))
    divseq= Andiv(nstorp,networka,reflen,anT,ran,s,da)

    np.save("AnL_d2_seqdiv_test_"+outan[a]+"_%d.npy"%(myid),divseq)

pypar.finalize()


