#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:01:36 2018

@author: fangjim9
"""

### functions ###

def hierachynet(bo,recf,ovl):  
    switchboard = mdp.hinet.Rectangular2dSwitchboard(in_channels_xy = (bo,bo),
                                                 field_channels_xy=(recf,recf),
                                                 field_spacing_xy =(ovl,ovl))

    sfa_dim = 48
    sfa_lower_out= 32 #64

    sfanode = mdp.nodes.SFANode(input_dim = switchboard.out_channel_dim, output_dim = sfa_dim)
    sfa2node = mdp.nodes.QuadraticExpansionNode(input_dim=sfa_dim)
    sfanode2 = mdp.nodes.SFANode(input_dim = sfa2node.output_dim,output_dim = sfa_lower_out)

    flownode = mdp.hinet.FlowNode(mdp.Flow([sfanode,sfa2node,sfanode2]))

    sfalayer = mdp.hinet.CloneLayer(flownode, n_nodes = switchboard.output_channels)


    sfa_upper_bo = 6
    sfa_upper_recf = 4
    sfa_upper_out = 32 
    sfa_top_out = 10
    ovl2 = 2 

    switchboard2 =  mdp.hinet.Rectangular2dSwitchboard(in_channels_xy = (sfa_upper_bo,sfa_upper_bo),
                                                 field_channels_xy=(sfa_upper_recf,sfa_upper_recf),
                                                 field_spacing_xy = (ovl2,ovl2), 
                                                 in_channel_dim = sfa_lower_out)                        
#                                                
    sfa_uppernode = mdp.nodes.SFANode(input_dim = switchboard2.out_channel_dim, output_dim = sfa_dim)
    sfa_upperexp = mdp.nodes.QuadraticExpansionNode(input_dim = sfa_dim)


    sfa_uppernode2 = mdp.nodes.SFANode(input_dim = sfa_upperexp.output_dim, output_dim = sfa_upper_out)
    upper_flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_uppernode,sfa_upperexp,sfa_uppernode2]))
    upper_sfalayer = mdp.hinet.CloneLayer(upper_flownode, n_nodes = switchboard2.output_channels)


    sfa_top_node = mdp.nodes.SFANode(input_dim = upper_sfalayer.output_dim, output_dim = sfa_dim)
    sfa_topexp =mdp.nodes.QuadraticExpansionNode(input_dim = sfa_dim)


    sfa_topnode2 =mdp.nodes.SFANode(input_dim = sfa_topexp.output_dim,output_dim = sfa_top_out)
    sfa_over_node = mdp.hinet.FlowNode(mdp.Flow([sfa_top_node,sfa_topexp,sfa_topnode2]))
    
    network = mdp.Flow([switchboard,sfalayer,switchboard2,upper_sfalayer,sfa_over_node])

    return network                                                 

    
def whiten(refseq):
    pcanode = mdp.nodes.PCANode()
    pcanode(refseq)    
    D = diag(pcanode.d**-.5)

    return pcanode,D

# scale down the input image
def lowdi(highdi,lendata,nbin,bo):
    lowout  = zeros((lendata,nbin*nbin))
    for t in range(lendata):
        pattern = reshape(highdi[t],(bo,bo))
        indx = nonzero(pattern)[0]
        indy = nonzero(pattern)[1]
        H,tlx,tly = histogram2d(indx,indy,nbin,[[0,bo],[0,bo]],normed= False)
        lowout[t] = reshape(H,(nbin*nbin))
    return lowout


def noseqAN(stoep,lseq,lseg,q,di,sd):

    reseq = zeros((lseq,di))
    p = q+normal(0,(sd+1e-20),di)


    for j in range(lseq-1):

        p=q+normal(0,(sd+1e-20),di)

        disma = sqrt(sum((stoep[:,:di]-p)**2,1))
        mind = where(disma == min(disma))[0]
        shuffle(mind)
        ind = mind[0]

        reseq[j] = stoep[ind,:di]
        q = stoep[ind,lseg*di-di::]

    reseq[lseq-1] = stoep[ind,lseg*di-di::]
    return reseq



def distancewithan(re,slow,stoep,icue,lseg,di,nosd,orislow):
    #l = zeros(re)
    reseq = noseqAN(stoep,re,lseg,slow[icue],di,nosd)
    l = sqrt(sum((slow-reseq)**2,1))
    lori = sqrt(sum((orislow-reseq[:,:4])**2,1))   # retrieval error which only considers sfa features except AN components
    
    return l,lori

    
def distancewithoutk(re,slow,stoep,icue,lseg,di,nosd):
    reseq = noseqAN(stoep,re,lseg,slow[icue],di,nosd)
    l = sqrt(sum((slow-reseq)**2,1))
    
    return l

### jump in seq ###
def noseqnew(stoep,lseq,lseg,q,di,sd):

    reseq = zeros((lseq,di))

    #qq = (q+normal(0,(sd+1e-20),di))*ones([len(stoep),len(q)])
    p = q+normal(0,(sd+1e-20),di)
    reind = zeros(lseq-1)

    for j in range(lseq-1):

        p=q+normal(0,(sd+1e-20),di)

        disma = sqrt(sum((stoep[:,:di]-p)**2,1))
        mind = where(disma == min(disma))[0]
        shuffle(mind)
        ind = mind[0]

        reseq[j] = stoep[ind,:di]
        q = stoep[ind,lseg*di-di::]
        reind[j] =  ind
    reseq[lseq-1] = stoep[ind,lseg*di-di::]
    return reseq,reind
    
### jump am seq ##
def jumpamseq(re,slow,stoep,icue,lseg,di,nosd):
      reseq,reind = noseqnew(stoep,re,lseg,slow[0],di,nosd)
      errori = zeros(2)
      for e in range(1,re-1):
          if (int(reind[e-1]/49)!= int(reind[e]/49)):
               errori[0] += 1   
          else:
               errori[1] += 1    
      return errori

def jumpinseq(re,slow,stoep,icue,lseg,di,nosd):
      reseq,reind = noseqnew(stoep,re,lseg,slow[0],di,nosd)
      errori = zeros(2)
      for e in range(1,re-1):
          if (int(reind[e-1]/49)== int(reind[e]/49)) and ((reind[e]-1)!=reind[e-1]):
               errori[0] += 1    # jump within seq
          else:
               errori[1] += 1    # no jump or jump out of seq
      return errori

# generate pieces #
def genpiece(slow,lseg):
    seg = mdp.nodes.TimeFramesNode(lseg)
    stoep = seg(slow)
    
    return stoep

def segpool(pooldata,ls,nst):
    for i in range(nst):                                   
        segslow = genpiece(pooldata[i],ls)
        if i== 0:
            allep = segslow
        else:
            allep = concatenate((allep,segslow),0)
    return allep
