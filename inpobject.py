# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:34:15 2015

@author: jing
"""
########################################
##        Input:  SHAPE L             ##
########### lissaU trajectory ##########
def Lmore(T,p,q,omiga,bo,bar_l,bar_w,no):   
    t = linspace(0,T,T)

    a = 100
    b = 100
    lx = a*sin(p*t + random()*2*pi)+150#+15
    ly = b*sin(q*t + random()*2*pi)+150 #15

    xx = floor(lx)
    yy = floor(ly)
    x = ones([bo,bo])*range(0,bo)
    y = x.T
    alpha0 = random()*2*pi
    alpha =1e-20+omiga*t+alpha0 
    #alpha =1e-20+omiga*t#+alpha0    
    patterns = zeros((T,size(x)))

    for i in range(T):
        a =  alpha[i]
        
        r = abs(cos(a)+1e-20)/(cos(a)+1e-20)
        q = abs(sin(a)+1e-20)/(sin(a)+1e-20)
        hdl = abs(tan(a)*x-y+ yy[i]-tan(a)*xx[i])/sqrt(tan(a)**2+1)-bar_l-1e-20
        hdw1 = q*((-1/tan(a))*x-y+yy[i]+xx[i]/tan(a))/sqrt(1/tan(a)**2+1)-(bar_l)-1e-20
        hdw2 = q*((-1/tan(a))*x-y+yy[i]+xx[i]/tan(a))/sqrt(1/tan(a)**2+1)-(bar_l-2*bar_w)-1e-20
    
    
        hdl = -(hdl/abs(hdl)-1)/2.
        hdw = hdw1*hdw2    
        hdw = -(hdw/abs(hdw)-1)/2.
        hinp = abs(hdl*hdw)
        
    
        vdl1 = r*(tan(a)*x-y+ yy[i]-tan(a)*xx[i])/sqrt(tan(a)**2+1)-(bar_l) -1e-20
        vdl2 =  r*(tan(a)*x-y+ yy[i]-tan(a)*xx[i])/sqrt(tan(a)**2+1)-(bar_l-2*bar_w) -1e-20
        vdw = abs((-1/tan(a))*x-y+yy[i]+xx[i]/tan(a))/sqrt(1/tan(a)**2+1)-bar_l-1e-20
    
        vdl = vdl1*vdl2
        vdl = -(vdl/abs(vdl)-1)/2.
        vdw = -(vdw/abs(vdw)-1)/2.
        vinp = abs(vdl*vdw)
    
        inp = ((hinp + vinp)>0)*1
        patterns[i] = reshape(inp,size(inp))  
    return patterns


######### random walk trajectory #############
def Lrw(T,omega,bo,bar_l,bar_w):   
    lx = randint(50,251)
    ly = randint(50,251)

    x = ones([bo,bo])*range(0,bo)
    y = x.T

    a = random()*2*pi
    omega = normal(0,0.035*e,T)
 
    patterns = zeros((T,size(x)))

    for i in range(T):
        a = a+omega[i]+1e-20 
        vx = abs(normal(5,1.3))  # mean velocity and sd
        vy = abs(normal(5,1.3))  # mean velocity and sd

        mdx = binomial(1,0.5)
        mdy = binomial(1,0.5)
        delta_x = vx*(1*mdx+(-1)*(1-mdx))
        delta_y = vy*(1*mdy+(-1)*(1-mdy))
        
        lx = rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*250+(1-mdx)* 50-lx),abs(delta_x))
              + min((abs(mdx*250+(1-mdx)* 50-lx)-abs(delta_x)),0)))
        ly = rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*250+(1-mdy)* 50-ly),abs(delta_y))
              + min((abs(mdy*250+(1-mdy)* 50-ly)-abs(delta_y)),0))) 
              
        r = abs(cos(a)+1e-20)/(cos(a)+1e-20)
        q = abs(sin(a)+1e-20)/(sin(a)+1e-20)
        hdl = abs(tan(a)*x-y+ ly-tan(a)*lx)/sqrt(tan(a)**2+1)-bar_l-1e-20
        hdw1 = q*((-1/tan(a))*x-y+ly+lx/tan(a))/sqrt(1/tan(a)**2+1)-(bar_l)-1e-20
        hdw2 = q*((-1/tan(a))*x-y+ly+lx/tan(a))/sqrt(1/tan(a)**2+1)-(bar_l-2*bar_w)-1e-20
    
    
        hdl = -(hdl/abs(hdl)-1)/2.
        hdw = hdw1*hdw2    
        hdw = -(hdw/abs(hdw)-1)/2.
        hinp = abs(hdl*hdw)
        
    
        vdl1 = r*(tan(a)*x-y+ ly-tan(a)*lx)/sqrt(tan(a)**2+1)-(bar_l) -1e-20
        vdl2 =  r*(tan(a)*x-y+ ly-tan(a)*lx)/sqrt(tan(a)**2+1)-(bar_l-2*bar_w) -1e-20
        vdw = abs((-1/tan(a))*x-y+ly+lx/tan(a))/sqrt(1/tan(a)**2+1)-bar_l-1e-20
    
        vdl = vdl1*vdl2
        vdl = -(vdl/abs(vdl)-1)/2.
        vdw = -(vdw/abs(vdw)-1)/2.
        vinp = abs(vdl*vdw)
    
        inp = ((hinp + vinp)>0)*1
        patterns[i] = reshape(inp,size(inp))  
    return patterns
    
