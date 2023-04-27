import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tabulate import tabulate
import h5py
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import scipy
from scipy import stats
from scipy.integrate import cumtrapz
from scipy.ndimage import gaussian_filter as gf
from matplotlib.lines import Line2D
import pandas
import corner
import sympy as sp
import os
from scipy.interpolate import RegularGridInterpolator
import sys
sys.path.insert(0, './COMPAS')
from spin_class import * 

def abs_to_app(abs_mag,dL):  #dL in parsec!
    
    app_mag=abs_mag + 5*np.log10( dL/10 ) 
    
    return app_mag

def app_to_abs(app_mag,dL): #dL in parsec!
    
    abs_mag=app_mag-5*np.log10( dL/10 )
    
    return abs_mag
    
def abs_mag(m,dl): # dl in Mpc
	dl *= 1.e6
	return m - 5. * np.log10(dl/10.)

def app_to_flux(app_mag):
    
    flux=10**(-(app_mag+48.6)/2.5)
    
    return flux    
    
def flux_to_app(flux):
    
    app_mag=-2.5*np.log10(flux)-48.6
    
    return app_mag
    
def GW_duty(dH,dL,dV,dK,num):
    ranH = np.random.uniform(0.,1.,num)
    ranL = np.random.uniform(0.,1.,num)
    ranV = np.random.uniform(0.,1.,num)
    ranK = np.random.uniform(0.,1.,num)

    Hbool = np.zeros(num)
    Lbool = np.zeros(num)
    Vbool = np.zeros(num)
    Kbool = np.zeros(num)

    Hbool[ranH<=dH] = True
    Lbool[ranL<=dL] = True
    Vbool[ranV<=dV] = True
    Kbool[ranK<=dK] = True
    
    det_bool=[Hbool,Lbool,Vbool,Kbool]

    return det_bool
    
def obtainM1BHandM2NS_spin(m1, m2, spin1, spin2):
    m1bh, m2ns = np.zeros_like(m1), np.zeros_like(m1)
    spinbh, spinns = np.zeros_like(m1), np.zeros_like(m1)
    maskm1heavier = (m1 >= m2)
    maskm2heavier = (m1 < m2)
    
    m1bh[maskm1heavier] = m1[maskm1heavier] 
    m1bh[maskm2heavier] = m2[maskm2heavier]
    m2ns[maskm1heavier] = m2[maskm1heavier]
    m2ns[maskm2heavier] = m1[maskm2heavier]
    
    spinbh[maskm1heavier] = spin1[maskm1heavier] 
    spinbh[maskm2heavier] = spin2[maskm2heavier]
    spinns[maskm1heavier] = spin2[maskm1heavier]
    spinns[maskm2heavier] = spin1[maskm2heavier]
    
    return m1bh, m2ns, spinbh, spinns # m1bh has all the heaviest systems

def f_weights_fixed(mbh,mns,thv,spin_bh,z,r0,min_delay_time,data_path,w_type,spinM1,spinM2,bhx,nsx,spinx,binsx,binsy,binsz):
    fdata = h5.File(data_path)
    fDCO = fdata['doubleCompactObjects']
    
    M1 = fDCO['M1'][...].squeeze()    # Compact object mass of star 1 
    M2 = fDCO['M2'][...].squeeze()   # Compact object mass of star 2
    
    m1, m2, spin1, spin2 = obtainM1BHandM2NS_spin(M1, M2, spinM1, spinM2)
    weights = fdata['weights_intrinsic'][w_type][...].squeeze()
    
    w = weights
    H, edges= np.histogramdd((m1,m2,spin1),weights = w,bins=(binsx,binsy,binsz),range=(bhx,nsx,spinx))
    xedges = edges[0]
    yedges = edges[1]
    zedges = edges[2]
    
    xcentre = np.zeros(binsx)
    ycentre = np.zeros(binsy)
    zcentre = np.zeros(binsz)
    
    for j in range(len(xedges)-1):
        xcentre[j] = xedges[j] + (xedges[j+1] - xedges[j]) / 2.
    
    for j in range(len(yedges)-1):
        ycentre[j] = yedges[j] + (yedges[j+1] - yedges[j]) / 2.
        
    for j in range(len(zedges)-1):
        zcentre[j] = zedges[j] + (zedges[j+1] - zedges[j]) / 2.
    
    interp_3 = RegularGridInterpolator((xcentre,ycentre,zcentre),H,bounds_error=False,fill_value=0.) #bounds_error=False,fill_value=0.
    w_m1m2spin = interp_3((mbh,mns,spin_bh))
    
    #Redshift z
    
    r0BNS = r0
    tcosmo = np.arange(0.,13.,min_delay_time)*1.e9 #5.e-2
    zeta_0 = np.linspace(10**(-2.3),10**(1.),1000) #0.6
    tcosmo_0 = cosmo.lookback_time(zeta_0).value*1.e9
    zp=np.interp(tcosmo,tcosmo_0,zeta_0)
    phi = 0.015*((1+zp)**(2.7))/(1+((1+zp)/2.9)**5.6)
    
    rd = np.zeros(len(tcosmo))
    for i in range(len(rd)-1):
        y = phi*(tcosmo-tcosmo[i])**-1
        rd[i] = np.trapz(y[i+1:],tcosmo[i+1:])
    rho = np.interp(z,zp,rd)*(r0BNS/rd[0])
    dvdz=4.*np.pi*cosmo.differential_comoving_volume(z).to("Gpc3 sr-1").value
    dpdz=rho*dvdz/(1+z)
    w_z = dpdz*z
    
    #Viewing angle
    w_thv = np.sin(thv)
    
    #Total
    #w = w_m1m2spin * w_thv * w_z  #uniform distribution
    w = thv* mbh * spin_bh * w_m1m2spin * w_thv * w_z #log distribution in mbh e spinbh
    
    #Monte Carlo
    cz = cumtrapz(dpdz[z.argsort()],np.sort(z),initial=0.)
    C_mc =cz[-1]/np.sum(w)
    
    return C_mc, w
	
    
def f_weights(mbh,mns,thv,spin_bh,z,data_path,spinM1,spinM2,bhx,nsx,spinx,binsx,binsy,binsz):
    
    print("Computing weights")
    
    fdata = h5.File(data_path)
    fDCO = fdata['doubleCompactObjects']
    
    M1 = fDCO['M1'][...].squeeze()    # Compact object mass of star 1 
    M2 = fDCO['M2'][...].squeeze()   # Compact object mass of star 2

    redshifts = fdata['Rates_mu00.035_muz-0.23_alpha0.0_sigma00.39_sigmaz0.0']['redshifts'][...].squeeze()
    w_per_z_per_system = fdata['Rates_mu00.035_muz-0.23_alpha0.0_sigma00.39_sigmaz0.0']['merger_rate'][...].squeeze()
    DCOmask = fdata['Rates_mu00.035_muz-0.23_alpha0.0_sigma00.39_sigmaz0.0']['DCOmask'][...].squeeze()
    
    m1, m2, spin1, spin2 = obtainM1BHandM2NS_spin(M1[DCOmask], M2[DCOmask], spinM1[DCOmask], spinM2[DCOmask])
    weights = fDCO['weight'][...].squeeze()[DCOmask]
    
    #Creating a 4D histogram
    H = np.zeros([binsx,binsy,binsz,len(w_per_z_per_system[0,:])])

    dpdz = np.zeros(w_per_z_per_system.shape[1])
    
    for i in range (len(w_per_z_per_system[0,:])):
        w_per_z = w_per_z_per_system[:,i] 
        dvdz = 4.*np.pi*cosmo.differential_comoving_volume(redshifts[i]).to('Gpc3 sr-1').value
        dpdz[i] = np.sum(w_per_z)*dvdz/(1+redshifts[i])
        W = w_per_z*dvdz/(1+redshifts[i])
        H[:,:,:,i], edges= np.histogramdd((m1,m2,spin1),weights = W,bins=(binsx,binsy,binsz),range=(bhx,nsx,spinx))
    xedges = edges[0]
    yedges = edges[1]
    zedges = edges[2]
    
    xcentre = np.zeros(binsx)
    ycentre = np.zeros(binsy)
    zcentre = np.zeros(binsz)
    
    for j in range(len(xedges)-1):
        xcentre[j] = xedges[j] + (xedges[j+1] - xedges[j]) / 2.
    
    for j in range(len(yedges)-1):
        ycentre[j] = yedges[j] + (yedges[j+1] - yedges[j]) / 2.
        
    for j in range(len(zedges)-1):
        zcentre[j] = zedges[j] + (zedges[j+1] - zedges[j]) / 2.
    
    #Interpolation of the 4D instogram
    
    interp_4 = RegularGridInterpolator((xcentre,ycentre,zcentre,redshifts[:100]),H)    
    w_m1m2zspin = np.zeros(len(mbh))
    w_m1m2zspin[z<0.9] = interp_4((mbh[z<0.9],mns[z<0.9],spin_bh[z<0.9],z[z<0.9]))
    
    #Viewing angle
    w_thv = np.sin(thv)
    
    #Total
    w = thv*mbh*spin_bh*w_m1m2zspin * w_thv * z
    
    #Monte Carlo
    dpdz_i = np.interp(z[z<0.9],redshifts[:100],dpdz)
    cz = cumtrapz(dpdz_i[z[z<0.9].argsort()],np.sort(z[z<0.9]),initial=0.)
    C_mc =cz[-1]/np.sum(w)
    
    return C_mc, w

