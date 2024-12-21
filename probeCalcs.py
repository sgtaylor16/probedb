import sqlite3
import pandas as pd
import re
import os
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from typing import List

def readAlData():
    sndir = {'rpgzg3_1':'4H-1',
         'rpgzg3_2':'4H-2',
         'rpgzg3_3':'4H-3',
         'rpgzg3_4':'4H-4',
         'gc4go4_2':'5H-1',
         'gc4go4_3':'5H-2',
         'gc4go4_4':'5H-3',
         'gc4go4_5':'5H-4',
    }

    li = []
    heightre = re.compile('H[1-5]')
    probetypere = re.compile('R[4,5]H_')
    probenumre = re.compile('_[1-9]_H[1-9]')
    probesnre = re.compile('(?<=HP_).*?(?=_H)')
    for onefile in os.listdir("../Vectoflow/allcsvfiles"):
        if '.csv' in onefile:
            df = pd.read_csv('../Vectoflow/allcsvfiles/'+ onefile,header = 12,sep=';',decimal=',')
            df.columns = df.columns.str.strip()
            df['Probe Height'] = int(heightre.search(onefile)[0][1:])
            df['Probe SN'] = probesnre.search(onefile)[0]
            df['Probe Code'] = df['Probe SN'].apply(lambda x: sndir[x])
            df['unique'] = df['Probe Code'] + 'H' + df['Probe Height'].astype(str)
            li.append(df)  
    alldata = pd.concat(li,axis=0,ignore_index=True)
    alldata.columns = ['Theta','Phi','Alpha','Beta','P1','P2','P3','P4','P5','Pref','Pt','Ps','Ts','Ttot','M','Probe Height','Probe SN','Probe Code','unique']
    frame = alldata.rename({'M':'Mach #'},axis=1)

    frame['P1']= frame['P1'] + frame['Pref']
    frame['P2']= frame['P2'] + frame['Pref']
    frame['P3']= frame['P3'] + frame['Pref']
    frame['P4']= frame['P4'] + frame['Pref']
    frame['P5']= frame['P5'] + frame['Pref']
    frame['pbar'] = 0.25 * (frame['P2'] + frame['P3'] + frame["P4"] + frame['P5']) 
    frame['Cpalpha'] = (frame['P5'] - frame['P4']) / (frame['P1'] - frame['pbar'])
    frame['Cpbeta'] = (frame['P3'] - frame['P2']) / (frame['P1'] - frame['pbar'])
    frame['Cp_static'] = (frame['pbar'] - frame['Ps']) / (frame['P1'] - frame['pbar'])
    frame['Cp_total'] = (frame['P1'] - frame['Pt']) / (frame['P1'] - frame['pbar'])
    #frame['Cp_mach'] = ((frame['P1'] - frame['pbar']) / (frame['P1']))
    frame['Cp_mach'] = 1 - (frame['pbar']) /(frame['P1'])
    frame = frame.loc[frame['Mach #'] < 1]

    return frame

def check_pressure_count(pressures:List[float]):
    if len(pressures) != 5:
        raise ValueError("Input must be a list of 5 pressures")

def cp_alpha(pressures:List[float]) -> float:
    check_pressure_count(pressures)
    p1,p2,p3,p4,p5 = pressures
    pbar = .25*(p2+p3+p4+p5)
    return (p5 - p4)/(p1 - pbar)

def cp_beta(pressures:List[float]) -> float:
    check_pressure_count(pressures)
    p1,p2,p3,p4,p5 = pressures
    pbar = .25*(p2+p3+p4+p5)
    return (p3 - p2)/(p1 - pbar)

def cp_mach(pressures:List[float]) -> float:
    check_pressure_count(pressures)
    p1,p2,p3,p4,p5 = pressures
    pbar = .25*(p2+p3+p4+p5)

    return 1 - (pbar/p1)

def calc_static_pressure(pressures:List[float],cp_static:float) -> float:
    check_pressure_count(pressures)
    p1,p2,p3,p4,p5 = pressures
    pbar = .25*(p2+p3+p4+p5)
    D = p1 - pbar
    return -(cp_static * D - pbar)

def calc_total_pressure(pressures:List[float],cp_total:float) -> float:
    check_pressure_count(pressures)
    p1,p2,p3,p4,p5 = pressures
    pbar = .25*(p2+p3+p4+p5)
    D = p1 - pbar
    return p1 - cp_total * D

class Probe:
    def __init__(self,dbloc: str,rakename:str,probeheight:int):
        self.dbloc = dbloc
        con = sqlite3.connect(dbloc)
        probe_id = pd.read_sql_query("SELECT ID FROM PROBES WHERE RAKE_SN = '{}' AND HEIGHT = {}".format(rakename,probeheight),con)
        
        if probe_id.shape[0] == 0:
            raise ValueError("Probe not found in database")
        elif probe_id.shape[0] > 1:
            raise ValueError("Multiple probes found in database")
        else:
            self.probe_id = probe_id.loc[0,'ID']


        # Load the calibration data
        alpha  = pd.read_sql_query("SELECT COEFF, VALUE FROM ALPHA WHERE PROBE_ID = {}".format(self.probe_id),con)
        alpha = alpha.sort_values(by='COEFF')
        self.alpha = alpha['VALUE'].to_numpy()

        beta = pd.read_sql_query("SELECT COEFF, VALUE FROM BETA WHERE PROBE_ID = {}".format(self.probe_id),con)
        beta = beta.sort_values(by='COEFF')
        self.beta = beta['VALUE'].to_numpy()

        mach = pd.read_sql_query("SELECT COEFF, VALUE FROM MACH WHERE PROBE_ID = {}".format(self.probe_id),con)
        mach = mach.sort_values(by='COEFF')
        self.mach = mach['VALUE'].to_numpy()

        static = pd.read_sql_query("SELECT COEFF, VALUE FROM STATIC_PRESSURE WHERE PROBE_ID = {}".format(self.probe_id),con)
        static = static.sort_values(by='COEFF')
        self.static = static['VALUE'].to_numpy()

        total = pd.read_sql_query("SELECT COEFF, VALUE FROM TOTAL_PRESSURE WHERE PROBE_ID = {}".format(self.probe_id),con)
        total = total.sort_values(by='COEFF')
        self.total = total['VALUE'].to_numpy()

    def predict(self,pressures:List[float]):
        check_pressure_count(pressures)
        p1,p2,p3,p4,p5 = pressures

        x = np.array([cp_mach(pressures),cp_alpha(pressures),cp_beta(pressures)]).reshape(1,-1)

        totalX = PolynomialFeatures(degree=2).fit_transform(x).flatten()
        staticX = PolynomialFeatures(degree=4).fit_transform(x).flatten()
        machX = PolynomialFeatures(degree=4).fit_transform(x).flatten()
        angleX = PolynomialFeatures(degree=5).fit_transform(x).flatten()

        total = calc_total_pressure(pressures,self.total.dot(totalX))
        static = calc_static_pressure(pressures,self.static.dot(staticX))
        mach = self.mach.dot(machX)
        alpha = self.alpha.dot(angleX)
        beta = self.beta.dot(angleX)

        return {'total':total,'static':static,'mach':mach,'alpha':alpha,'beta':beta}

def calc_values(probesn:str,height:int,pressures:List[float]):

