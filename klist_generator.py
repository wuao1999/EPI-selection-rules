import os
from matplotlib.cbook import index_of
import numpy as np
import csv
import matplotlib.pyplot as plt
from itertools import islice
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.numeric import NaN
import spglib as spg
import direct_product as dp
import pandas as pd
from pandas import Series,DataFrame
#Klistgenerator

pd.set_option('display.max_rows', 3896)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

def kpts_vis(cartesian_cor):
        fig = plt.figure(figsize=(5,5))
        ax = Axes3D(fig)
        for i in range(len(cartesian_cor)):
            if i != 0:
                ax.scatter(cartesian_cor[i][0],cartesian_cor[i][1],cartesian_cor[i][2],c='blue',label='degenerate points')
        
        ax.scatter(cartesian_cor[0][0],cartesian_cor[0][1],cartesian_cor[0][2],c='red',label='original point')
        plt.show()
        #plt.savefig('1.pdf')

def get_POSCAR(poscar):
    ff=open(poscar,'r')
    content=ff.readlines()
    volumefactor=float(content[1].strip())
    a=content[6].strip().split()
    a=list(map(int,a))
    natom=sum(a)
    axis=np.zeros((3,3))
    positions=np.zeros((natom,3))
    nspecies=len(content[5].split())
    natom_of_species=content[6].strip().split()
    numbers=list()
    for i in range(nspecies):
        for j in range(int(natom_of_species[i])):
            numbers.append(i+1)
    for i in range(3):
        for j in range(3):
            axis[i][j]=content[i+2].split()[j]
    lattice=volumefactor*axis
    for i in range(natom):
        for j in range(3):
            positions[i][j]=content[i+8].split()[j]
    return lattice,positions,numbers

def get_optmats():
    f=csv.reader(open('right_PbTe.csv','r'))
    mode=list()   
    ctt=list()
    idx=list()
    iele=list()
    symbolele=list()
    axe1=list()
    axe2=list()
    axe3=list()
    character=list()
    for element in f:
        ctt.append(element)
    for i in range(len(ctt)):
        if len(ctt[i][0].split())==4:
            idx.append(i)
    try:
        ntrmats=int((idx[1]-idx[0]-2)/6)
    except:
        ntrmats=int((len(ctt)-2)/6)#得到群元数目    
    mode1=ctt[slice(ntrmats*6+1)]
    mode2=ctt[slice(ntrmats*6*2+1)]    
    A=list()
    for i in range(ntrmats):
        for j in range(6):
            ele=mode1[6*i+j+1]
            if j==0:
                iele.append(ele)
            if j==1:
                symbolele.append(ele)
            if j==2:
                axe1.append(ele)
            if j==3:
                axe2.append(ele)
            if j==4:
                axe3.append(ele)
            if j==5:
                character.append(ele)
    for i in range(ntrmats):
        A.append(
                np.array([  
                            [axe1[i][0].split()[0],  axe1[i][0].split()[1],  axe1[i][0].split()[2]],
                            [axe2[i][0].split()[0],  axe2[i][0].split()[1],  axe2[i][0].split()[2]],
                            [axe3[i][0].split()[0],  axe3[i][0].split()[1],  axe3[i][0].split()[2]]  ])
                                                                                                        )
    for i in range(len(A)):
        A[i]=A[i].astype(np.float)

    return A

def real2reciprocal(real_lattice):
    pi=3.1415926535898
    reciprocal_lattice=np.zeros((3,3))
    for i in range(3):
        reciprocal_lattice[i]=2*pi*(
                                    np.cross(real_lattice[(i+1)%3],real_lattice[(i+2)%3])
                                    /
                                    np.dot(real_lattice[0],np.cross(real_lattice[1],real_lattice[2]))     
                                    )

    return reciprocal_lattice

def generate_k(kpoint,poscar):
    kin=np.array(kpoint)
    tolerance = 8
    #Type in k point in crystal coordinates
    lattice, positions, numbers = get_POSCAR(poscar)
    cell = (lattice, positions, numbers)
    primitive_lattice, scaled_positions, numbers = spg.find_primitive(cell, symprec=1e-5)
    symmetry_diction = spg.get_symmetry(cell, symprec=1e-3)
    space_group = spg.get_spacegroup(cell, symprec=1e-3)
    # print('space group from phonondb:',end='')
    # print(space_group)
    #print(symmetry_diction)

    idxlst=[]
    
    R = symmetry_diction["rotations"]
    t = symmetry_diction["translations"]

    #Time reversal operation

    time_rev_mat = np.array([[-1,0,0],
                             [0,-1,0],
                             [0,0,-1]])
    
    

    nopts = len(R)
    crystal_cor = np.zeros((nopts,3))
    reciprocal_lattice = np.array(real2reciprocal(primitive_lattice))
    reciprocal_lattice = np.round(reciprocal_lattice,tolerance)

    for i in range(3):
        while kin[i]>0.5:
            kin[i]+=-1
        while kin[i]<=-0.5:
            kin[i]+=1
    crystal_cor_ori = kin
    
    crystal_cor = np.dot(R,np.transpose(crystal_cor_ori))+t # r'=Rr+t
    t_crystal_cor = np.dot(crystal_cor,time_rev_mat)
    crystal_cor = np.concatenate((crystal_cor,t_crystal_cor),axis=0)

    
    crystal_cor_ir = np.unique(crystal_cor,axis=0)
    crystal_cor_ir = np.round(crystal_cor_ir,10)
    nopts_ir=len(crystal_cor_ir)
    #Visualization
    #kpts_vis(cartesian_cor)
    qpts_cry_cor=crystal_cor_ir-crystal_cor_ir[0]
    neqk=0
    for qpts in qpts_cry_cor:
        if qpts[0]%1==0 and qpts[1]%1==0 and qpts[2]%1==0:
            neqk+=1    

    for i in range(nopts_ir):
        for j in range(3):
            while qpts_cry_cor[i][j]>0.5:
                qpts_cry_cor[i][j]+=-1
            while qpts_cry_cor[i][j]<=-0.5:
                qpts_cry_cor[i][j]+=1
    qpts_cry_cor=np.round(qpts_cry_cor,10)
    qpts_cry_cor=np.unique(qpts_cry_cor,axis=0)

    
    # print('Kpoint degeneracies:'+str(len(qpts_cry_cor)))
    # print('Qpointlist:'+'\n'+str(qpts_cry_cor))

    return qpts_cry_cor,space_group

def myprint(*args):

    print(*args)

    #passp

if __name__ == '__main__':
    data = dp.get_VBM_CBM('valley_info')
    data = data.reindex(columns=['sg','material','mp_number','VBM','VBM degeneracy','CBM','CBM degeneracy','Direct_gap','Global_gap'], fill_value=NaN)
     
    error_list = []
    for index,row in data.iterrows():
        try:
            # print("<================================================================>")            
            mp = row['mp_number']
            # print("index:"+str(idx+1))
            # print(kin[idx]["material"])
            # print("space group:",end='')
            # print(kin[idx]["sg"])
            # mp = material
            # print("material:"+mp+'\n')
            os.chdir("/home/awu/12-SRTE/data/"+mp)
            kCBM = row["CBM"]
            kVBM = row["VBM"]
            for i in range(3):
                while kCBM[i]>0.5:
                    kCBM[i]+=-1
                while kCBM[i]<=-0.5:
                    kCBM[i]+=1
            # print("CBM Location:",end='')
            # print(kCBM)
            CBMqpts_cry_cor=generate_k(kCBM,"POSCAR")
            data['CBM degeneracy'][index] = len(CBMqpts_cry_cor)
            
            
            
            
            for i in range(3):
                while kVBM[i]>0.5:
                    kVBM[i]+=-1
                while kVBM[i]<=-0.5:
                    kVBM[i]+=1
            # print("VBM Location:",end='')
            # print(kVBM)
            VBMqpts_cry_cor=generate_k(kVBM,"POSCAR")
            data['VBM degeneracy'][index] = len(VBMqpts_cry_cor)
            print(CBMqpts_cry_cor)
            print(VBMqpts_cry_cor)
        except Exception as error:
            error_list.append(error)
    #data['CBM degeneracy'][1] = 1
    print(data)   
    os.chdir("/home/awu/BORN/")
    # data1 = data.sort_values(by='CBM degeneracy',axis=0,ascending=False)
    # data2 = data.sort_values(by='VBM degeneracy',axis=0,ascending=False)
    data.to_csv('data.csv')
    # print(data1)
    # print(data2)