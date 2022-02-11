import os
from matplotlib.cbook import index_of
import numpy as np
import csv
import matplotlib.pyplot as plt
from itertools import islice
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.numeric import NaN
from numpy.linalg.linalg import matrix_power
from pandas.core.frame import DataFrame
import spglib as spg
import klist_generator as kg
import irrep as irp
import numpy as np
import sys
import phonopy
import matplotlib.pyplot as plt
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import read_crystal_structure
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import time
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import numpy as np
import direct_product as dp
import re

def expotential2linear(expotential_form):
    pi=3.1415926535898
    x=expotential_form[0]*(np.cos(expotential_form[1]*pi/180))
    y=expotential_form[0]*(np.sin(expotential_form[1]*pi/180))
    y=round(y,5)
    linear_form=complex(x,y)

    return linear_form

def little_group_spider(sg,k):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--incognito")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    driver.get('https://www.cryst.ehu.es/rep/repres.html')
    spgnumber = str(sg)
    try:
        spg_box = driver.find_element_by_name("ita")
    except:
        sup_buttom = driver.find_elements_by_xpath("//div/div[2]/button[3]")
        sup_buttom[0].click()
        nxt_buttom = driver.find_elements_by_xpath("//div/div[3]/p[2]/a")
        nxt_buttom[0].click()
        spg_box = driver.find_element_by_name("ita")
    
    
    
    next_button = driver.find_element_by_name("next")
    spg_box.send_keys(spgnumber)
    next_button.click()

    # to page https://cryst.ehu.es/cgi-bin/cryst/programs/representations_vec.pl

    k0 = driver.find_elements_by_name("k0")
    k1 = driver.find_elements_by_name("k1")
    k2 = driver.find_elements_by_name("k2")

    k0[0].send_keys(k[0])
    k1[0].send_keys(k[1])
    k2[0].send_keys(k[2])

    countinue_buttom = driver.find_element_by_name('continue')
    countinue_buttom.click()
    tables = driver.find_elements_by_xpath("//pre")

    s = tables[0].get_attribute("innerHTML")

    ff = open('k_spg_ir.txt','w')
    
    ff.writelines(s)
    s = s.split('\n')
    # ff = open('k_spg_ir.txt','r')
    # s = ff.readlines()
    
    ir_label = []
    ir_dim = []
    q = 0
    for i in range(len(s)):  
        if s[i].find('Space group number') != -1:
            sg = s[i].split(',')[0].split()[-1]
            sglabel = s[i].split(',')[1]
        if s[i].find('Lattice type') != -1:
            lattice_type = s[i].split()[-1]
        if s[i].find('translation coset representatives') != -1:
            nopt = int(s[i-1].split()[-1])
            rotations = np.zeros((nopt,3,3))
            translation = np.zeros((nopt,3))
            
            for j in range(nopt):
                for k in range(3):
                    for h in range(3):
                        rotations[j][k][h] = s[i+3+k+(j//4)*5].split()[(j%4)*4+h]
                        translation[j][k] = s[i+3+k+(j//4)*5].split()[(j%4)*4+3]
        if s[i].find("allowed irreps") != -1:
            n_ir = int(s[i].split()[-3])   
            ir_character = np.zeros((n_ir,nopt),dtype = complex)  
        if s[i].find('dimension') != -1:
            ir_label.append(s[i].split()[1])
            ir_dim.append(s[i].split()[-1])
            a = re.findall('\d+.?\d*',s[i+2])             
            for i in range(nopt):
                character = (float(a[i*2]),float(a[i*2+1]))
                character = dp.expotential2linear(character)
                ir_character[q][i] = character
            q += 1
    k_little_group = {
        "nopt" : nopt,
        "rotations" : rotations,
        "translation" : translation,
        "n_ir" : n_ir,
        "ir_label" : ir_label,
        "ir_dim" : ir_dim,
        "ir_character" : ir_character
    }

    return k_little_group
#time.sleep(10000)  

def get_electron(filename):
    ff=open(filename,'r')
    content=ff.readlines()
    high_sym_points=[]
    VB_irreps=[]
    CB_irreps=[]
    for line in content[1:]:
        if line.find('NELECT') !=-1:
            nelect=int(line.split()[line.split().index('NELECT')+2])
        if line.find('sg') != -1:
            for i in line.split():
                if i.find('sg') !=-1:
                    sg=int(i[2:])
        if line.find('Irreps') !=-1 :
            high_sym_points.append(line.split()[5])
    nelect_per_state=2
    for line in content[1:]:
        if line.find('band') !=-1:
            if line.split()[0].find('band') !=-1:
                region=line.split()[1].split(':')[0].split('-')
                bmin=int(region[0])
                if len(region) >= 2:
                    bmax = int(region[1])
                    if bmin <= nelect and bmax >= nelect:
                        VB_irreps.append(line.split()[2:])
                    if bmin <= nelect+1 and bmax >= nelect+1:
                        CB_irreps.append(line.split()[2:])
                else:
                    if bmin == nelect:
                        VB_irreps.append(line.split()[2:])
                    if bmin == nelect+1:
                        CB_irreps.append(line.split()[2:])
    Elecdict={
            'sg'                :   sg,
            'nelect'            :   nelect,
            'high_sym_points'   :   high_sym_points,
            'VB_irreps'         :   VB_irreps,
            'CB_irreps'         :   CB_irreps
            }
    return Elecdict

""" def linear2expotential(linear_form):
 """

def get_phonon(filename):
    ff=open(filename,'r')
    content=ff.readlines()
    idx=[]
    imode=[]
    frequency=[]
    for line in content:
        if line.find('LABEL') != -1:
            idx.append(content.index(line))
            label=line.split()
            imode.append(label[1])
            frequency.append(label[3].split(')')[0])
    nmode=len(idx)
    nmatrix=int((len(content)/nmode-1)/6)
    mode_info=[]
    symbol=[]
    matrices=[]
    characters=[]
    for i in range(nmode):
        mode_info.append(content[1+nmatrix*6*i+i:1+nmatrix*6*(i+1)+i])
    flag=0
    for mode in mode_info:
        for i in range(nmatrix):
            mat=np.zeros((3,3))
            for j in range(6):
                element=mode[6*i+j]                
                if j==1:
                    symbol.append(element)
                for k in range(3):
                    if j==k+2:
                        for h in range(3):
                            mat[k][h]=element.split()[h]
                if j==5:
                    q=element.split()
                    norm=float(q[1].split(',')[0])
                    degree=float(q[2].split(')')[0])
                    exponent=(norm,degree)
                    linear=expotential2linear(exponent)
                    characters.append(linear)
            if flag==0:
                matrices.append(mat)
        flag=1
    characters=np.array(characters)
    characters=np.reshape(characters,(nmode,nmatrix))

    Phonondict={
            'imode'                :    imode,
            'frequency'            :    frequency,
            'nmatrix'              :    nmatrix,
            'matrices'             :    matrices,
            'characters'           :    characters 
            }
    if Phonondict['imode']==['1']:
        Phonondict['characters']=np.vstack((Phonondict['characters'],Phonondict['characters'],Phonondict['characters']))
    if Phonondict['imode']==['1','3']:
        Phonondict['characters']=np.insert(Phonondict['characters'],1,values=Phonondict['characters'][0],axis=0)
    if Phonondict['imode']==['1','2']:
        Phonondict['characters']=np.insert(Phonondict['characters'],2,values=Phonondict['characters'][1],axis=0)
    return Phonondict

def get_VBM_CBM(filename):
    kin = []
    ff = open(filename,'r')
    content = ff.readlines()
    for line in content:
        if line.find('HSP') != -1:
            info = line.split()
            sg = info[0]
            material = info[1]
            mp_number = info[2]
            VBM = line.split("[")[1].split("]")[0].split()            
            CBM = line.split("]")[-2].split("[")[-1].split()
            Direct_gap = float(line.split()[line.split().index("Direct_gap:")+1])
            Global_gap = float(line.split()[line.split().index("Global_gap:")+1])
            VBM[:] = map(float,VBM)
            for i in range(3):
                VBM[i] = round(VBM[i],10)
            CBM[:] = map(float,CBM)
            for i in range(3):
                CBM[i] = round(CBM[i],10)
            dic = {
                "sg" : sg,
                "material" : material,
                "mp_number" : mp_number,
                "VBM" : VBM,
                "CBM" : CBM,
                "Direct_gap" : Direct_gap,
                "Global_gap" : Global_gap
                            }
            
            kin.append(dic)
    data = DataFrame(kin)

    return data

def direct_product(kpoint,qpoint,el_ir_file,ph_ir_file,mode):   
    print("scattering channel: ",end='')
    print(qpoint)
    #get_electronic_ir
    elecdic=get_electron(el_ir_file)  
    print(elecdic)
    sg = elecdic['sg']
    k_position = {'u':kpoint[0], 'v':kpoint[1], 'w':kpoint[2]}
    lgrps = irp.loadIR(sg, k_position = k_position)
    for grp in lgrps:        
        if grp.klabel in elecdic['high_sym_points']:
            irreps = grp.irreps
            idx=elecdic['high_sym_points'].index(grp.klabel)
            for ir in irreps:
                if ir.label in elecdic[mode+'_irreps'][idx][0]:
                    electron_irr={
                        'matrices':grp.rotP,
                        'characters':ir.characters
                    }
                #elif ir.label in elecdic[mode+'_irreps'][idx]
    print(electron_irr)
    electron_irr["matrices"] = np.unique(electron_irr["matrices"],axis=0)    
    #get_phonon_ir
    phonon=phonopy.load(ph_ir_file)
    symmetry = phonon.get_symmetry()
    phonon.set_irreps(qpoint, 1e-4)
    ct = phonon.get_irreps()
    ct.writeright_csv()#调用phonopy.irreps
    phonon_irr=get_phonon('q_ir_info')
    nmode=len(phonon_irr['imode'])
    print(phonon_irr)
    #get_mutual_subgroup
    elgroup=electron_irr['matrices']
    phgroup=phonon_irr['matrices']
    nphmats=len(phgroup)
    mutual_subgroup=[]
    idxelec=[]
    idxphon=[]
    ie=0
    for i in elgroup:
        ip=0
        for j in phgroup:
            if (i==j).all()==True:
                mutual_subgroup.append(i)
                idxelec.append(ie)
                idxphon.append(ip)
            ip+=1
        ie+=1    
    elcharacter=np.zeros(len(mutual_subgroup),dtype=np.complex)
    phcharacter=np.zeros((3,len(mutual_subgroup)),dtype=np.complex)
    for i in range(len(idxelec)):
        elcharacter[i]=electron_irr['characters'][idxelec[i]]
    for i in range(len(idxphon)):
        for j in range(3):
            phcharacter[j][int(i)]=phonon_irr['characters'][j][idxphon[i]]    
    direct_product=np.dot(elcharacter*elcharacter.reshape(1,len(idxelec)),np.transpose(phcharacter))
    direct_product=direct_product.reshape(3)
    dp = np.array([[0,0],[0,0],[0,0]])
    for i in range(3):
        dp[i][0]=abs(round(np.real(direct_product[i]),3))
        dp[i][1]=abs(round(np.imag(direct_product[i]),3))


    if len(mutual_subgroup)!=1:
        selection_rules={
            'ZA':'Allowed' if direct_product[0]!=0 else 'Prohibited',
            'TA':'Allowed' if direct_product[1]!=0 else 'Prohibited',
            'LA':'Allowed' if direct_product[2]!=0 else 'Prohibited'
        }
    else:
            selection_rules={
            'ZA':'Prohibited',
            'TA':'Prohibited',
            'LA':'Prohibited',
            'Reason':'Uncompatible'
        }
    def print_report():
        print('================electron_operations================')
        for i in elgroup:
            print(i)
        print('=================phonon_operations==================')
        for i in phgroup:
            print(i)
        print('=================mutual_operations==================')
        for i in mutual_subgroup:
            print(i)
        if len(mutual_subgroup)==1:
            print('Uncompatible!')
        else:
            print('===========subgroup_electron_characters=============')
            print(elcharacter)
            print('=============subgroup_phonon_characters=============')
            print(phcharacter)
            print('==================direct_product====================')
            print(direct_product)
        return
    print_report()
    return dp

def main():
    data = dp.get_VBM_CBM('valley_info')
    data = data.reindex(columns=['sg','material','mp_number','VBM','VBM degeneracy','VB prohibiting rate','CBM','CBM degeneracy','CB prohibiting rate','Direct_gap','Global_gap'], fill_value=NaN)
    error_list = []
    for index,row in data[5862:5863].iterrows():
        print('=============================================================================================')
        print('index: ',end='')
        print(index)
        # print(row)
        try:
            mp = row['mp_number']
            os.chdir("/home2/awu/12-SRTE/data/"+mp)
            kCBM = row["CBM"]
            kVBM = row["VBM"]
            VBMqpts_cry_cor,sg=kg.generate_k(kVBM,"POSCAR")
            data.iloc[index,4] = len(VBMqpts_cry_cor)
            CBMqpts_cry_cor,sg=kg.generate_k(kCBM,"POSCAR")
            data.iloc[index,7] = len(CBMqpts_cry_cor)
            
            for i in range(3):
                while kVBM[i]>0.5:
                    kVBM[i]+=-1
                while kVBM[i]<=-0.5:
                    kVBM[i]+=1        
            
            if len(VBMqpts_cry_cor) != 1:
                VBselection_rules = pd.DataFrame(columns=['ZA','LA','TA'],index=range(len(VBMqpts_cry_cor)-1))
                iq = 0
                nzeros = 0
                for qi in VBMqpts_cry_cor: 
                    if (qi == [0,0,0]).all()==False:                     
                        sr = direct_product(kVBM,qi,mp+".dat","phonopy.yaml","VB")
                        VBselection_rules['ZA'][iq] = (sr[0][0],sr[0][1])
                        if sr[0][0] == 0 and sr[0][1] == 0:
                            nzeros += 1
                        VBselection_rules['TA'][iq] = (sr[1][0],sr[1][1])
                        if sr[1][0] == 0 and sr[1][1] == 0:
                            nzeros += 1
                        VBselection_rules['LA'][iq] = (sr[2][0],sr[2][1])
                        if sr[2][0] == 0 and sr[2][1] == 0:
                            nzeros += 1
                        iq += 1
                print('VB selection rules')
                print(VBselection_rules)
                a = VBselection_rules['ZA'].value_counts()
                VBprohibit_rate = float(nzeros/(3*iq))
                VBprohibit_rate = "%.2f%%" % (VBprohibit_rate * 100)                
            else:
                VBprohibit_rate = "%.2f%%" % (0 * 100)
            data.iloc[index,5] = VBprohibit_rate
            
            for i in range(3):
                    while kCBM[i]>0.5:
                        kCBM[i]+=-1
                    while kCBM[i]<=-0.5:
                        kCBM[i]+=1
            
            if len(CBMqpts_cry_cor) != 1:
                CBselection_rules = pd.DataFrame(columns=['ZA','LA','TA'],index=range(len(CBMqpts_cry_cor)-1))
                iq = 0
                nzeros = 0
                for qi in CBMqpts_cry_cor:
                    if (qi == [0,0,0]).all()==False:                    
                        sr = direct_product(kCBM,qi,mp+".dat","phonopy.yaml","CB")
                        CBselection_rules['ZA'][iq] = (sr[0][0],sr[0][1])
                        if sr[0][0] == 0 and sr[0][1] == 0:
                            nzeros += 1
                        CBselection_rules['TA'][iq] = (sr[1][0],sr[1][1])
                        if sr[1][0] == 0 and sr[1][1] == 0:
                            nzeros += 1
                        CBselection_rules['LA'][iq] = (sr[2][0],sr[2][1])
                        if sr[2][0] == 0 and sr[2][1] == 0:
                            nzeros += 1
                        iq += 1
                print('CB selection rules')
                print(CBselection_rules)
                a = CBselection_rules['ZA'].value_counts()
                CBprohibit_rate = float(nzeros/(3*iq))
                CBprohibit_rate = "%.2f%%" % (CBprohibit_rate * 100)
            else:
                CBprohibit_rate = "%.2f%%" % (0 * 100)
            data.iloc[index,8] = CBprohibit_rate
            
            os.chdir("/home2/awu/BORN/")
            #data.to_csv('data.csv')
        except Exception as error:
            print(error)

if __name__ == '__main__':
    # outputmode = sys.argv[1]
    main()
    

