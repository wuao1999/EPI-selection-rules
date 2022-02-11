import os
import sys
import re
import io
from collections import defaultdict
import getopt
import numpy as np
from math import cos, sin, acos, asin, pi
from numpy.linalg import inv, norm
import IR_DSG

locate=os.path.abspath(os.path.dirname(__file__))

# =====================================================================================
# Lattice
# =====================================================================================
SGTricP = [1, 2]
SGMonoP = [3, 4, 6, 7, 10, 11, 13, 14]
SGMonoB = [5, 8, 9, 12, 15]
SGOrthP = [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, \
           48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
SGOrthB1 = [20, 21, 35, 36, 37, 63, 64, 65, 66, 67, 68]
SGOrthB2 = [38, 39, 40, 41]
SGOrthF = [22, 42, 43, 69, 70]
SGOrthI = [23, 24, 44, 45, 46, 71, 72, 73, 74]
SGTetrP = [75, 76, 77, 78, 81, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, \
           95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 111, 112, \
           113, 114, 115, 116, 117, 118, 123, 124, 125, 126, 127, \
           128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]
SGTetrI = [79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110, 119, 120, \
           121, 122, 139, 140, 141, 142]
SGTrigP = [146, 148, 155, 160, 161, 166, 167]
SGHexaP = [143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156, \
           157, 158, 159, 162, 163, 164, 165, 168, 169, 170, 171, \
           172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, \
           183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194]
SGCubcP = [195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218, \
           221, 222, 223, 224]
SGCubcF = [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]
SGCubcI = [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]


def PrimInConv(gid):
    if gid in SGTricP + SGMonoP + SGOrthP + SGTetrP + SGHexaP + SGCubcP:
        return np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    elif gid in SGMonoB + SGOrthB1:
        return np.array([[1. / 2, -1. / 2, 0.], [1. / 2, 1. / 2, 0.], [0., 0., 1.]])
    elif gid in SGOrthB2:
        return np.array([[1., 0., 0.], [0., 1. / 2, 1. / 2], [0., -1. / 2, 1. / 2]])
    elif gid in SGOrthI + SGTetrI + SGCubcI:
        return np.array([[-1. / 2, 1. / 2, 1. / 2], [1. / 2, -1. / 2, 1. / 2], [1. / 2, 1. / 2, -1. / 2]])
    elif gid in SGOrthF + SGCubcF:
        return np.array([[0., 1. / 2, 1. / 2], [1. / 2, 0., 1. / 2], [1. / 2, 1. / 2, 0.]])
    elif gid in SGTrigP:
        return np.array([[2. / 3, 1. / 3, 1. / 3], [-1. / 3, 1. / 3, 1. / 3], [-1. / 3, -2. / 3, 1. / 3]])
    else:
        raise ValueError('wrong gid !!!')


# =============================================================================
# Little group and irreps
# =============================================================================
class LittleGroup:  # little group at a special k point
    def __init__(self):
        self.gid = ''  # space group number as a string
        self.klabel = ''  # label for the k point
        self.kvecC = np.zeros(3)  # k vector
        self.rotC = []  # the rotation part of all operations as a list
        self.tauC = []  # the translation vector of all operations as a list
        self.su2s = []  # the SU2 matrices of all operations as a list, read from Bilbao
        self.irreps = []  # the irreducible representations as a list, each being an 'Irrep' object
        #
        self.kvecP = []
        self.rotP = []
        self.tauP = []
        #
        self.shiftC = np.array([0, 0, 0])
        self.shiftP = np.array([0, 0, 0])
        #
        self.basisP = []  # primitive basis, read from POSCAR
        self.su2c = []  # SU2 matrix calculated from SO3 matrix
        self.index = []  # index of SU2 matrix need to time -1, in order to align with Bilbao

        self.degenerate_pair=[] # list to collect degenerate irrep pairs
        self.kvec_orig = []


    def prim(self):
        # C : direct,     prim. lattices in conv.
        #   : reciprocal, conv. lattices in prim.
        # D : reciprocal, prim. lattices in conv.
        #   : direct,     conv. lattices in prim.
        # Cij*Di'j=delta_i'i
        #
        # ai = Cij*Aj,  bi = Dij*Bj
        # Aj = ai*Dij,  Bj = bi*Cij
        C = PrimInConv(self.gid)
        D = np.linalg.inv(C).T
        #
        self.kvecP = np.dot(C, self.kvecC)
        self.tauP = [np.dot(D, T) for T in self.tauC]
        rotP = [np.dot(np.dot(D, R), C.T) for R in self.rotC]
        self.rotP = [np.array(np.round(R), dtype=int) for R in rotP]
        if any([np.max(abs(self.rotP[i] - rotP[i])) > 1e-4 for i in range(len(rotP))]):
            raise TypeError('Non-integer rotation matrix !!!')
        #
    def shift(self, shift):
        # g*(r-r0) + t = r'-r0
        # => g*r + (r0-g*r0+t) = r'
        if norm(shift) > 1e-4:
            C = PrimInConv(self.gid)
            D = np.linalg.inv(C).T
            self.shiftC = np.array(shift)
            self.shiftP = np.dot(D, self.shiftC)
            dtauC = [self.shiftC - np.dot(self.rotC[i], self.shiftC) for i, t in enumerate(self.tauC)]
            self.tauC = [t + dtauC[i] for i, t in enumerate(self.tauC)]
            self.tauP = [np.dot(D, T) for T in self.tauC]

    def SU2(self):  # SU2 matrix calculated from SO3 matrix, may be different from su2s read from Bilbao
        sigma0 = np.array([[1, 0], [0, 1]], dtype=complex)  # Pauli matrix
        sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)

        A = np.array([self.basisP[0], self.basisP[1], self.basisP[2]], dtype=float).T
        B = np.linalg.inv(A)
        for iop in range(len(self.rotP)):
            rotCart = np.dot(np.dot(A, self.rotP[iop]), B)  # SO3 matrix in Cartesian coordinates
            angle, axis = get_rotation(rotCart)
            su2 = cos(angle / 2) * sigma0 - 1j * sin(angle / 2) \
                  * (axis[0] * sigma1 + axis[1] * sigma2 + axis[2] * sigma3)
            self.su2c.append(su2)

    def __str__(self):
        strg = 'k label: ' + self.klabel + '\n'
        strg += '  k = %6.3f %6.3f %6.3f (prim)  %6.3f %6.3f %6.3f (conv) \n' % (tuple(self.kvecC) + tuple(self.kvecP))
        for ii in range(len(self.rotP) // 2):
            strg += 'op %3d \n' % ii
            strg += '    %4d %4d %4d\n' % tuple(self.rotC[ii][0])
            strg += '    %4d %4d %4d\n' % tuple(self.rotC[ii][1])
            strg += '    %4d %4d %4d\n' % tuple(self.rotC[ii][2])
            strg += '     %10.7f %10.7f %10.7f\n' % tuple(self.tauP[ii])
            strg += '    (%10.7f %10.7f)  (%10.7f %10.7f) \n' % (self.su2s[ii][0][0].real, self.su2s[ii][0][0].imag, \
                                                                 self.su2s[ii][0][1].real, self.su2s[ii][0][1].imag)
            strg += '    (%10.7f %10.7f)  (%10.7f %10.7f) \n' % (self.su2s[ii][1][0].real, self.su2s[ii][1][0].imag, \
                                                                 self.su2s[ii][1][1].real, self.su2s[ii][1][1].imag)
        return strg


class Irrep:
    def __init__(self):
        self.label = ''  # label for the representation. 'd' as postfix indicates double-valued irreps
        self.dim = 1  # dimension of the representation
        self.matrices = np.ones((1, 1))  # representation matrices, ordered as the operations of the belonging group
        self.characters = []  # characters as a list, ordered the same as matrices


def get_rotation(R):
    det = np.linalg.det(R)
    tmpR = det * R
    arg = (np.trace(tmpR) - 1) / 2
    if arg > 1:
        arg = 1
    elif arg < -1:
        arg = -1
    angle = acos(arg)
    axis = np.zeros((3, 1))
    if abs(abs(angle) - pi) < 1e-4:
        for i in range(3):
            axis[i] = 1
            axis = axis + np.dot(tmpR, axis)
            if max(abs(axis)) > 1e-1:
                break
        assert max(abs(axis)) > 1e-1, 'can\'t find axis'
        axis = axis / np.linalg.norm(axis)
    elif abs(angle) > 1e-3:
        # standard case, see Altmann's book
        axis[0] = tmpR[2, 1] - tmpR[1, 2]
        axis[1] = tmpR[0, 2] - tmpR[2, 0]
        axis[2] = tmpR[1, 0] - tmpR[0, 1]
        axis = axis / sin(angle) / 2
    elif abs(angle) < 1e-4:
        axis[0] = 1

    return angle, axis

#k_position = {'u':0, 'v':0, 'w':0}
def loadIR(gid, k_position, basisP=np.eye(3), test=False):
    lgrps = []
    file = io.StringIO(IR_DSG.IR_DSG_str)
    HSL_data = np.load('/home2/awu/BORN/HSLdata.npy', allow_pickle=True)[gid-1]
    basisP = PrimInConv(gid)

    # find the space group
    while True:
        line = file.readline()
        print(line, end='') if test else 0
        if line.startswith('gid ' + str(gid)):
            print('****') if test else 0
            break
        elif not line:
            raise ValueError('I can\'t find ``gid ' + str(gid) + '\" in IR_DSG.dat')
    # Read data
    # k points ========================================================
    line = file.readline()
    print(line, end='') if test else 0
    assert line.startswith('nk')
    nk = int(line.split()[1])
    for i in range(nk):
        grp = LittleGroup()  ####
        grp.basisP = basisP  # primitive basis
        line = file.readline()
        print(line, end='') if test else 0
        assert line.startswith('k_label')
        grp.klabel = line.split()[1]
        grp.gid = gid
        line = file.readline()
        print(line, end='') if test else 0
        assert line.startswith('k_vec')
        s = re.search('\[(.*?)\]', line).groups()[0]
        grp.kvecC = np.array([float(x) for x in s.split()])
        grp.kvec_orig = grp.kvecC
        # operations ==================================================
        line = file.readline()
        print(line, end='') if test else 0
        assert line.startswith('nop')
        nop = int(line.split()[1])
        grp.rotC = []
        grp.tauC = []
        for i in range(nop):
            line = file.readline()
            print(line, end='') if test else 0
            assert line.startswith('operation')
            rot, trans, su2 = re.findall('\[(.*?)\]', line)
            rot = np.array([float(x) for x in rot.split()])
            rot.shape = (3, 3)
            grp.rotC.append(rot)
            trans = np.array([float(x) for x in trans.split()])
            grp.tauC.append(trans)
            su2 = np.array([complex(x) for x in su2.split()])
            su2.shape = (2, 2)
            grp.su2s.append(su2)
        # irreps ======================================================
        line = file.readline()
        print(line, end='') if test else 0
        assert line.startswith('nir')
        nir = int(line.split()[1])
        irreps = []
        for j in range(nir):
            ir = Irrep()
            line = file.readline()
            print(line, end='') if test else 0
            assert line.startswith('ir_label')
            ir.label = line.split()[1]
            line = file.readline()
            print(line, end='') if test else 0
            assert line.startswith('ir_dim')
            ir.dim = int(line.split()[1])
            line = file.readline()
            print(line, end='') if test else 0
            assert line.startswith('ir_matrices')
            mats = re.findall('\[(.*?)\]', line)
            mats = [np.array([complex(x) for x in m.split()]) for m in mats]
            for m in mats:
                m.shape = (ir.dim, ir.dim)
            ir.matrices = mats
            line = file.readline()
            print(line, end='') if test else 0
            assert line.startswith('ir_characters')
            chrcts = re.search('\[(.*?)\]', line).groups()[0]
            ir.characters = np.array([complex(x) for x in chrcts.split()])
            irreps.append(ir)
        # collect
        grp.irreps = irreps
        lgrps.append(grp)

    # read HSL data, only used for type-2 non-magnetic sg
    for k_data in HSL_data: 
        grp = LittleGroup()
        grp.basisP = basisP  # primitive basis
        grp.gid = gid
        assert k_data['gid'] == gid
        grp.klabel = k_data['k_label']
        grp.kvecC = k_data['k_vec']
        grp.kvec_orig = str(k_data['k_vec'])
        grp.rotC = k_data['rot']
        grp.tauC = k_data['tau']
        grp.su2s = k_data['su2']
        for irlabel, irmats in zip(k_data['ir_labels'], k_data['irreps']):
            ir = Irrep()
            ir.label = irlabel
            ir.dim = irmats[0].shape[0]
            ir.matrices = irmats
            grp.irreps.append(ir)
        grp = replace_HSL(grp, sub_dict=k_position)
        lgrps.append(grp)

    # Post-processing
    shift = get_shift(gid)
    for grp in lgrps:
        grp.prim()
        grp.shift(shift)
       #grp.SU2()
   #grp_GM = [grp for grp in lgrps if grp.klabel == 'GM'][0]
   #lgrps = align_su2(lgrps, grp_GM.su2s, grp_GM.su2c) if msg_dict['msg_type'] != 3 else lgrps 

    #lgrps = read_TR_degeneracy(gid, lgrps)
    return lgrps


def read_TR_degeneracy(gid, lgrps):
    TRdeg_data = np.load('TRdeg_pair_data.npy', allow_pickle=True)
    gid_data = TRdeg_data[gid - 1]
    for ik, grp in enumerate(lgrps):
        grp.degenerate_pair = gid_data[grp.klabel]  #like  [[1,2],[3,3],[4,4],[5,6]]
    return lgrps

def replace_HSL(grp, sub_dict):
    def replace(key, val, exp):
        if key not in exp:
            return exp
        elif exp[exp.index(key) - 1] in ['+', '-', '*',',','[','(']:
            return re.sub(key, str(val), exp)
        else:
            return re.sub(key, '*'+str(val), exp)

   #sub_dict = {'u':0, 'v':0, 'w':0}
    for key, val in sub_dict.items():
        grp.kvecC = replace(key, val, str(grp.kvecC))
        for ir in grp.irreps:
            irmats = []
            for mat in ir.matrices:
                if type(mat).__name__ == 'ndarray':
                    irmats.append(mat)
                    continue
                irmats.append([replace(key, val, str(num)) for num in mat])
            ir.matrices = irmats

    grp.kvecC = eval(grp.kvecC) # grp.kvecC is a str of numbers (no variable), and can directly evaluated
    for ir in grp.irreps:
        irmats = []
        for mat in ir.matrices:
            if type(mat).__name__ == 'ndarray':
                irmats.append(mat)
                continue
            irmats.append(np.array([eval(num) for num in mat], dtype=complex).reshape((ir.dim, ir.dim)))
        ir.matrices = irmats
        ir.characters = [np.trace(mat) for mat in ir.matrices]

    return grp


def align_su2(lgrps, su2_set1, su2_set2, align_which='c', tol=1e-4):
    # align su2_set2 to su2_set1. align_which='s'/'c' determines whether su2s or su2c to align
    def get_multi_table(mat_list, tol=1e-4):
        # for an input mat_list, return its multiplication table
        N = len(mat_list)
        multi_table = 100 * np.ones((n, n))  # multiplication table of su2_set1, times 100 to distinguish error
        for i, mi in enumerate(mat_list):
            for j, mj in enumerate(mat_list):
                mi_mj = mi @ mj
                for k, mk in enumerate(mat_list):
                    if norm(mk - mi_mj) < tol:
                        multi_table[i, j] = k
                        break
                    elif norm(mk + mi_mj) < tol:
                        multi_table[i, j] = -k
                        break
                assert multi_table[i, j] != 100, (multi_table, mat_list)
        return multi_table
    
    # If the input two su2 sets are same, return lgrps and change nothing 
    if all([norm(u1 - u2) < tol for u1, u2 in zip(su2_set1, su2_set2)]):
        return lgrps

    grp_GM = [grp for grp in lgrps if grp.klabel == 'GM'][0]
    n = round(len(grp_GM.rotP) / 2)

    su2_set1 = [u.copy() for u in su2_set1[:n]]
    su2_set2 = [u.copy() for u in su2_set2[:n]]
    multi_table1 = get_multi_table(su2_set1)  
    multi_table2 = get_multi_table(su2_set2) 

    minus = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if multi_table2[i, j] == -multi_table1[i, j] and multi_table1[i, j] != 0:
                minus[i] += 1

    max_ = np.max(minus)
    minus_index = []
    if max_ > 0:
        for i in range(n):
            if minus[i] == max_:
                minus_index.append(i)

    # check whether or not the multiplication tables match
    if len(minus_index) > 0:
        for i in minus_index:
            su2_set1[i] *= -1
        for i in range(n):
            for j in range(n):
                tmp1 = np.dot(su2_set1[i], su2_set1[j])
                for k, op3 in enumerate(su2_set1):
                    if np.max(np.array(abs(tmp1 - op3))) < 1e-4:
                        multi_table1[i, j] = k
                        break
                    elif np.max(np.array(abs(tmp1 + op3))) < 1e-4:
                        multi_table1[i, j] = -k
                        break
                assert multi_table1[i, j] != 100, ('cannot find multiplication table for Bilbao SU2!', \
                                                   i, su2_set1[i], j, su2_set1[j], 'i*j', tmp1)
    assert np.max(np.abs(multi_table1 - multi_table2)) < 1e-4, 'cannot find the right SU2 matrix to time -1!'

    minus_operation = [rotP for irot, rotP in enumerate(grp_GM.rotP) if irot in minus_index]
    # use the su2 alignment made at GM to align su2 of other grp
    if len(minus_index) > 0:
        for grp in lgrps:
            for irot, rot in enumerate(grp.rotP):
                for R in minus_operation:
                    if norm(np.abs(R - rot)) < 1e-6:
                        if align_which == 'c':
                            grp.su2c[irot] *= -1  
                        else:
                            grp.su2s[irot] *= -1
                        break
    return lgrps

def get_shift(sg):
    shift = np.zeros(3)
    if sg in [48, 86, 126, 210]:
        shift = np.array([-0.25, -0.25, -0.25])
    elif sg in [70, 201]:
        shift = np.array([-0.375, -0.375, -0.375])
    elif sg in [85, 129, 130]:
        shift = np.array([0.25, -0.25, 0])
    elif sg in [50, 59, 125]:
        shift = np.array([-0.25, -0.25, 0])
    elif sg in [133, 134, 137]:
        shift = np.array([0.25, -0.25, 0.25])
    elif sg in [141, 142]:
        shift = np.array([0.5, 0.25, 0.125])
    elif sg == 68:
        shift = np.array([0, -0.25, 0.25])
    elif sg == 88:
        shift = np.array([0, 0.25, 0.125])
    elif sg == 138:
        shift = np.array([0.25, -0.25, -0.25])
    elif sg in [222, 224]:
        shift = np.array([0.25, 0.25, 0.25])
    elif sg == 227:
        shift = np.array([0.125, 0.125, 0.125])
    return shift



if __name__ == '__main__':
    gid = 61
    k_position = {'u':0, 'v':0, 'w':0}
    lgrps = loadIR(gid,k_position)
    gp = []
    for grp in lgrps:
        gp.append(grp)
        print(grp)
        # print(grp.klabel, grp.kvec_orig)
        # print(grp.degenerate_pair)
    #    for ir in grp.irreps:
    #       print(ir.label)
    #       print(ir.characters)
    #print(gp[:])




