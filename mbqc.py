# Import packages

from joblib import Parallel, delayed, cpu_count
import numpy as np
from pymatching import Matching
import pymatching as pm
from stim import DetectorErrorModel
from statsmodels.stats.proportion import proportion_confint
from lmfit import Model


Ncpu = cpu_count()
np.set_printoptions(precision = 10,suppress = True)

# Prepare some dictionary to simplify the code
# we use period boundary conditions in x,y axis so the RHG cluster state is considered as propagating toric code
# In total, we have 2d+1 layers
# odd layer have d^2 primal qubits, 2d^2 dual qubits
# even layer have d^2 dual qubits, 2d^2 primal qubits
# decoding primal lattice gives correlation surface of dual logical
# in our notation, we may call primal qubit as data and dual qubit as ancilla 

def dict_qidx_stab(d):
    qidx_stab = {}
    for i in range(d+1):
        for j in range(d):
            for k in range(d):

                idx1 = i*d**2+j*d+k
                qidx = 3*i*d**2+j*d+k
                
                idx2 = (i-1)*d**2+j*d+k
                if i == 0:
                    idx2 = {}
                elif i == d:
                    idx1 = {}
                qidx_stab[qidx] = (idx1,idx2)

    for i in range(d):
        for j in range(2*d):
            for k in range(d):
                qidx = (3*i+1)*d**2+j*d+k
                if j%2 == 0:
                    idx1 = i*d**2+int(j/2)*d+k
                    idx2 = i*d**2+((int(j/2)-1)%d)*d+k
                else:
                    idx1 = i*d**2+int((j-1)/2)*d+k                    
                    idx2 = i*d**2+int((j-1)/2)*d+(k-1)%d
                qidx_stab[qidx] = (idx1,idx2)
    return qidx_stab

def dict_anc_qidx(d):
    anc_qidx = {}
    for i in range(d+1):
        for j in range(2*d):
            for k in range(d):
                anc = 3*i*d**2+j*d+k
                if i != 0 and i != d:
                    if j%2 == 0:
                        qidx1 = ((3*((i-1)%d)+1))*d**2+j*d+k
                        qidx2 = 3*i*d**2+int(j/2)*d+k
                        qidx3 = (3*i+1)*d**2+j*d+k
                        qidx4 = 3*i*d**2+((int(j/2)-1)%d)*d+k
                    else:
                        qidx1 = 3*i*d**2+int((j-1)/2)*d+k
                        qidx2 = (3*((i-1)%d)+1)*d**2+j*d+k
                        qidx3 = 3*i*d**2+int((j-1)/2)*d+(k-1)%d
                        qidx4 = (3*i+1)*d**2+j*d+k
                elif i == 0:
                    if j%2 == 0:
                        qidx1 = {}
                        qidx2 = 3*i*d**2+int(j/2)*d+k
                        qidx3 = (3*i+1)*d**2+j*d+k
                        qidx4 = 3*i*d**2+((int(j/2)-1)%d)*d+k
                    else:
                        qidx1 = 3*i*d**2+int((j-1)/2)*d+k
                        qidx2 = {}
                        qidx3 = 3*i*d**2+int((j-1)/2)*d+(k-1)%d
                        qidx4 = (3*i+1)*d**2+j*d+k
                elif i == d:
                    if j%2 == 0:
                        qidx1 = ((3*((i-1)%d)+1))*d**2+j*d+k
                        qidx2 = 3*i*d**2+int(j/2)*d+k
                        qidx3 = {}
                        qidx4 = 3*i*d**2+((int(j/2)-1)%d)*d+k
                    else:
                        qidx1 = 3*i*d**2+int((j-1)/2)*d+k
                        qidx2 = (3*((i-1)%d)+1)*d**2+j*d+k
                        qidx3 = 3*i*d**2+int((j-1)/2)*d+(k-1)%d
                        qidx4 = {}

                anc_qidx[anc] = (qidx1,qidx2,qidx3,qidx4)

    for i in range(d):
        for j in range(d):
            for k in range(d):
                anc = (3*i+2)*d**2+j*d+k
                qidx1 = (3*i+1)*d**2+(2*j+1)*d+k
                qidx2 = (3*i+1)*d**2+2*j*d+(k-1)%d
                qidx3 = (3*i+1)*d**2+(2*((j-1)%d)+1)*d+k
                qidx4 = (3*i+1)*d**2+2*j*d+k
                anc_qidx[anc] = (qidx1,qidx2,qidx3,qidx4)
    return anc_qidx

def dict_qidx_anc(d):
    qidx_anc = {}
    for i in range(d+1):
        for j in range(d):
            for k in range(d):
                qidx = 3*i*d**2+j*d+k
                anc1 = 3*i*d**2+(2*j+1)*d+k
                anc2 = 3*i*d**2+2*j*d+k
                anc3 = 3*i*d**2+(2*j+1)*d+(k+1)%d
                anc4 = 3*i*d**2+2*((j+1)%d)*d+k
                qidx_anc[qidx] = (anc1,anc2,anc3,anc4)

    for i in range(d):
        for j in range(2*d):
            for k in range(d):
                qidx = (3*i+1)*d**2+j*d+k
                if j%2 == 0:
                    anc1 = 3*(i+1)*d**2+j*d+k
                    anc2 = (3*i+2)*d**2+int(j/2)*d+(k+1)%d
                    anc3 = (3*i)*d**2+j*d+k
                    anc4 = (3*i+2)*d**2+int(j/2)*d+k
                else:
                    anc1 = (3*i+2)*d**2+int((j-1)/2)*d+k
                    anc2 = 3*(i+1)*d**2+j*d+k
                    anc3 = (3*i+2)*d**2+(int((j+1)/2)%d)*d+k
                    anc4 = (3*i)*d**2+j*d+k
                qidx_anc[qidx] = (anc1,anc2,anc3,anc4)
    return qidx_anc


# Main body

class MBQC:

    def __init__(
            self,
            d, # code distance
            error, # error rate
            Re = 1, # ratio of erasure (leakage) error
            # We support two options: 'BE' (biased erasure conversion) & 'LD' (method in our work) 
            method = 'LD'):
        self.d = d
        self.error = error
        self.pe = error*Re
        self.pd = error*(1-Re)
        self.method = method
        self.qidx2stab = dict_qidx_stab(self.d)
        self.anc2qidx = dict_anc_qidx(self.d)
        self.qidx2anc = dict_qidx_anc(self.d)
        cor_sur_list = []
        for i in range(self.d):
            cor_sur_list.append(np.arange(0,self.d)+self.d**2*(3*i+1))
        self.cor_sur = np.concatenate(cor_sur_list)
        pass

    def rhg_era_dem_be(self):
        ran = np.random.random(3*self.d**3-self.d**2)
        era_list = np.nonzero((ran<1-(1-self.pe)**4).astype(np.uint8))
        erastr_list = []

        for era in era_list[0]:
            stab1,stab2 = self.qidx2stab[era+self.d**2][0],self.qidx2stab[era+self.d**2][1]
            if era+self.d**2 in self.cor_sur :
                erastr = 'error(0.5) D%d D%d L0' %(stab1,stab2)
            else:
                erastr = 'error(0.5) D%d D%d' %(stab1,stab2)
            erastr_list.append(erastr)

        dem = DetectorErrorModel('\n'.join(erastr_list))
        return dem
    
    def rhg_pauli_dem(self):

        # This is a predesigned decoding graph of pauli error (depolarization)
        # with two qubit depolarization error rate pd
        # there are 32/15*pd probability that primal qubit encounters single qubit Z error
        # there are 4/15*pd probability that primal qubit encounters hook Z error, two types
        # error probability of qubits on the boundary is a little different
        # hook error with 8/15*pd is effective to single qubit error

        single_str_list = []
        hook_str_list = []

        for i in range(3*self.d**3+self.d**2):
            if 3*self.d**3 > i > self.d**2-1:
                stab1,stab2 = self.qidx2stab[i][0],self.qidx2stab[i][1]

                if i < 3*self.d**2 or i > 3*self.d**3-2*self.d**2-1:
                    if i in self.cor_sur :
                        single_str = 'error(%.5f) D%d D%d L0' %(28/15*self.pd,stab1,stab2)
                    else:
                        single_str = 'error(%.5f) D%d D%d' %(28/15*self.pd,stab1,stab2)
                else:
                    if i in self.cor_sur :
                        single_str = 'error(%.5f) D%d D%d L0' %(32/15*self.pd,stab1,stab2)
                    else:
                        single_str = 'error(%.5f) D%d D%d' %(32/15*self.pd,stab1,stab2)

                single_str_list.append(single_str)

    
        for i in range(3*self.d**3):

            if 3*self.d**3-2*self.d**2 > i >2*self.d**2-1:
                data1,data2 = self.anc2qidx[i][2],self.anc2qidx[i][3]
                symdif = list(set(self.qidx2stab[data1]).symmetric_difference(set(self.qidx2stab[data2])))
                stab1,stab2 = symdif[0],symdif[1]
                if data1 in self.cor_sur  or data2 in self.cor_sur :
                    hook_str = 'error(%.5f) D%d D%d L0' %(8/15*self.pd,stab1,stab2)
                else:
                    hook_str = 'error(%.5f) D%d D%d' %(8/15*self.pd,stab1,stab2)
                hook_str_list.append(hook_str)

        str = single_str_list+hook_str_list
        dem_pauli = DetectorErrorModel('\n'.join(str))
        return dem_pauli
    
    # This is a function that samples the leakage instance in CZ gate 
    # and return results to modify the decoding graph
    
    def CZ_gate(self):
        ran = np.random.random([4,3*self.d**3-2*self.d**2])
        # 1 means LP(dual-primal),2 means PL
        # We first consider the main body, where CZ gate between 2nd layers and 2d layers is noisy

        # 4*(3d^3-2*d^2), the column is dual qubit (ancilla qubit)
        cz_dual = np.array((ran < self.pe).astype(np.uint8)+(ran < 1/2*self.pe).astype(np.uint8))

        # We find the column that has "1" that represent ancilla leakage
        anc_leakage = np.any(cz_dual == 1,axis=0).astype(np.uint8)
        # print('initial anc leakage idx',anc_leakage,np.argwhere(anc_leakage!=0).reshape(-1)) 
        anc_leakage_arg = np.argwhere(anc_leakage!=0).reshape(-1)+2*self.d**2
        # print('shifted anc leakage idx',anc_leakage_arg)
        length = anc_leakage_arg.shape[0]

        # Then we find all idx that cz gate represent 1, this length >= length above because one column could have two "1"
        find_anc_cz_qidx = np.asarray(np.nonzero(cz_dual == 1))

        # We select the column that the first leaked gate for each ancilla
        col_idx = [np.argwhere(find_anc_cz_qidx[1]==anc_leakage_arg[i]-2*self.d**2).reshape(-1)[0] for i in range(length)]
        
        anc_first_leak_cz = find_anc_cz_qidx[:,col_idx]
        anc_first_leak_cz[1] += 2*self.d**2

        # We find tailored pauli error
        tailored_pauli_list = [self.anc2qidx[anc_first_leak_cz[1][i]][anc_first_leak_cz[0][i]]\
                                for i in range(length)]
        
        # We find qidx leakage, we need to gaurantee that the CZ gate happens 'data leakage-2' 
        # while at the same time no 'ancilla leakage-1' happens before
        qidx_leak_idx = np.argwhere(cz_dual == 2)
        qidx_leakage_arg = []
        for i in range(len(qidx_leak_idx)):
            if qidx_leak_idx[i][1]+2*self.d**2 in anc_first_leak_cz[1]:
                if qidx_leak_idx[i][0] > anc_first_leak_cz[0]\
                [np.argwhere(anc_first_leak_cz[1]==qidx_leak_idx[i][1]+2*self.d**2)[0][0]]:
                    continue
            qidx_leakage_arg.append(self.anc2qidx[qidx_leak_idx[i][1]+2*self.d**2][qidx_leak_idx[i][0]])
        
        # We then consider the noisy CZ gate between 1st and 2nd layer / 2d th and 2d+1 th layer
        ran_bon = np.random.random([4*self.d**2])
        cz_dual_bon = np.array((ran_bon < self.pe).astype(np.uint8)+(ran_bon < 1/2*self.pe).astype(np.uint8))
        cz_noisy_idx = np.argwhere(cz_dual_bon!=0).reshape(-1)
        anc_leakage_arg = anc_leakage_arg.tolist()

        for i in range(len(cz_noisy_idx)):
            idx = cz_noisy_idx[i]
            if cz_dual_bon[idx] == 1:
                if idx < 2*self.d**2:
                    anc_leakage_arg.append(int(idx))
                    tailored_pauli_list.append(int(self.d**2+idx))
                else:
                    anc_leakage_arg.append(int(idx-2*self.d**2+3*self.d**3))
                    tailored_pauli_list.append(int(3*self.d**3-4*self.d**2+idx))
            if cz_dual_bon[idx] == 2:
                if idx < 2*self.d**2:
                    if int(int(idx+self.d**2)) not in qidx_leakage_arg:
                        qidx_leakage_arg.append(int(idx+self.d**2))
                else:
                    if int(int(3*self.d**3-4*self.d**2+idx)) not in qidx_leakage_arg:
                        qidx_leakage_arg.append(int(3*self.d**3-4*self.d**2+idx))

        anc_first_leak_cz_bulk = anc_first_leak_cz

        return qidx_leakage_arg,anc_leakage_arg,tailored_pauli_list,anc_first_leak_cz_bulk
    
    def rhg_era_dem_ld(self,qidx_leakage_arg,tailored_pauli_list,anc_first_leak_cz_bulk):
        era_list = list(set([*qidx_leakage_arg,*tailored_pauli_list]))
        erastr_list = []

        for era in era_list:
            stab1,stab2 = self.qidx2stab[era][0],self.qidx2stab[era][1]
            if era not in self.cor_sur :
                erastr = 'error(0.5) D%d D%d' %(stab1,stab2)
            else:
                erastr = 'error(0.5) D%d D%d L0' %(stab1,stab2)
            erastr_list.append(erastr)

        propagatestr_list = []
        length = len(anc_first_leak_cz_bulk[0])

        for i in range(length):

            if anc_first_leak_cz_bulk[0][i] == 0:
                data = self.anc2qidx[anc_first_leak_cz_bulk[1][i]][0]
                stab1,stab2 = self.qidx2stab[data][0],self.qidx2stab[data][1]
                if data not in self.cor_sur :
                    propagatestr = 'error(0.5) D%d D%d' %(stab1,stab2)
                else:
                    propagatestr = 'error(0.5) D%d D%d L0' %(stab1,stab2)
                propagatestr_list.append(propagatestr)

            if anc_first_leak_cz_bulk[0][i] == 2:
                data = self.anc2qidx[anc_first_leak_cz_bulk[1][i]][3]
                stab1,stab2 = self.qidx2stab[data][0],self.qidx2stab[data][1]
                if data not in self.cor_sur :
                    propagatestr = 'error(0.5) D%d D%d' %(stab1,stab2)
                else:
                    propagatestr = 'error(0.5) D%d D%d L0' %(stab1,stab2)
                propagatestr_list.append(propagatestr)

            if anc_first_leak_cz_bulk[0][i] == 1:
                data1,data2 = self.anc2qidx[anc_first_leak_cz_bulk[1][i]][2],\
                    self.anc2qidx[anc_first_leak_cz_bulk[1][i]][3]
                symdif = list(set(self.qidx2stab[data1]).symmetric_difference(set(self.qidx2stab[data2])))
                stab1,stab2 = symdif[0],symdif[1]
                if data1 in self.cor_sur  or data2 in self.cor_sur :
                    propagatestr = 'error(0.5) D%d D%d L0' %(stab1,stab2)
                else:
                    propagatestr = 'error(0.5) D%d D%d' %(stab1,stab2)
                propagatestr_list.append(propagatestr)

        strlist = erastr_list+propagatestr_list
        dem = DetectorErrorModel('\n'.join(strlist))
        return dem


    def replace(self,dem,dem_era,anc_leakage_arg,qidx_leakage_arg):

        m = Matching.from_detector_error_model(dem)
        length = len(anc_leakage_arg)
        m1 = Matching.from_detector_error_model(dem_era)

        for i in range(length):
            anc_idx = anc_leakage_arg[i]
            if 3*self.d**3 > anc_idx >2*self.d**2-1:
                data1 = self.anc2qidx[anc_idx][0]
                stab1,stab2 = self.qidx2stab[data1][0],self.qidx2stab[data1][1]
                if data1 not in self.cor_sur:
                    faultid = set()
                else:
                    faultid = {0}
                if m1.has_edge(stab1,stab2):
                    p = 0.5
                    w = m.get_edge_data(stab1,stab2)['weight']
                    if w == 0 and data1 not in qidx_leakage_arg:
                        m.add_edge(stab1,stab2,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='replace')
                    else:
                        m.add_edge(stab1,stab2,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='independent')
                else:
                    p = 0
                    m.add_edge(stab1,stab2,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='independent')

                data2 = self.anc2qidx[anc_idx][1]
                stab3,stab4 = self.qidx2stab[data2][0],self.qidx2stab[data2][1]
                if data2 not in self.cor_sur :
                    faultid = set()
                else:
                    faultid = {0}
                if m1.has_edge(stab3,stab4):
                    p = 0.5
                    w = m.get_edge_data(stab3,stab4)['weight']
                    if w == 0 and data2 not in qidx_leakage_arg:
                        m.add_edge(stab3,stab4,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='replace')
                    else:
                        m.add_edge(stab3,stab4,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='independent')
                else:
                    p = 0
                    m.add_edge(stab3,stab4,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='independent')
                
                data3 = self.anc2qidx[anc_idx][2]
                stab5,stab6 = self.qidx2stab[data3][0],self.qidx2stab[data3][1]
                if data3 not in self.cor_sur :
                    faultid = set()
                else:
                    faultid = {0}
                if m1.has_edge(stab5,stab6):
                    p = 0.5
                    w = m.get_edge_data(stab3,stab4)['weight']
                    if w == 0 and data3 not in qidx_leakage_arg:
                        m.add_edge(stab5,stab6,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='replace')
                    else:
                        m.add_edge(stab5,stab6,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='independent')
                else:
                    p = 0
                    m.add_edge(stab5,stab6,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='independent')
                    

                data4 = self.anc2qidx[anc_idx][3]
                stab7,stab8 = self.qidx2stab[data4][0],self.qidx2stab[data4][1]
                if data4 not in self.cor_sur :
                    faultid = set()
                else:
                    faultid = {0}
                if m1.has_edge(stab7,stab8):
                    p = 0.5
                    w = m.get_edge_data(stab7,stab8)['weight']
                    if w == 0 and data4 not in qidx_leakage_arg:
                        m.add_edge(stab7,stab8,fault_ids=faultid,weight=np.log(3),error_probability=p,merge_strategy='replace')
                    else:
                        m.add_edge(stab7,stab8,fault_ids=faultid,weight=np.log(3),error_probability=p,merge_strategy='independent')
                else:
                    p = 0
                    m.add_edge(stab7,stab8,fault_ids=faultid,weight=np.log(3),error_probability=p,merge_strategy='independent')

                symdif = list(set(self.qidx2stab[data3]).symmetric_difference(set(self.qidx2stab[data4])))
                stab9,stab10 = symdif[0],symdif[1]
                if data4 not in self.cor_sur  and data3 not in self.cor_sur :
                    faultid = set()
                else:
                    faultid = {0}
                if m1.has_edge(stab9,stab10):
                    p = 0.5
                    m.add_edge(stab9,stab10,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='replace')
                else:
                    p = 0
                    m.add_edge(stab9,stab10,fault_ids=faultid,weight=np.log(7),error_probability=p,merge_strategy='independent')
        
            
        return m
    
    def logical_error_rate(self,nshot):
        count = 0
        dem_pauli = self.rhg_pauli_dem()
        for _ in range(nshot):
            if self.method == 'LD':
                qidx_leakage_arg,anc_leakage_arg,tailored_pauli_list,anc_first_leak_cz = self.CZ_gate()
                dem_era = self.rhg_era_dem_ld(qidx_leakage_arg,tailored_pauli_list,anc_first_leak_cz)
                dem = dem_pauli+dem_era
                m = self.replace(dem,dem_era,anc_leakage_arg,qidx_leakage_arg)
            elif self.method == 'BE':
                dem_era = self.rhg_era_dem_be()
                dem = dem_pauli+dem_era
                m = Matching.from_detector_error_model(dem)
            q,s = m.add_noise()
            predition = m.decode(s)
            if np.not_equal(predition,q):
                count += 1
        return count

    def logical_parallel(self,nshot):
        # nshot should be integer times of Ncpu of the used computer
        errorlist = np.repeat(self.error,Ncpu)
        result_list = Parallel(n_jobs=-1)(delayed(self.logical_error_rate)
                                          (int(nshot/Ncpu)) for p in errorlist)
        result = np.sum(np.array(result_list))
        return (result,nshot)
    
    def logical_error_rate_mixed(self,nshot,num_leakage_sample):
        count = 0
        dem_pauli = self.rhg_pauli_dem()
        for _ in range(num_leakage_sample):
            if self.method == 'LD':
                qidx_leakage_arg,anc_leakage_arg,tailored_pauli_list,anc_first_leak_cz = self.CZ_gate()
                dem_era = self.rhg_era_dem_ld(qidx_leakage_arg,tailored_pauli_list,anc_first_leak_cz)
                dem = dem_pauli+dem_era
                m = self.replace(dem,dem_era,anc_leakage_arg,qidx_leakage_arg)
            elif self.method == 'BE':
                dem_era = self.rhg_era_dem_be()
                dem = dem_pauli+dem_era
                m = Matching.from_detector_error_model(dem)
            for _ in range(int(nshot/num_leakage_sample)):
                q,s = m.add_noise()
                predition = m.decode(s)
                if np.not_equal(predition,q):
                    count += 1
        return count

    def logical_parallel_mixed(self,nshot,num_leakage_sample):
        errorlist = np.repeat(self.error,Ncpu)
        result_list = Parallel(n_jobs=-1)(delayed(self.logical_error_rate_mixed)
                                          (int(nshot/Ncpu),num_leakage_sample) for p in errorlist)
        result = np.sum(np.array(result_list))
        return (result,nshot)
    
    # def logical_test(self):
    #     if self.method == 'BE':
    #         dem_pauli = self.rhg_pauli_dem()
    #         dem_era = self.rhg_era_dem_be()
    #         dem = dem_pauli+dem_era
    #         m = Matching.from_detector_error_model(dem)
    #         q,s = m.add_noise()
    #         predition = m.decode(s)
    #         print(m.edges())
    #         print(q)
    #         print(s)
    #         print(predition)

    #     else:
    #         return False

    
def raw_data(failure_shots,precision: int = 10):
    return round(failure_shots[0]/failure_shots[1],precision)

def get_pfail(shots, fails, alpha=0.01, confint_method="binom_test"):
    pfail_low, pfail_high = proportion_confint(
        fails, shots, alpha=alpha, method=confint_method)
    pfail = (pfail_low + pfail_high) / 2
    delta_pfail = pfail_high - pfail
    return pfail, delta_pfail

def threshold_model(p,a,b,c,alpha,pth):
    # dlist is np.array
    dlist = np.array([7,9,11,13])
    y = np.outer(dlist**alpha,(p-pth))
    pl = a*y**2+b*y+c
    return pl

def fit_threshold(errorlist,logical_errorlist):
    thres_model = Model(threshold_model)
    params = thres_model.make_params(a=10,b=1,c=0,alpha = 1,pth = 0.036)
    result = thres_model.fit(logical_errorlist, params, p = errorlist)
    return result