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

# Main part of project (Toric code)

class SWAP:

    def __init__(
        self,
        d: int, # code distance
        p: float = 0.0, # total error probability
        Re: float = 1.0, # ratio of Rydberg decay
        eta: float = 0.0755, 
        # ratio of leakage & leakage in CNOT gate, generally we have eta = 0.0755
        # sometimes we set eta = 0 to test the harm of 'critical fault'
        leakage_detection: str = 'Perfect', # or 'Partial' which refers that we can only detect only one type of Rydberg decay error
        etam: float = 1.0, # ratio of detected parts of Rydberg decay
        locate_method: str = 'General',
        # Three options: 'General','Trivial','Critical'
        # 'General' means that we reweight the edges according to average probability
        # 'Trivial' means that we account the error from Rydberg decay just as Pauli error. 
        # Together with 'Perfect' in leakage_detection
        # 'Critical' means we assume the critical fault is bound to happen at that location
        # This methods leads to slightly smaller threshold but preserves the distance
        feed_forward: bool = False,
        # We consider two different circuits. 
        # One is with an additional CNOT gate and another is with feed-forward gate
        # These two circuits are equal in error propagation
        # It only determines whether we need to account error from that CNOT gate
        logicals: str = 'X1X2'
        # There are two X logical operators in toric codes.
        # However, only the longitudinal one is affected by critical error
        # So when considering 'effective error distance' we only need to consider logical error rate of X2
    ):
        self.pe = p*Re
        self.pd = p*(1-Re)
        self.d = d
        self.rounds = d 
        # We only consider d rounds of syndrome measurement
        # The readers can also test the performance for different rounds
        self.eta = eta
        self.leakage_detection = leakage_detection
        if self.leakage_detection == 'Perfect':
            self.etam = 0
        else:
            self.etam = etam
        self.locate_method = locate_method
        self.feed_forward = feed_forward
        self.logicals = logicals

    def cnot_sequence(self):
        dictx = {}
        dictz = {}
        for i in range(self.d):
            for j in range(self.d):
                stab = i*self.d+j
                dictx[stab] = ((2*i+1)*self.d+j,2*i*self.d+(j-1)%self.d,2*i*self.d+j,(2*i-1)%(2*self.d)*self.d+j)
                dictz[stab] = (2*((i+1)%self.d)*self.d+j,(2*i+1)*self.d+(j+1)%self.d,(2*i+1)*self.d+j,2*i*self.d+j)
        return dictx,dictz

    def stab_to_qidx(self):
        dictx = {}
        dictz = {}
        for i in range(self.d):
            for j in range(self.d):             
                stab = i*self.d+j
                qidxx = (2*i-1)%(2*self.d)*self.d+j
                dictx[stab] = qidxx
                qidxz = 2*i*self.d+j
                dictz[stab] = qidxz
        return dictx,dictz

    # Generate Rydberg decay error in circuit without feed-forwad gate when both type of Rydberg decay is detected
    
    def cnotgate_perfect(self,dictx,dictz,cnotsequencex,cnotsequencez):

        ran_x = np.random.random([5,self.rounds,self.d**2]) # d**2 stablizers * d rounds syndrome measurement
        ran_z = np.random.random([5,self.rounds,self.d**2])

        # 1 means LP(ancilla-data),2 means PL, 3 means LL
        cnot_ancx = np.array((ran_x < self.pe*self.eta).astype(np.uint8)+\
                             (ran_x < (1+self.eta)/2*self.pe).astype(np.uint8)+(ran_x < self.pe).astype(np.uint8))
        cnot_ancz = np.array((ran_z < self.pe*self.eta).astype(np.uint8)+\
                             (ran_z < (1+self.eta)/2*self.pe).astype(np.uint8)+(ran_z < self.pe).astype(np.uint8))


        # First we consider ancilla leakage. Such error remains as data leakage error for next period
        stabx_to_data = np.vectorize(lambda x:dictx[x],otypes=[np.uint16])
        stabz_to_data = np.vectorize(lambda x:dictz[x],otypes=[np.uint16])
        data_leakagex = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        data_leakagez = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        # region
        # Consider ancx   
        ancx_leakage = np.any(cnot_ancx%2 == 1,axis=0).astype(np.uint8)
        ancx_leakage_arg = np.argwhere(ancx_leakage!=0).reshape(-1,2)
        roundsx,leakagex = ancx_leakage_arg[:,0],ancx_leakage_arg[:,1]
        resultx = stabx_to_data(leakagex)
        data_leakagex[roundsx+1,resultx] = 1

        # Consider ancz  
        ancz_leakage = np.any(cnot_ancz%2 == 1,axis=0).astype(np.uint8)
        ancz_leakage_arg = np.argwhere(ancz_leakage!=0).reshape(-1,2)
        roundsz,leakagez = ancz_leakage_arg[:,0],ancz_leakage_arg[:,1]
        resultz = stabz_to_data(leakagez)
        data_leakagez[roundsz+1,resultz] = 1

        # Then we consider data leakage. Such error is measurement error in this period. 
        # We first consider z syndrome measurement. Data leakage comes from history or the cnot gate in this round
        # We need to rearrange cnot_ancx and cnot_ancz array so that the 3rd dimension represents data qubit that exchange with z syndrome qubit
        # Namely, the first qubit in 3rd dimension represents the data qubit that exchange with the first Z syndrome
        # In the final round, all data qubit and ancilla qubit are measured

        cnot_data_zsyndrome = np.concatenate(([cnot_ancz[0,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,0])]]
                                            ,[cnot_ancx[1,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,1])]]
                                            ,[cnot_ancx[2,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,2])]]
                                            ,[cnot_ancz[3,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,3])]]
                                            ,[cnot_ancz[4,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,3])]]))
        
        datazsyndrome_leakage = np.any(cnot_data_zsyndrome > 1,axis=0).astype(np.uint8)
        datazsyndrome_leakage_arg = np.argwhere(datazsyndrome_leakage!=0).reshape(-1,2) # leakage from cnot gate in this round
        # zsyndrome_leakage has d+1 lines, instead of d. Because we have included final measurement of ancilla qubit
        zsyndrome_leakage = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancz_leakage)) # leakage from history
        zsyndrome_leakage[datazsyndrome_leakage_arg[:,0],datazsyndrome_leakage_arg[:,1]] = 1

        # X syndrome measurement is only related to distinguish the leakage error.
        # 3rd dimension should represents the data qubit exchange with x sydnrome. for d = 3, it represents 15,16,17,3,4,5,9,10,11,
        # however, real sequence starts from qubit exchange with the second line, namely 3,4,5,9,10,11,15,16,17
        cnot_data_xsyndrome = np.concatenate(([cnot_ancx[0,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,0])]]
                                            ,[cnot_ancz[1,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,1])]]
                                            ,[cnot_ancz[2,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,2])]]
                                            ,[cnot_ancx[3,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]
                                            ,[cnot_ancx[4,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]))
        dataxsyndrome_leakage = np.any(cnot_data_xsyndrome > 1,axis=0).astype(np.uint8)
        dataxsyndrome_leakage_arg = np.argwhere(dataxsyndrome_leakage!=0).reshape(-1,2) # leakage from cnot gate in this round
        xsyndrome_leakage = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancx_leakage)) # leakage from history
        xsyndrome_leakage[dataxsyndrome_leakage_arg[:,0],(dataxsyndrome_leakage_arg[:,1]+self.d)%(self.d**2)] = 1 # we account the sequence here

        # Next we consider x error and z syndrome measurement.
        data_tailoredx = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        zsyndrome_tailoredx = np.zeros([self.rounds,self.d**2],dtype=np.uint8)

        # We first consider the first three gates in X syndrome measurement circuit. They propagate tailored x error.
        cnot_ancx3 = cnot_ancx[0:3,:]
        find_leak = np.asarray(np.nonzero(cnot_ancx3%2 == 1))
        # find_leak[0] represents which cnot gate happens leakage. find_leak[1] represents the rounds and find_leak[2] represents which x stabilizer.
        tailoredx_from_ancx3 = np.zeros([3,self.rounds,self.d**2],dtype=np.uint8)
        tailoredx_from_ancx3[find_leak[0],find_leak[1],find_leak[2]] = 1
        tailoredx_from_ancx3[2] += tailoredx_from_ancx3[1]
        
        # We consider data qubit in even lines, first we notice that the first propagated x error from x syndrome can be considered as data qubit error from last round
        data_tailoredx[0:self.rounds,np.array(list(cnotsequencex.values()))[:,0]] += tailoredx_from_ancx3[0]
        # Then whether the qubit in the next round encounters tailored x is determined by whether the data qubit is leaked between the third and fourth cnot gate in this round
        # 3rd dimension represents the data qubit exchange with x sydnrome.
        cnot_data_xsyndrome12 = np.concatenate(([cnot_ancx[0,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,0])]]
                                            ,[cnot_ancz[1,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,1])]]))
        cnot_data_xsyndrome34 = np.concatenate(([cnot_ancz[2,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,2])]]
                                            ,[cnot_ancx[3,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]))
        index_3rd_dimension = np.sort(np.array(list(cnotsequencex.values()))[:,3])
        dataxsyndrome_leakage12 = np.any(cnot_data_xsyndrome12 > 1,axis=0).astype(np.uint8) # leakage from cnot gate in this round
        dataxsyndrome_leakage34 = np.any(cnot_data_xsyndrome34 > 1,axis=0).astype(np.uint8) # leakage from cnot gate in this round
        data_tailoredx[1:self.rounds+1,index_3rd_dimension] += dataxsyndrome_leakage34
        data_tailoredx[0:self.rounds,index_3rd_dimension] += dataxsyndrome_leakage12
        data_tailoredx[1:self.rounds+1,:] += data_leakagex[1:self.rounds+1,:]

        # Consider data qubit in odd lines
        # First, if the data qubit is leaked from last round or in this round, it propagates a tailored x error to data qubit
        data_tailoredx[self.rounds,:] += data_leakagez[self.rounds,:]
        data_tailoredx[2:self.rounds+1,:] += data_leakagez[1:self.rounds,:]
        # leakage from cnot gate in this round
        data_tailoredx[1:self.rounds+1,np.array(list(cnotsequencez.values()))[:,3]] += datazsyndrome_leakage 

        # Then if propagated x error from ancilla qubit happens, the error is propagated to data qubit in the next round. This kind of error generates vertical hook error.
        # Such kind of error is considered below
        # we ignored the x error from the forth cnot gate because it comes alone with leakage error in data qubit in the next round
        vertical_hook = np.zeros([self.rounds,2*self.d**2],dtype=np.uint8)
        vertical_hook[:,np.array(list(cnotsequencex.values()))[:,1]] += tailoredx_from_ancx3[1]
        vertical_hook[:,np.array(list(cnotsequencex.values()))[:,2]] += tailoredx_from_ancx3[2]

        data_erasure = np.where(data_tailoredx != 0,1,data_tailoredx)

        #  Z syndrome measurement error.

        zsyndrome_tailoredx += np.any(cnot_ancz[0:4]%2 == 1,axis=0).astype(np.uint8)
        zsyndrome_tailoredx += np.any(cnot_ancz[0:3] == 2,axis=0).astype(np.uint8)
        zsyndrome_tailoredx[1:self.rounds,:] += ancz_leakage[0:self.rounds-1,:][:,(np.arange(0,self.d**2)+self.d)%(self.d**2)]

        syndrome_erasure= np.where(zsyndrome_tailoredx+zsyndrome_leakage[0:self.rounds] != 0,1,zsyndrome_tailoredx)

        # endregion
        return data_erasure,syndrome_erasure,vertical_hook,zsyndrome_leakage,xsyndrome_leakage

    # Generate Rydberg decay error in circuit with feed-forwad gate when both type of Rydberg decay is detected

    def cnotgate_perfect_feedforward(self,dictx,dictz,cnotsequencex,cnotsequencez):

        ran_x = np.random.random([4,self.rounds,self.d**2]) # d**2 stablizers * d rounds syndrome measurement
        ran_z = np.random.random([4,self.rounds,self.d**2])

        # 1 means LP(ancilla-data),2 means PL, 3 means LL
        cnot_ancx = np.array((ran_x < self.pe*self.eta).astype(np.uint8)+\
                             (ran_x < (1+self.eta)/2*self.pe).astype(np.uint8)+(ran_x < self.pe).astype(np.uint8))
        cnot_ancz = np.array((ran_z < self.pe*self.eta).astype(np.uint8)+\
                             (ran_z < (1+self.eta)/2*self.pe).astype(np.uint8)+(ran_z < self.pe).astype(np.uint8))
        
        cnot_ancx = np.concatenate([cnot_ancx,np.zeros([1,self.rounds,self.d**2],dtype=np.uint8)],axis=0)
        cnot_ancz = np.concatenate([cnot_ancz,np.zeros([1,self.rounds,self.d**2],dtype=np.uint8)],axis=0)


        # First we consider ancilla leakage. Such error remains as data leakage error for next period
        stabx_to_data = np.vectorize(lambda x:dictx[x],otypes=[np.uint16])
        stabz_to_data = np.vectorize(lambda x:dictz[x],otypes=[np.uint16])
        data_leakagex = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        data_leakagez = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        # region
        # Consider ancx   
        ancx_leakage = np.any(cnot_ancx%2 == 1,axis=0).astype(np.uint8)
        ancx_leakage_arg = np.argwhere(ancx_leakage!=0).reshape(-1,2)
        roundsx,leakagex = ancx_leakage_arg[:,0],ancx_leakage_arg[:,1]
        resultx = stabx_to_data(leakagex)
        data_leakagex[roundsx+1,resultx] = 1

        # Consider ancz  
        ancz_leakage = np.any(cnot_ancz%2 == 1,axis=0).astype(np.uint8)
        ancz_leakage_arg = np.argwhere(ancz_leakage!=0).reshape(-1,2)
        roundsz,leakagez = ancz_leakage_arg[:,0],ancz_leakage_arg[:,1]
        resultz = stabz_to_data(leakagez)
        data_leakagez[roundsz+1,resultz] = 1

        # Then we consider data leakage. Such error is measurement error in this period. 
        # We first consider z syndrome measurement. Data leakage comes from history or the cnot gate in this round
        # We need to rearrange cnot_ancx and cnot_ancz array so that the 3rd dimension represents data qubit that exchange with z syndrome qubit
        # Namely, the first qubit in 3rd dimension represents the data qubit that exchange with the first Z syndrome
        # In the final round, all data qubit and ancilla qubit are measured

        cnot_data_zsyndrome = np.concatenate(([cnot_ancz[0,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,0])]]
                                            ,[cnot_ancx[1,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,1])]]
                                            ,[cnot_ancx[2,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,2])]]
                                            ,[cnot_ancz[3,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,3])]]
                                            ,[cnot_ancz[4,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,3])]]))
        
        datazsyndrome_leakage = np.any(cnot_data_zsyndrome > 1,axis=0).astype(np.uint8)
        datazsyndrome_leakage_arg = np.argwhere(datazsyndrome_leakage!=0).reshape(-1,2) # leakage from cnot gate in this round
        # zsyndrome_leakage has d+1 lines, instead of d. Because we have included final measurement of ancilla qubit
        zsyndrome_leakage = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancz_leakage)) # leakage from history
        zsyndrome_leakage[datazsyndrome_leakage_arg[:,0],datazsyndrome_leakage_arg[:,1]] = 1

        # X syndrome measurement is only related to distinguish the leakage error.
        # 3rd dimension should represents the data qubit exchange with x sydnrome. for d = 3, it represents 15,16,17,3,4,5,9,10,11,
        # however, real sequence starts from qubit exchange with the second line, namely 3,4,5,9,10,11,15,16,17
        cnot_data_xsyndrome = np.concatenate(([cnot_ancx[0,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,0])]]
                                            ,[cnot_ancz[1,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,1])]]
                                            ,[cnot_ancz[2,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,2])]]
                                            ,[cnot_ancx[3,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]
                                            ,[cnot_ancx[4,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]))
        dataxsyndrome_leakage = np.any(cnot_data_xsyndrome > 1,axis=0).astype(np.uint8)
        dataxsyndrome_leakage_arg = np.argwhere(dataxsyndrome_leakage!=0).reshape(-1,2) # leakage from cnot gate in this round
        xsyndrome_leakage = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancx_leakage)) # leakage from history
        xsyndrome_leakage[dataxsyndrome_leakage_arg[:,0],(dataxsyndrome_leakage_arg[:,1]+self.d)%(self.d**2)] = 1 # we account the sequence here

        # Next we consider x error and z syndrome measurement.
        data_tailoredx = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        zsyndrome_tailoredx = np.zeros([self.rounds,self.d**2],dtype=np.uint8)

        # We first consider the first three gates in X syndrome measurement circuit. They propagate tailored x error.
        cnot_ancx3 = cnot_ancx[0:3,:]
        find_leak = np.asarray(np.nonzero(cnot_ancx3%2 == 1))
        # find_leak[0] represents which cnot gate happens leakage. find_leak[1] represents the rounds and find_leak[2] represents which x stabilizer.
        tailoredx_from_ancx3 = np.zeros([3,self.rounds,self.d**2],dtype=np.uint8)
        tailoredx_from_ancx3[find_leak[0],find_leak[1],find_leak[2]] = 1
        tailoredx_from_ancx3[2] += tailoredx_from_ancx3[1]
        
        # We consider data qubit in even lines, first we notice that the first propagated x error from x syndrome can be considered as data qubit error from last round
        data_tailoredx[0:self.rounds,np.array(list(cnotsequencex.values()))[:,0]] += tailoredx_from_ancx3[0]
        # Then whether the qubit in the next round encounters tailored x is determined by whether the data qubit is leaked between the third and fourth cnot gate in this round
        # 3rd dimension represents the data qubit exchange with x sydnrome.
        cnot_data_xsyndrome12 = np.concatenate(([cnot_ancx[0,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,0])]]
                                            ,[cnot_ancz[1,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,1])]]))
        cnot_data_xsyndrome34 = np.concatenate(([cnot_ancz[2,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,2])]]
                                            ,[cnot_ancx[3,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]))
        index_3rd_dimension = np.sort(np.array(list(cnotsequencex.values()))[:,3])
        dataxsyndrome_leakage12 = np.any(cnot_data_xsyndrome12 > 1,axis=0).astype(np.uint8) # leakage from cnot gate in this round
        dataxsyndrome_leakage34 = np.any(cnot_data_xsyndrome34 > 1,axis=0).astype(np.uint8) # leakage from cnot gate in this round
        data_tailoredx[1:self.rounds+1,index_3rd_dimension] += dataxsyndrome_leakage34
        data_tailoredx[0:self.rounds,index_3rd_dimension] += dataxsyndrome_leakage12
        data_tailoredx[1:self.rounds+1,:] += data_leakagex[1:self.rounds+1,:]

        # Consider data qubit in odd lines
        # First, if the data qubit is leaked from last round or in this round, it propagates a tailored x error to data qubit
        data_tailoredx[self.rounds,:] += data_leakagez[self.rounds,:]
        data_tailoredx[2:self.rounds+1,:] += data_leakagez[1:self.rounds,:]
        # leakage from cnot gate in this round
        data_tailoredx[1:self.rounds+1,np.array(list(cnotsequencez.values()))[:,3]] += datazsyndrome_leakage 

        # Then if propagated x error from ancilla qubit happens, the error is propagated to data qubit in the next round. This kind of error generates vertical hook error.
        # Such kind of error is considered below
        # we ignored the x error from the forth cnot gate because it comes alone with leakage error in data qubit in the next round
        vertical_hook = np.zeros([self.rounds,2*self.d**2],dtype=np.uint8)
        vertical_hook[:,np.array(list(cnotsequencex.values()))[:,1]] += tailoredx_from_ancx3[1]
        vertical_hook[:,np.array(list(cnotsequencex.values()))[:,2]] += tailoredx_from_ancx3[2]

        data_erasure = np.where(data_tailoredx != 0,1,data_tailoredx)

        #  Z syndrome measurement error.

        zsyndrome_tailoredx += np.any(cnot_ancz[0:4]%2 == 1,axis=0).astype(np.uint8)
        zsyndrome_tailoredx += np.any(cnot_ancz[0:3] == 2,axis=0).astype(np.uint8)
        zsyndrome_tailoredx[1:self.rounds,:] += ancz_leakage[0:self.rounds-1,:][:,(np.arange(0,self.d**2)+self.d)%(self.d**2)]

        syndrome_erasure= np.where(zsyndrome_tailoredx+zsyndrome_leakage[0:self.rounds] != 0,1,zsyndrome_tailoredx)

        # endregion
        return data_erasure,syndrome_erasure,vertical_hook,zsyndrome_leakage,xsyndrome_leakage

    # Generate Rydberg decay error in circuit without feed-forwad gate when we only distinguish one type of Rydberg decay

    def cnotgate(self,dictx,dictz,cnotsequencex,cnotsequencez):

        ran_x = np.random.random([5,self.rounds,self.d**2]) # d**2 stablizers * d rounds syndrome measurement
        ran_z = np.random.random([5,self.rounds,self.d**2])
        cnot_ancx = np.zeros([5,self.rounds,self.d**2], dtype=np.uint8)
        cnot_ancz = np.zeros([5,self.rounds,self.d**2], dtype=np.uint8)

        # Here we further divide the leakage instance into detected leakage and undetected leakage.
        # If the gate is single-leakage, then the leaked qubit has 50% probability to be detected leakage and 50% the other one
        # If the gate is leakage & leakage, then there must be one qubit to be detected leakage and the other one is undetetcted
        # 1 means LP(ancilla-data, undetected),2 means PL(undetected), 
        # 3 means LP(ancilla-data, detected),4 means PL(detected), 
        # 5 means LL(ancilla detected),6 means LL(data detected)

        # Note that part of the leakage is not detected. 
        # Such kind of error should be accounted by adjusting the weight of decoding graph.
        # Similar to pauli error but with different propagation.
        # We assume leakage to ground state manifold (undetected leakage) and atom loss (detected leakage) has (1-etam):etam
        # By ignoring the branch induced by leakage & leakage, 
        # we need to account pe*(1-etam) undetected leakage in decoding graph.

        # region
        # Define probabilities
        prob56 = self.eta*self.pe
        prob1234 = (1-self.eta)*self.pe    
        # Generate values based on probabilities
        mask_ancx_56 = ran_x<prob56
        mask_ancx_1234 = (ran_x>=prob56)&(ran_x<prob56+prob1234)
        mask_ancz_56 = ran_z<prob56
        mask_ancz_1234 = (ran_z>=prob56)&(ran_z<prob56+prob1234)

        # Assign values
        # Here we assume atom loss and leakage to ground state manifold has ratio 1:1 or etam
        
        cnot_ancx[mask_ancx_56] = np.random.choice([5,6],size=np.sum(mask_ancx_56))
        cnot_ancx[mask_ancx_1234] = np.random.choice([1,2,3,4], p = [(1-self.etam)/2,(1-self.etam)/2,self.etam/2,self.etam/2],
                                                    size=np.sum(mask_ancx_1234))
        cnot_ancz[mask_ancz_56] = np.random.choice([5,6],size=np.sum(mask_ancz_56))
        cnot_ancz[mask_ancz_1234] = np.random.choice([1,2,3,4], p = [(1-self.etam)/2,(1-self.etam)/2,self.etam/2,self.etam/2],
                                                    size=np.sum(mask_ancz_1234))

        # endregion
        # First we consider ancilla leakage. Such error remains as data leakage error for next period

        stabx_to_data = np.vectorize(lambda x:dictx[x],otypes=[np.uint16])
        stabz_to_data = np.vectorize(lambda x:dictz[x],otypes=[np.uint16])
        data_leakagex = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        data_leakagez = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)


        # region
        # Consider ancx   
        ancx_leakage = (np.any(cnot_ancx%2==1,axis=0)|np.any(cnot_ancx==6,axis=0)).astype(np.uint8)
        ancx_lea_detects = (np.any(cnot_ancx==3,axis=0)|np.any(cnot_ancx==5,axis=0)).astype(np.uint8)
        ancx_leakage_arg = np.argwhere(ancx_leakage!=0).reshape(-1,2)
        roundsx,leakagex = ancx_leakage_arg[:,0],ancx_leakage_arg[:,1]
        resultx = stabx_to_data(leakagex)
        data_leakagex[roundsx+1,resultx] = 1

        # Consider ancz  
        ancz_leakage = (np.any(cnot_ancz%2==1,axis=0)|np.any(cnot_ancz==6,axis=0)).astype(np.uint8)
        ancz_lea_detects = (np.any(cnot_ancz==3,axis=0)|np.any(cnot_ancz==5,axis=0)).astype(np.uint8)
        ancz_leakage_arg = np.argwhere(ancz_leakage!=0).reshape(-1,2)
        roundsz,leakagez = ancz_leakage_arg[:,0],ancz_leakage_arg[:,1]
        resultz = stabz_to_data(leakagez)
        data_leakagez[roundsz+1,resultz] = 1

        # Then we consider data leakage. Such error is measurement error in this period. 
        # We first consider z syndrome measurement. Data leakage comes from history or the cnot gate in this round
        # We need to rearrange cnot_ancx and cnot_ancz array so that the 3rd dimension represents data qubit that exchange with z syndrome qubit
        # Namely, the first qubit in 3rd dimension represents the data qubit that exchange with the first Z syndrome
        # In the final round, all data qubit and ancilla qubit are measured

        cnot_data_zsyndrome = np.concatenate(([cnot_ancz[0,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,0])]]
                                            ,[cnot_ancx[1,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,1])]]
                                            ,[cnot_ancx[2,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,2])]]
                                            ,[cnot_ancz[3,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,3])]]
                                            ,[cnot_ancz[4,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,3])]]))
        
        datazsyndrome_leakage = (np.any(cnot_data_zsyndrome>1,axis=0)&np.all(cnot_data_zsyndrome!=3,axis=0)).astype(np.uint8)
        datazsyndrome_leakage_arg = np.argwhere(datazsyndrome_leakage!=0).reshape(-1,2) # leakage from cnot gate in this round
        # zsyndrome_leakage has d+1 lines, instead of d. Because we have included final measurement of ancilla qubit
        zsyndrome_leakage = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancz_leakage)) # leakage from history
        zsyndrome_leakage[datazsyndrome_leakage_arg[:,0],datazsyndrome_leakage_arg[:,1]] = 1

        # Then we check whether the leakage is detected
        datazsyndrome_leakage_detects = (np.any(cnot_data_zsyndrome==4,axis=0)|np.any(cnot_data_zsyndrome==6,axis=0)).astype(np.uint8)
        datazsyndrome_leakage_detects_arg = np.argwhere(datazsyndrome_leakage_detects!=0).reshape(-1,2)
        zsyndrome_leakage_detects = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancz_lea_detects))
        zsyndrome_leakage_detects[datazsyndrome_leakage_detects_arg[:,0],datazsyndrome_leakage_detects_arg[:,1]] = 1

        # X syndrome measurement is only related to distinguish the leakage error.
        # 3rd dimension should represents the data qubit exchange with x sydnrome. 
        # for d = 3, it represents 15,16,17,3,4,5,9,10,11,
        # however, real sequence starts from qubit exchange with the second line, namely 3,4,5,9,10,11,15,16,17
        cnot_data_xsyndrome = np.concatenate(([cnot_ancx[0,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,0])]]
                                            ,[cnot_ancz[1,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,1])]]
                                            ,[cnot_ancz[2,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,2])]]
                                            ,[cnot_ancx[3,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]
                                            ,[cnot_ancx[4,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]))
        dataxsyndrome_leakage = (np.any(cnot_data_xsyndrome>1,axis=0)&np.all(cnot_data_xsyndrome!=3,axis=0)).astype(np.uint8)
        dataxsyndrome_leakage_arg = np.argwhere(dataxsyndrome_leakage!=0).reshape(-1,2) # leakage from cnot gate in this round
        xsyndrome_leakage = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancx_leakage)) # leakage from history
        xsyndrome_leakage[dataxsyndrome_leakage_arg[:,0],(dataxsyndrome_leakage_arg[:,1]+self.d)%(self.d**2)] = 1 # we account the sequence here

        # Then we check whether the leakage is detected
        dataxsyndrome_leakage_detects = (np.any(cnot_data_xsyndrome==4,axis=0)|np.any(cnot_data_xsyndrome==6,axis=0)).astype(np.uint8)
        dataxsyndrome_leakage_detects_arg = np.argwhere(dataxsyndrome_leakage_detects!=0).reshape(-1,2)
        xsyndrome_leakage_detects = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancx_lea_detects))
        xsyndrome_leakage_detects[dataxsyndrome_leakage_detects_arg[:,0],(dataxsyndrome_leakage_detects_arg[:,1]+self.d)%(self.d**2)] = 1

        # Next we consider x error and z syndrome measurement.
        data_tailoredx = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        zsyndrome_tailoredx = np.zeros([self.rounds,self.d**2],dtype=np.uint8)

        # We first consider the first three gates in X syndrome measurement circuit. They propagate tailored x error.
        cnot_ancx3 = cnot_ancx[0:3,:]
        find_leak = np.concatenate((np.asarray(np.nonzero(cnot_ancx3%2==1)),np.asarray(np.nonzero(cnot_ancx3==6))),axis=1).astype(np.uint8)
        # find_leak[0] represents which cnot gate happens leakage. find_leak[1] represents the rounds and find_leak[2] represents which x stabilizer.
        tailoredx_from_ancx3 = np.zeros([3,self.rounds,self.d**2],dtype=np.uint8)
        tailoredx_from_ancx3[find_leak[0],find_leak[1],find_leak[2]] = 1
        # horizontal_hook = tailoredx_from_ancx3[0]
        tailoredx_from_ancx3[2] += tailoredx_from_ancx3[1]
        
        # We consider data qubit in even lines, first we notice that the first propagated x error from x syndrome can be considered as data qubit error from last round
        data_tailoredx[0:self.rounds,np.array(list(cnotsequencex.values()))[:,0]] += tailoredx_from_ancx3[0]
        # Then whether the qubit in the next round encounters tailored x is determined by whether the data qubit is leaked between the third and fourth cnot gate in this round
        # 3rd dimension represents the data qubit exchange with x sydnrome.
        cnot_data_xsyndrome12 = np.concatenate(([cnot_ancx[0,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,0])]]
                                            ,[cnot_ancz[1,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,1])]]))
        cnot_data_xsyndrome34 = np.concatenate(([cnot_ancz[2,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,2])]]
                                            ,[cnot_ancx[3,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]))
        index_3rd_dimension = np.sort(np.array(list(cnotsequencex.values()))[:,3])
        dataxsyndrome_leakage12 = (np.any(cnot_data_xsyndrome12>1,axis=0)&np.all(cnot_data_xsyndrome12!=3,axis=0)).astype(np.uint8) # leakage from cnot gate in this round
        dataxsyndrome_leakage34 = (np.any(cnot_data_xsyndrome34>1,axis=0)&np.all(cnot_data_xsyndrome34!=3,axis=0)).astype(np.uint8) # leakage from cnot gate in this round
        data_tailoredx[1:self.rounds+1,index_3rd_dimension] += dataxsyndrome_leakage34
        data_tailoredx[0:self.rounds,index_3rd_dimension] += dataxsyndrome_leakage12
        data_tailoredx[1:self.rounds+1,:] += data_leakagex[1:self.rounds+1,:]

        # Consider data qubit in odd lines
        # First, data leakage propagates to data qubit error in the next round (50%X) error
        # One exception is the final round, the remains as a data leakage
        data_tailoredx[self.rounds,:] += data_leakagez[self.rounds,:]
        data_tailoredx[2:self.rounds+1,:] += data_leakagez[1:self.rounds,:]
        # leakage from cnot gate in this round
        data_tailoredx[1:self.rounds+1,np.array(list(cnotsequencez.values()))[:,3]] += datazsyndrome_leakage 
        

        # Then if propagated x error from ancilla qubit happens, the error is propagated to data qubit in the next round. 
        # This kind of error generates vertical hook error.
        # Such kind of error is considered below
        # we ignored the x error from the forth cnot gate because it comes alone with leakage error in data qubit in the next round
        vertical_hook = np.zeros([self.rounds,2*self.d**2],dtype=np.uint8)
        vertical_hook[:,np.array(list(cnotsequencex.values()))[:,1]] += tailoredx_from_ancx3[1]
        vertical_hook[:,np.array(list(cnotsequencex.values()))[:,2]] += tailoredx_from_ancx3[2]

        data_erasure = np.where(data_tailoredx != 0,1,data_tailoredx)

        #  Z syndrome measurement error.

        zsyndrome_tailoredx += (np.any(cnot_ancz[0:4]%2==1,axis=0)|np.any(cnot_ancz[0:4]==6,axis=0)).astype(np.uint8)
        zsyndrome_tailoredx += (np.any(cnot_ancz[0:3]==2,axis=0)|np.any(cnot_ancz[0:3]==4,axis=0)).astype(np.uint8)
        zsyndrome_tailoredx[1:self.rounds,:] += ancz_leakage[0:self.rounds-1,:][:,(np.arange(0,self.d**2)+self.d)%(self.d**2)]

        syndrome_erasure= np.where(zsyndrome_tailoredx+zsyndrome_leakage[0:self.rounds] != 0,1,zsyndrome_tailoredx)
        # endregion
        return data_erasure,syndrome_erasure,vertical_hook,zsyndrome_leakage_detects,xsyndrome_leakage_detects

    # Generate Rydberg decay error in circuit with feed-forwad gate when we only distinguish one type of Rydberg decay

    def cnotgate_feedforward(self,dictx,dictz,cnotsequencex,cnotsequencez):
        # we don't account error from the last cnot gate in each round of syndrome measurement
        ran_x = np.random.random([4,self.rounds,self.d**2]) # d**2 stablizers * d rounds syndrome measurement
        ran_z = np.random.random([4,self.rounds,self.d**2])
        cnot_ancx = np.zeros([4,self.rounds,self.d**2], dtype=np.uint8)
        cnot_ancz = np.zeros([4,self.rounds,self.d**2], dtype=np.uint8)

        # Here we further divide the leakage instance into detected leakage and undetected leakage.
        # If the gate is single-leakage, then the leaked qubit has 50% probability to be detected leakage and 50% the other one
        # If the gate is leakage & leakage, then there must be one qubit to be detected leakage and the other one is undetetcted
        # 1 means LP(ancilla-data, undetected),2 means PL(undetected), 
        # 3 means LP(ancilla-data, detected),4 means PL(detected), 
        # 5 means LL(ancilla detected),6 means LL(data detected)

        # Note that part of the leakage is not detected. 
        # Such kind of error should be accounted by adjusting the weight of decoding graph.
        # Similar to pauli error but with different propagation.
        # We assume leakage to ground state manifold (undetected leakage) and atom loss (detected leakage) has (1-etam):etam
        # By ignoring the branch induced by leakage & leakage, 
        # we need to account pe*(1-etam) undetected leakage in decoding graph.

        # region
        # Define probabilities
        prob56 = self.eta*self.pe
        prob1234 = (1-self.eta)*self.pe    
        # Generate values based on probabilities
        mask_ancx_56 = ran_x<prob56
        mask_ancx_1234 = (ran_x>=prob56)&(ran_x<prob56+prob1234)
        mask_ancz_56 = ran_z<prob56
        mask_ancz_1234 = (ran_z>=prob56)&(ran_z<prob56+prob1234)

        # Assign values
        # Here we assume atom loss and leakage to ground state manifold has ratio 1:1 or etam
        
        cnot_ancx[mask_ancx_56] = np.random.choice([5,6],size=np.sum(mask_ancx_56))
        cnot_ancx[mask_ancx_1234] = np.random.choice([1,2,3,4], p = [(1-self.etam)/2,(1-self.etam)/2,self.etam/2,self.etam/2],
                                                    size=np.sum(mask_ancx_1234))
        cnot_ancz[mask_ancz_56] = np.random.choice([5,6],size=np.sum(mask_ancz_56))
        cnot_ancz[mask_ancz_1234] = np.random.choice([1,2,3,4], p = [(1-self.etam)/2,(1-self.etam)/2,self.etam/2,self.etam/2],
                                                    size=np.sum(mask_ancz_1234))
        
        cnot_ancx = np.concatenate([cnot_ancx,np.zeros([1,self.rounds,self.d**2],dtype=np.uint8)],axis=0)
        cnot_ancz = np.concatenate([cnot_ancz,np.zeros([1,self.rounds,self.d**2],dtype=np.uint8)],axis=0)

        # endregion
        # First we consider ancilla leakage. Such error remains as data leakage error for next period

        stabx_to_data = np.vectorize(lambda x:dictx[x],otypes=[np.uint16])
        stabz_to_data = np.vectorize(lambda x:dictz[x],otypes=[np.uint16])
        data_leakagex = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        data_leakagez = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)


        # region
        # Consider ancx   
        ancx_leakage = (np.any(cnot_ancx%2==1,axis=0)|np.any(cnot_ancx==6,axis=0)).astype(np.uint8)
        ancx_lea_detects = (np.any(cnot_ancx==3,axis=0)|np.any(cnot_ancx==5,axis=0)).astype(np.uint8)
        ancx_leakage_arg = np.argwhere(ancx_leakage!=0).reshape(-1,2)
        roundsx,leakagex = ancx_leakage_arg[:,0],ancx_leakage_arg[:,1]
        resultx = stabx_to_data(leakagex)
        data_leakagex[roundsx+1,resultx] = 1

        # Consider ancz  
        ancz_leakage = (np.any(cnot_ancz%2==1,axis=0)|np.any(cnot_ancz==6,axis=0)).astype(np.uint8)
        ancz_lea_detects = (np.any(cnot_ancz==3,axis=0)|np.any(cnot_ancz==5,axis=0)).astype(np.uint8)
        ancz_leakage_arg = np.argwhere(ancz_leakage!=0).reshape(-1,2)
        roundsz,leakagez = ancz_leakage_arg[:,0],ancz_leakage_arg[:,1]
        resultz = stabz_to_data(leakagez)
        data_leakagez[roundsz+1,resultz] = 1

        # Then we consider data leakage. Such error is measurement error in this period. 
        # We first consider z syndrome measurement. Data leakage comes from history or the cnot gate in this round
        # We need to rearrange cnot_ancx and cnot_ancz array so that the 3rd dimension represents data qubit that exchange with z syndrome qubit
        # Namely, the first qubit in 3rd dimension represents the data qubit that exchange with the first Z syndrome
        # In the final round, all data qubit and ancilla qubit are measured

        cnot_data_zsyndrome = np.concatenate(([cnot_ancz[0,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,0])]]
                                            ,[cnot_ancx[1,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,1])]]
                                            ,[cnot_ancx[2,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,2])]]
                                            ,[cnot_ancz[3,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,3])]]
                                            ,[cnot_ancz[4,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,3])]]))
        
        datazsyndrome_leakage = (np.any(cnot_data_zsyndrome>1,axis=0)&np.all(cnot_data_zsyndrome!=3,axis=0)).astype(np.uint8)
        datazsyndrome_leakage_arg = np.argwhere(datazsyndrome_leakage!=0).reshape(-1,2) # leakage from cnot gate in this round
        # zsyndrome_leakage has d+1 lines, instead of d. Because we have included final measurement of ancilla qubit
        zsyndrome_leakage = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancz_leakage)) # leakage from history
        zsyndrome_leakage[datazsyndrome_leakage_arg[:,0],datazsyndrome_leakage_arg[:,1]] = 1

        # Then we check whether the leakage is detected
        datazsyndrome_leakage_detects = (np.any(cnot_data_zsyndrome==4,axis=0)|np.any(cnot_data_zsyndrome==6,axis=0)).astype(np.uint8)
        datazsyndrome_leakage_detects_arg = np.argwhere(datazsyndrome_leakage_detects!=0).reshape(-1,2)
        zsyndrome_leakage_detects = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancz_lea_detects))
        zsyndrome_leakage_detects[datazsyndrome_leakage_detects_arg[:,0],datazsyndrome_leakage_detects_arg[:,1]] = 1

        # X syndrome measurement is only related to distinguish the leakage error.
        # 3rd dimension should represents the data qubit exchange with x sydnrome. 
        # for d = 3, it represents 15,16,17,3,4,5,9,10,11,
        # however, real sequence starts from qubit exchange with the second line, namely 3,4,5,9,10,11,15,16,17
        cnot_data_xsyndrome = np.concatenate(([cnot_ancx[0,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,0])]]
                                            ,[cnot_ancz[1,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,1])]]
                                            ,[cnot_ancz[2,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,2])]]
                                            ,[cnot_ancx[3,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]
                                            ,[cnot_ancx[4,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]))
        dataxsyndrome_leakage = (np.any(cnot_data_xsyndrome>1,axis=0)&np.all(cnot_data_xsyndrome!=3,axis=0)).astype(np.uint8)
        dataxsyndrome_leakage_arg = np.argwhere(dataxsyndrome_leakage!=0).reshape(-1,2) # leakage from cnot gate in this round
        xsyndrome_leakage = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancx_leakage)) # leakage from history
        xsyndrome_leakage[dataxsyndrome_leakage_arg[:,0],(dataxsyndrome_leakage_arg[:,1]+self.d)%(self.d**2)] = 1 # we account the sequence here

        # Then we check whether the leakage is detected
        dataxsyndrome_leakage_detects = (np.any(cnot_data_xsyndrome==4,axis=0)|np.any(cnot_data_xsyndrome==6,axis=0)).astype(np.uint8)
        dataxsyndrome_leakage_detects_arg = np.argwhere(dataxsyndrome_leakage_detects!=0).reshape(-1,2)
        xsyndrome_leakage_detects = np.concatenate((np.zeros([1,self.d**2],dtype=np.uint8),ancx_lea_detects))
        xsyndrome_leakage_detects[dataxsyndrome_leakage_detects_arg[:,0],(dataxsyndrome_leakage_detects_arg[:,1]+self.d)%(self.d**2)] = 1

        # Next we consider x error and z syndrome measurement.
        data_tailoredx = np.zeros([self.rounds+1,2*self.d**2],dtype=np.uint8)
        zsyndrome_tailoredx = np.zeros([self.rounds,self.d**2],dtype=np.uint8)

        # We first consider the first three gates in X syndrome measurement circuit. They propagate tailored x error.
        cnot_ancx3 = cnot_ancx[0:3,:]
        find_leak = np.concatenate((np.asarray(np.nonzero(cnot_ancx3%2==1)),np.asarray(np.nonzero(cnot_ancx3==6))),axis=1).astype(np.uint8)
        # find_leak[0] represents which cnot gate happens leakage. find_leak[1] represents the rounds and find_leak[2] represents which x stabilizer.
        tailoredx_from_ancx3 = np.zeros([3,self.rounds,self.d**2],dtype=np.uint8)
        tailoredx_from_ancx3[find_leak[0],find_leak[1],find_leak[2]] = 1
        # horizontal_hook = tailoredx_from_ancx3[0]
        tailoredx_from_ancx3[2] += tailoredx_from_ancx3[1]
        
        # We consider data qubit in even lines, first we notice that the first propagated x error from x syndrome can be considered as data qubit error from last round
        data_tailoredx[0:self.rounds,np.array(list(cnotsequencex.values()))[:,0]] += tailoredx_from_ancx3[0]
        # Then whether the qubit in the next round encounters tailored x is determined by whether the data qubit is leaked between the third and fourth cnot gate in this round
        # 3rd dimension represents the data qubit exchange with x sydnrome.
        cnot_data_xsyndrome12 = np.concatenate(([cnot_ancx[0,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,0])]]
                                            ,[cnot_ancz[1,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,1])]]))
        cnot_data_xsyndrome34 = np.concatenate(([cnot_ancz[2,:][:,np.argsort(np.array(list(cnotsequencez.values()))[:,2])]]
                                            ,[cnot_ancx[3,:][:,np.argsort(np.array(list(cnotsequencex.values()))[:,3])]]))
        index_3rd_dimension = np.sort(np.array(list(cnotsequencex.values()))[:,3])
        dataxsyndrome_leakage12 = (np.any(cnot_data_xsyndrome12>1,axis=0)&np.all(cnot_data_xsyndrome12!=3,axis=0)).astype(np.uint8) # leakage from cnot gate in this round
        dataxsyndrome_leakage34 = (np.any(cnot_data_xsyndrome34>1,axis=0)&np.all(cnot_data_xsyndrome34!=3,axis=0)).astype(np.uint8) # leakage from cnot gate in this round
        data_tailoredx[1:self.rounds+1,index_3rd_dimension] += dataxsyndrome_leakage34
        data_tailoredx[0:self.rounds,index_3rd_dimension] += dataxsyndrome_leakage12
        data_tailoredx[1:self.rounds+1,:] += data_leakagex[1:self.rounds+1,:]

        # Consider data qubit in odd lines
        # First, data leakage propagates to data qubit error in the next round (50%X) error
        # One exception is the final round, the remains as a data leakage
        data_tailoredx[self.rounds,:] += data_leakagez[self.rounds,:]
        data_tailoredx[2:self.rounds+1,:] += data_leakagez[1:self.rounds,:]
        # leakage from cnot gate in this round
        data_tailoredx[1:self.rounds+1,np.array(list(cnotsequencez.values()))[:,3]] += datazsyndrome_leakage 
        

        # Then if propagated x error from ancilla qubit happens, the error is propagated to data qubit in the next round. 
        # This kind of error generates vertical hook error.
        # Such kind of error is considered below
        # we ignored the x error from the forth cnot gate because it comes alone with leakage error in data qubit in the next round
        vertical_hook = np.zeros([self.rounds,2*self.d**2],dtype=np.uint8)
        vertical_hook[:,np.array(list(cnotsequencex.values()))[:,1]] += tailoredx_from_ancx3[1]
        vertical_hook[:,np.array(list(cnotsequencex.values()))[:,2]] += tailoredx_from_ancx3[2]

        data_erasure = np.where(data_tailoredx != 0,1,data_tailoredx)

        #  Z syndrome measurement error.

        zsyndrome_tailoredx += (np.any(cnot_ancz[0:4]%2==1,axis=0)|np.any(cnot_ancz[0:4]==6,axis=0)).astype(np.uint8)
        zsyndrome_tailoredx += (np.any(cnot_ancz[0:3]==2,axis=0)|np.any(cnot_ancz[0:3]==4,axis=0)).astype(np.uint8)
        zsyndrome_tailoredx[1:self.rounds,:] += ancz_leakage[0:self.rounds-1,:][:,(np.arange(0,self.d**2)+self.d)%(self.d**2)]

        syndrome_erasure= np.where(zsyndrome_tailoredx+zsyndrome_leakage[0:self.rounds] != 0,1,zsyndrome_tailoredx)
        # endregion
        return data_erasure,syndrome_erasure,vertical_hook,zsyndrome_leakage_detects,xsyndrome_leakage_detects

    # Construct detector error model from error generated from Rydberg decay
    # If we consider erasure error, that is enough
    # But when we consider 'erasure-like' error, the weight should not be 0
         
    def swap_dem_era(self,daera,synera,vhook):

        rounds_daera,qidx_daera = np.nonzero(daera)
        len_daera = rounds_daera.shape[0]
        daera_str_list = []

        if self.logicals == "X1X2":
            logical_x1,logical_x2 = True,True
        elif self.logicals == "X1":
            logical_x1,logical_x2 = True,False
        elif self.logicals == "X2":
            logical_x1,logical_x2 = False,True


        for i in range(len_daera):
            rounds = rounds_daera[i]
            idx = qidx_daera[i]
            if (idx//self.d)%2 == 0:
                stab1,stab2 = rounds*self.d**2+(idx//(2*self.d))*self.d+(idx%self.d),\
                    rounds*self.d**2+(((idx//(2*self.d))-1)%self.d)*self.d+(idx%self.d)
            else:
                stab1,stab2 = rounds*self.d**2+((idx-self.d)//(2*self.d))*self.d+(idx%self.d),\
                    rounds*self.d**2+(((idx-self.d)//(2*self.d)))*self.d+(idx-1)%self.d

            if idx in np.arange(self.d,2*self.d**2+self.d,2*self.d) and logical_x1:
                daerastr = 'error(0.5) D%d D%d L0' %(stab1,stab2)
            elif idx in np.arange(0,self.d) and logical_x2:
                daerastr = 'error(0.5) D%d D%d L1' %(stab1,stab2)
            else:
                daerastr = 'error(0.5) D%d D%d' %(stab1,stab2)
            daera_str_list.append(daerastr)

        rounds_synera,zstabidx_synera = np.nonzero(synera)
        len_synera = rounds_synera.shape[0]
        synera_str_list = []

        for i in range(len_synera):
            rounds = rounds_synera[i]
            stab_idx = zstabidx_synera[i]
            stab1,stab2 = rounds*self.d**2+stab_idx,(rounds+1)*self.d**2+stab_idx
            synera_str = 'error(0.5) D%d D%d' %(stab1,stab2)
            synera_str_list.append(synera_str)
    
        rounds_vhook,qidx_vhook = np.nonzero(vhook)
        len_vhook = rounds_vhook.shape[0]
        vhook_str_list = []

        for i in range(len_vhook):
            rounds = rounds_vhook[i]
            qidx = qidx_vhook[i]
            m,n = qidx//(2*self.d),qidx%self.d
            stab1,stab2 = (rounds+1)*self.d**2+((m-1)%self.d)*self.d+n,rounds*self.d**2+m*self.d+n

            if qidx in np.arange(0,self.d) and logical_x2:
                vhookstr = 'error(0.5) D%d D%d L1' %(stab1,stab2)
            else:
                vhookstr = 'error(0.5) D%d D%d' %(stab1,stab2)

            vhook_str_list.append(vhookstr)
        strlist = daera_str_list + synera_str_list + vhook_str_list
        dem = DetectorErrorModel('\n'.join(strlist))
        return dem

    def swap_dem_pauli(self):

        if self.logicals == "X1X2":
            logical_x1,logical_x2 = True,True
        elif self.logicals == "X1":
            logical_x1,logical_x2 = True,False
        elif self.logicals == "X2":
            logical_x1,logical_x2 = False,True

        p = 4/15*self.pd
        
        # This function returns to a detector error model if 'leakage_detection' is 'Perfect'
        # returns a Matching if 'leakage_detection' is 'Partial'
        # The difference lies in whether we need to reweight the edges according to undetected leakage.
        # When using Matching, the initial graph is changed when using 'add_edge' to reweight
        # It is not convenient because we need to create the graph for multiple times
        if self.leakage_detection == 'Partial' or self.locate_method == 'Trivial':

            m = Matching()
            # Probability of undetected leakage.
            # Such error is generated but it needs additional account by weighting
            p_ul = self.pe*(1-self.etam)/4    

            for i in range(self.rounds+1):
                if i == 0:
                    peven = 4*p
                    podd = p
                    peeff = peven+3*p_ul
                    poeff = podd
                elif i == 1:
                    peven = 10*p
                    podd = 3*p
                    peeff = peven+10*p_ul
                    poeff = podd+5*p_ul
                elif i == self.rounds:
                    peven = 6*p
                    podd = 2*p
                    peeff = peven+7*p_ul
                    poeff = podd+15*p_ul
                else:
                    peven = 10*p
                    podd = 3*p
                    peeff = peven+10*p_ul
                    poeff = podd+10*p_ul
                for j in range(2*self.d):
                    if j%2 == 0:
                        for k in range(self.d):
                            stab1 = i*self.d**2+int(j/2)*self.d+k
                            stab2 = i*self.d**2+(int(j/2)-1)%self.d*self.d+k
                            if poeff != 0:
                                w = np.log((1-poeff)/poeff)
                                if j == 0 and logical_x2:
                                    m.add_edge(stab1,stab2,fault_ids={1},weight=w,error_probability=podd)
                                else:
                                    m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=podd)
                    else:
                        for k in range(self.d):
                            stab1 = i*self.d**2+int((j-1)/2)*self.d+k
                            stab2 = i*self.d**2+int((j-1)/2)*self.d+(k-1)%self.d
                            if peeff != 0:
                                w = np.log((1-peeff)/peeff)
                                if k == 0 and logical_x1:
                                    m.add_edge(stab1,stab2,fault_ids={0},weight=w,error_probability=peven)
                                else:
                                    m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=peven)

            pmeasure = 5*p
            for i in range(self.rounds):
                if i == 0:
                    pmeff = pmeasure + 12*p_ul
                else:
                    pmeff = pmeasure + 22*p_ul
                if pmeff != 0:
                    w = np.log((1-pmeff)/pmeff)
                    for j in range(self.d**2):
                        stab1,stab2 = i*self.d**2+j,(i+1)*self.d**2+j
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=pmeasure)
            
            phook1 = 2*p
            if phook1 != 0: 
                w = np.log((1-phook1)/phook1)
                for i in range(self.rounds):
                    for j in range(self.d):
                        for k in range(self.d):
                            stab1 = (i+1)*self.d**2+((j-1)%self.d)*self.d+(k-1)%self.d
                            stab2 = i*self.d**2+j*self.d+k
                            if j == 0 and k == 0 and logical_x1 and logical_x2:
                                m.add_edge(stab1,stab2,fault_ids={0,1},weight=w,error_probability=phook1)
                            elif j == 0 and logical_x2:
                                m.add_edge(stab1,stab2,fault_ids={1},weight=w,error_probability=phook1)
                            elif k == 0 and logical_x1:
                                m.add_edge(stab1,stab2,fault_ids={0},weight=w,error_probability=phook1)
                            else:
                                m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=phook1)

            phook2 = 5*p
            pehook2 = phook2+3*p_ul
            if pehook2 != 0:
                w = np.log((1-pehook2)/pehook2)
                for i in range(self.rounds):
                    for j in range(self.d):
                        for k in range(self.d):
                            stab1 = (i+1)*self.d**2+((j-1)%self.d)*self.d+(k-1)%self.d
                            stab2 = i*self.d**2+j*self.d+(k-1)%self.d 
                            if j == 0 and logical_x2:
                                m.add_edge(stab1,stab2,fault_ids={1},weight=w,error_probability=phook2)
                            else:
                                m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=phook2)

            # for i in range(rounds):
            #     phook3 = p # 2,3 equivalent
            #     for j in range(d):
            #         for k in range(d):
            #             stab1 = (i+1)*d**2+((j-1)%d)*d+k
            #             stab2 = i*d**2+j*d+k
            #             if j == 0:
            #                 str = 'error(%.8f) D%d D%d L0' %(phook3,stab1,stab2)
            #             else:
            #                 str = 'error(%.8f) D%d D%d' %(phook3,stab1,stab2)
            #             strlist.append(str)

            
            # phook4 = p 2,4 equivalent
            # w = np.log((1-phook4)/phook4)
            # for i in range(self.rounds):
            #     for j in range(self.d):
            #         for k in range(self.d):
            #             stab1 = (i+1)*self.d**2+((j+1)%self.d)*self.d+k
            #             stab2 = i*self.d**2+j*self.d+k
            #             if j == self.d and logical_x2:
            #                 m.add_edge(stab1,stab2,fault_ids={1},weight=w,error_probability=phook4)
            #             else:
            #                 m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=phook4)

            phook5 = 2*p
            if phook5 != 0:
                w = np.log((1-phook5)/phook5)
                for i in range(self.rounds):
                    for j in range(self.d):
                        for k in range(self.d):
                            stab1 = (i+1)*self.d**2+j*self.d+k
                            stab2 = i*self.d**2+j*self.d+(k+1)%self.d
                            if k == self.d-1 and logical_x1:
                                m.add_edge(stab1,stab2,fault_ids={0},weight=w,error_probability=phook5)
                            else:
                                m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=phook5)


            # for i in range(rounds):
            #     phook6 = p # 5,6 equivalent
            #     for j in range(d):
            #         for k in range(d):
            #             stab1 = i*d**2+j*d+k
            #             stab2 = (i+1)*d**2+j*d+(k-1)%d
            #             str = 'error(%.8f) D%d D%d' %(phook6,stab1,stab2)
            #             strlist.append(str)

            # for i in range(rounds):
            #     phook7 = 2*p # 3,7 equivalent
            #     for j in range(d):
            #         for k in range(d):
            #             stab1 = i*d**2+j*d+k
            #             stab2 = (i+1)*d**2+((j-1)%d)*d+k
            #             if j == 0:
            #                 str = 'error(%.8f) D%d D%d L0' %(phook7,stab1,stab2)
            #             else:
            #                 str = 'error(%.8f) D%d D%d' %(phook7,stab1,stab2)
            #             strlist.append(str)
            return m

        if self.leakage_detection == 'Perfect' and self.locate_method != 'Trivial':

            strlist = []

            for i in range(self.rounds+1):
                if i == 0:
                    peven = 4*p
                    podd = p
                elif i == self.rounds:
                    peven = 6*p
                    podd = 2*p
                else:
                    peven = 10*p
                    podd = 3*p
                for j in range(2*self.d):
                    if j%2 == 0:
                        for k in range(self.d):
                            stab1 = i*self.d**2+int(j/2)*self.d+k
                            stab2 = i*self.d**2+((int(j/2)-1)%self.d)*self.d+k
                            if j == 0 and logical_x2:
                                str = 'error(%.8f) D%d D%d L1' %(podd,stab1,stab2)
                            else:
                                str = 'error(%.8f) D%d D%d' %(podd,stab1,stab2)
                            strlist.append(str)
                    else:
                        for k in range(self.d):
                            stab1 = i*self.d**2+int((j-1)/2)*self.d+k
                            stab2 = i*self.d**2+int((j-1)/2)*self.d+(k-1)%self.d
                            if k == 0 and logical_x1:
                                str = 'error(%.8f) D%d D%d L0' %(peven,stab1,stab2)
                            else:
                                str = 'error(%.8f) D%d D%d' %(peven,stab1,stab2)
                            strlist.append(str)

            pmeasure = 5*p
            for i in range(self.rounds):
                for j in range(self.d**2):
                    stab1,stab2 = i*self.d**2+j,(i+1)*self.d**2+j
                    str = 'error(%.8f) D%d D%d' %(pmeasure,stab1,stab2)
                    strlist.append(str)
            
            phook1 = 2*p
            for i in range(self.rounds):
                for j in range(self.d):
                    for k in range(self.d):
                        stab1 = (i+1)*self.d**2+((j-1)%self.d)*self.d+(k-1)%self.d
                        stab2 = i*self.d**2+j*self.d+k
                        if j == 0 and k == 0 and logical_x1 and logical_x2:
                            str = 'error(%.8f) D%d D%d L0 L1' %(phook1,stab1,stab2)
                        elif j == 0 and logical_x2:
                            str = 'error(%.8f) D%d D%d L1' %(phook1,stab1,stab2)
                        elif k == 0 and logical_x1:
                            str = 'error(%.8f) D%d D%d L0' %(phook1,stab1,stab2)
                        else:
                            str = 'error(%.8f) D%d D%d' %(phook1,stab1,stab2)
                        strlist.append(str)

            phook2 = 5*p
            for i in range(self.rounds):
                for j in range(self.d):
                    for k in range(self.d):
                        stab1 = (i+1)*self.d**2+((j-1)%self.d)*self.d+k
                        stab2 = i*self.d**2+j*self.d+k
                        if j == 0 and logical_x2:
                            str = 'error(%.8f) D%d D%d L1' %(phook2,stab1,stab2)
                        else:
                            str = 'error(%.8f) D%d D%d' %(phook2,stab1,stab2)
                        strlist.append(str)

            phook5 = 2*p
            for i in range(self.rounds):
                for j in range(self.d):
                    for k in range(self.d):
                        stab1 = (i+1)*self.d**2+j*self.d+(k-1)%self.d
                        stab2 = i*self.d**2+j*self.d+k
                        if k == 0 and logical_x1:
                            str = 'error(%.8f) D%d D%d L0' %(phook5,stab1,stab2)
                        else:
                            str = 'error(%.8f) D%d D%d' %(phook5,stab1,stab2)
                        strlist.append(str)

            dem = DetectorErrorModel('\n'.join(strlist))

            return dem

    def swap_dem_pauli_feedforward(self):

        if self.logicals == "X1X2":
            logical_x1,logical_x2 = True,True
        elif self.logicals == "X1":
            logical_x1,logical_x2 = True,False
        elif self.logicals == "X2":
            logical_x1,logical_x2 = False,True

        p = 4/15*self.pd

        if self.leakage_detection == 'Partial' or self.locate_method == 'Trivial':
            
            m = Matching()
            p_ul = self.pe*(1-self.etam)/4    

            for i in range(self.rounds+1):
                if i == 0:
                    peven = 4*p
                    podd = p
                    peeff = peven+3*p_ul
                    poeff = podd
                elif i == 1:
                    peven = 8*p
                    podd = 2*p
                    peeff = peven+9*p_ul
                    poeff = podd+4*p_ul
                elif i == self.rounds:
                    peven = 4*p
                    podd = p
                    peeff = peven+6*p_ul
                    poeff = podd+13*p_ul
                else:
                    peven = 8*p
                    podd = 2*p
                    peeff = peven+9*p_ul
                    poeff = podd+8*p_ul
                for j in range(2*self.d):
                    if j%2 == 0:
                        for k in range(self.d):
                            stab1 = i*self.d**2+int(j/2)*self.d+k
                            stab2 = i*self.d**2+(int(j/2)-1)%self.d*self.d+k
                            if poeff != 0:
                                w = np.log((1-poeff)/poeff)
                                if j == 0 and logical_x2:
                                    m.add_edge(stab1,stab2,fault_ids={1},weight=w,error_probability=podd)
                                else:
                                    m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=podd)
                    else:
                        for k in range(self.d):
                            stab1 = i*self.d**2+int((j-1)/2)*self.d+k
                            stab2 = i*self.d**2+int((j-1)/2)*self.d+(k-1)%self.d
                            if peeff != 0:
                                w = np.log((1-peeff)/peeff)
                                if k == 0 and logical_x1:
                                    m.add_edge(stab1,stab2,fault_ids={0},weight=w,error_probability=peven)
                                else:
                                    m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=peven)
            pmeasure = 4*p
            for i in range(self.rounds):
                if i == 0:
                    pmeff = pmeasure + 11*p_ul
                else:
                    pmeff = pmeasure + 19*p_ul
                if pmeff != 0:
                    w = np.log((1-pmeff)/pmeff)
                    for j in range(self.d**2):
                        stab1,stab2 = i*self.d**2+j,(i+1)*self.d**2+j
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=pmeasure)
            
            phook1 = 2*p
            if phook1 != 0: 
                w = np.log((1-phook1)/phook1)
                for i in range(self.rounds):
                    for j in range(self.d):
                        for k in range(self.d):
                            stab1 = (i+1)*self.d**2+((j-1)%self.d)*self.d+(k-1)%self.d
                            stab2 = i*self.d**2+j*self.d+k
                            if j == 0 and k == 0 and logical_x1 and logical_x2:
                                m.add_edge(stab1,stab2,fault_ids={0,1},weight=w,error_probability=phook1)
                            elif j == 0 and logical_x2:
                                m.add_edge(stab1,stab2,fault_ids={1},weight=w,error_probability=phook1)
                            elif k == 0 and logical_x1:
                                m.add_edge(stab1,stab2,fault_ids={0},weight=w,error_probability=phook1)
                            else:
                                m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=phook1)

            phook2 = 4*p
            pehook2 = phook2+3*p_ul
            if pehook2 != 0:
                w = np.log((1-pehook2)/pehook2)
                for i in range(self.rounds):
                    for j in range(self.d):
                        for k in range(self.d):
                            stab1 = (i+1)*self.d**2+((j-1)%self.d)*self.d+(k-1)%self.d
                            stab2 = i*self.d**2+j*self.d+(k-1)%self.d 
                            if j == 0 and logical_x2:
                                m.add_edge(stab1,stab2,fault_ids={1},weight=w,error_probability=phook2)
                            else:
                                m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=phook2)

            phook5 = 2*p
            if phook5 != 0:
                w = np.log((1-phook5)/phook5)
                for i in range(self.rounds):
                    for j in range(self.d):
                        for k in range(self.d):
                            stab1 = (i+1)*self.d**2+j*self.d+k
                            stab2 = i*self.d**2+j*self.d+(k+1)%self.d
                            if k == self.d-1 and logical_x1:
                                m.add_edge(stab1,stab2,fault_ids={0},weight=w,error_probability=phook5)
                            else:
                                m.add_edge(stab1,stab2,fault_ids=set(),weight=w,error_probability=phook5)

            return m

        if self.leakage_detection == 'Perfect' and self.locate_method != 'Trivial':

            strlist = []

            for i in range(self.rounds+1):
                if i == 0:
                    peven = 4*p
                    podd = p
                elif i == self.rounds:
                    peven = 4*p
                    podd = p
                else:
                    peven = 8*p
                    podd = 2*p
                for j in range(2*self.d):
                    if j%2 == 0:
                        for k in range(self.d):
                            stab1 = i*self.d**2+int(j/2)*self.d+k
                            stab2 = i*self.d**2+(int(j/2)-1)%self.d*self.d+k
                            if j == 0 and logical_x2:
                                str = 'error(%.8f) D%d D%d L1' %(podd,stab1,stab2)
                            else:
                                str = 'error(%.8f) D%d D%d' %(podd,stab1,stab2)
                            strlist.append(str)
                    else:
                        for k in range(self.d):
                            stab1 = i*self.d**2+int((j-1)/2)*self.d+k
                            stab2 = i*self.d**2+int((j-1)/2)*self.d+(k-1)%self.d
                            if k == 0 and logical_x1:
                                str = 'error(%.8f) D%d D%d L0' %(peven,stab1,stab2)
                            else:
                                str = 'error(%.8f) D%d D%d' %(peven,stab1,stab2)
                            strlist.append(str)

            pmeasure = 4*p
            for i in range(self.rounds):
                for j in range(self.d**2):
                    stab1,stab2 = i*self.d**2+j,(i+1)*self.d**2+j
                    str = 'error(%.8f) D%d D%d' %(pmeasure,stab1,stab2)
                    strlist.append(str)
            
            phook1 = 2*p
            for i in range(self.rounds):
                for j in range(self.d):
                    for k in range(self.d):
                        stab1 = (i+1)*self.d**2+((j-1)%self.d)*self.d+(k-1)%self.d
                        stab2 = i*self.d**2+j*self.d+k
                        if j == 0 and k == 0 and logical_x1 and logical_x2:
                            str = 'error(%.8f) D%d D%d L0 L1' %(phook1,stab1,stab2)
                        elif j == 0 and logical_x2:
                            str = 'error(%.8f) D%d D%d L1' %(phook1,stab1,stab2)
                        elif k == 0 and logical_x1:
                            str = 'error(%.8f) D%d D%d L0' %(phook1,stab1,stab2)
                        else:
                            str = 'error(%.8f) D%d D%d' %(phook1,stab1,stab2)
                        strlist.append(str)

            phook2 = 4*p
            for i in range(self.rounds):
                for j in range(self.d):
                    for k in range(self.d):
                        stab1 = (i+1)*self.d**2+((j-1)%self.d)*self.d+k
                        stab2 = i*self.d**2+j*self.d+k
                        if j == 0 and logical_x2:
                            str = 'error(%.8f) D%d D%d L1' %(phook2,stab1,stab2)
                        else:
                            str = 'error(%.8f) D%d D%d' %(phook2,stab1,stab2)
                        strlist.append(str)

            phook5 = 2*p
            for i in range(self.rounds):
                for j in range(self.d):
                    for k in range(self.d):
                        stab1 = (i+1)*self.d**2+j*self.d+(k-1)%self.d
                        stab2 = i*self.d**2+j*self.d+k
                        if k == 0 and logical_x1:
                            str = 'error(%.8f) D%d D%d L0' %(phook5,stab1,stab2)
                        else:
                            str = 'error(%.8f) D%d D%d' %(phook5,stab1,stab2)
                        strlist.append(str)

            dem = DetectorErrorModel('\n'.join(strlist))
            return dem

    def pelist(self):
        if self.feed_forward == False:
            length = 10
        else:
            length = 8
        if self.pe != 0:
            pe_list = np.zeros([length])
            for i in range(length):
                pe_list[i] = self.pe*(1-self.pe)**i
            
            pe_halflist = np.zeros([int(length/2)])
            for i in range(int(length/2)):
                pe_halflist[i] = self.pe*(1-self.pe)**i
        else:
            pe_list = np.ones([length])
            pe_halflist = np.ones([int(length/2)])
        p_tot = np.sum(pe_list)
        p_tot_half = np.sum(pe_halflist)
        return pe_list/p_tot,pe_halflist/p_tot_half
    
    def replace_feedforward(self,m,dem_era,zsynlea,xsynlea,pe_list,pe_halflist):

        if type(m) is not pm.matching.Matching:
            m = Matching.from_detector_error_model(m)

        zsynlea_rounds,zsynlea_idx = np.nonzero(zsynlea)
        xsynlea_rounds,xsynlea_idx = np.nonzero(xsynlea)
        length1 = len(zsynlea_idx)
        length2 = len(xsynlea_idx)
        m1 = Matching.from_detector_error_model(dem_era)

        if self.logicals == "X1X2":
            logical_x1,logical_x2 = True,True
        elif self.logicals == "X1":
            logical_x1,logical_x2 = True,False
        elif self.logicals == "X2":
            logical_x1,logical_x2 = False,True

        if self.locate_method != 'Trivial':

            for i in range(length1):
                rounds,idx = zsynlea_rounds[i],zsynlea_idx[i]

                if idx in np.arange(0,self.d) and logical_x2:
                    faultidx2 = {1}
                else:
                    faultidx2 = set()

                if idx in np.arange(self.d,self.d*2) and logical_x2:
                    faultidx4 = {1}
                else:
                    faultidx4 = set()

                if idx in np.arange(self.d**2-self.d,self.d**2) and logical_x2:
                    faultidx3 = {1}
                else:
                    faultidx3 = set()

                if rounds == 0:
                    # M4z(2)
                    stab1,stab2 = idx,idx+self.d**2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=0,error_probability=p,merge_strategy='independent')

                    stab1,stab2 = min(idx+self.d**2,(idx-self.d)%(self.d**2)+self.d**2),max(idx+self.d**2,(idx-self.d)%(self.d**2)+self.d**2)
                    # Do(2)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=0,error_probability=p,merge_strategy='independent')

                    stab1,stab2 = (idx-self.d)%(self.d**2),(idx-self.d)%(self.d**2)+self.d**2
                    #M1z(2)
                    p_infer = pe_halflist[0]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    
                    if self.locate_method == 'Critical':
                        stab1,stab2 = min((idx-2*self.d)%(self.d**2)+self.d**2,(idx-self.d)%(self.d**2)+self.d**2),\
                            max((idx-2*self.d)%(self.d**2)+self.d**2,(idx-self.d)%(self.d**2)+self.d**2)
                        p_infer = pe_halflist[0]/2
                        # Do(2)-critical
                        if m1.has_edge(stab1,stab2):
                            p = 0.5
                            m.add_edge(stab1,stab2,fault_ids=faultidx4,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                        else:
                            p = 0
                            m.add_edge(stab1,stab2,fault_ids=faultidx4,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                
                elif rounds == self.d:

                    stab1,stab2 = self.d**2*(self.d-1)+idx,idx+self.d**3
                    # M4z(1)
                    p_infer = np.sum(pe_halflist[0:4])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                    stab1,stab2 = min(idx+self.d**3,(idx-self.d)%(self.d**2)+self.d**3),max(idx+self.d**3,(idx-self.d)%(self.d**2)+self.d**3)
                    # Do(1)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=0,error_probability=p,merge_strategy='independent')
                    
                    if self.locate_method == 'Critical':
                        stab1,stab2 = min(idx+self.d**3,(idx+self.d)%(self.d**2)+self.d**3),\
                            max(idx+self.d**3,(idx+self.d)%(self.d**2)+self.d**3)
                        p_infer = pe_halflist[0]/2
                        # Do(1) - Critical
                        if m1.has_edge(stab1,stab2):
                            p = 0.5
                            m.add_edge(stab1,stab2,fault_ids=faultidx3,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                        else:
                            p = 0
                            m.add_edge(stab1,stab2,fault_ids=faultidx3,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')        
                else:
                    # M4z(1)
                    stab1,stab2 = (rounds-1)*self.d**2+idx,rounds*self.d**2+idx
                    p_infer = np.sum(pe_list[0:4])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # M4z(2)
                    stab1,stab2 = rounds*self.d**2+idx,(rounds+1)*self.d**2+idx
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=0,error_probability=p,merge_strategy='independent')
                    # M1z(2)
                    stab1,stab2 = rounds*self.d**2+(idx-self.d)%(self.d**2),(rounds+1)*self.d**2+(idx-self.d)%(self.d**2)
                    p_infer = np.sum(pe_list[0:5])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')


                    # Do(2)
                    stab1,stab2 = min(idx+(rounds+1)*self.d**2,(idx-self.d)%(self.d**2)+(rounds+1)*self.d**2),\
                        max(idx+(rounds+1)*self.d**2,(idx-self.d)%(self.d**2)+(rounds+1)*self.d**2)
                    # p_infer = np.sum(pe_list[5:10])/2
                    p_infer = 1/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    
                    if self.locate_method == 'Critical':
                        stab1,stab2 = min((idx-2*self.d)%(self.d**2)+(rounds+1)*self.d**2,(idx-self.d)%(self.d**2)+(rounds+1)*self.d**2),\
                            max((idx-2*self.d)%(self.d**2)+(rounds+1)*self.d**2,(idx-self.d)%(self.d**2)+(rounds+1)*self.d**2)
                        p_infer = pe_list[4]/2
                        if m1.has_edge(stab1,stab2):
                            p = 0.5
                            m.add_edge(stab1,stab2,fault_ids=faultidx4,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                        else:
                            p = 0
                            m.add_edge(stab1,stab2,fault_ids=faultidx4,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                        stab1,stab2 = min(idx+rounds*self.d**2,(idx+self.d)%(self.d**2)+rounds*self.d**2),\
                            max(idx+rounds*self.d**2,(idx+self.d)%(self.d**2)+rounds*self.d**2)
                        p_infer = pe_list[0]/2
                        if m1.has_edge(stab1,stab2):
                            p = 0.5
                            m.add_edge(stab1,stab2,fault_ids=faultidx3,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                        else:
                            p = 0
                            m.add_edge(stab1,stab2,fault_ids=faultidx3,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
            
            for i in range(length2):
                rounds,idx = xsynlea_rounds[i],xsynlea_idx[i]
                s,t = idx//self.d,idx%self.d
                if s == 0 and logical_x2:
                    faultidx2 = {1}
                else:
                    faultidx2 = set()
                if t == 0 and logical_x1:
                    faultidx1 = {0}
                else:
                    faultidx1 = set()

                if rounds == 0:
                    # D4e(1)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t,((s-1)%self.d)*self.d+(t-1)%self.d),\
                        max(((s-1)%self.d)*self.d+t,((s-1)%self.d)*self.d+(t-1)%self.d)
                    p_infer = np.sum(pe_halflist[0:2])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # D4e(2)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t+self.d**2,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2),\
                        max(((s-1)%self.d)*self.d+t+self.d**2,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2)
                    p_infer = np.sum(pe_halflist[2:4])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                    stab1,stab2 = ((s-1)%self.d)*self.d+(t-1)%self.d,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2
                    p_infer = pe_halflist[1]/2
                    # M2(2)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                    stab1,stab2 = ((s-1)%self.d)*self.d+t,((s-1)%self.d)*self.d+t+self.d**2
                    p_infer = pe_halflist[2]/2
                    # M3(2)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                elif rounds == self.d:
                    # D4e(1)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t+self.d**3,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**3),\
                        max(((s-1)%self.d)*self.d+t+self.d**3,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**3)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=0,error_probability=p,merge_strategy='independent')

                    # D1e(0)
                    stab1,stab2 = min(s*self.d+t+self.d**2*(self.d-1),s*self.d+(t-1)%self.d+self.d**2*(self.d-1)),\
                        max(s*self.d+t+self.d**2*(self.d-1),s*self.d+(t-1)%self.d+self.d**2*(self.d-1))
                    p_infer = pe_halflist[0]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # V2
                    stab1,stab2 = s*self.d+(t-1)%self.d+self.d**2*(self.d-1),((s-1)%self.d)*self.d+(t-1)%self.d+self.d**3
                    p_infer = pe_halflist[1]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # V3    
                    stab1,stab2 = s*self.d+t+self.d**2*(self.d-1),((s-1)%self.d)*self.d+t+self.d**3
                    p_infer = np.sum(pe_halflist[1:3])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                else:
                    # D1e(0)
                    stab1,stab2 = min(s*self.d+t+self.d**2*(rounds-1),s*self.d+(t-1)%self.d+self.d**2*(rounds-1)),\
                        max(s*self.d+t+self.d**2*(rounds-1),s*self.d+(t-1)%self.d+self.d**2*(rounds-1))
                    p_infer = pe_list[0]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # D4e(1)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t+self.d**2*rounds,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*rounds),\
                        max(((s-1)%self.d)*self.d+t+self.d**2*rounds,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*rounds)
                    p_infer = np.sum(pe_list[0:6])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # D4e(2)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t+self.d**2*(rounds+1),((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*(rounds+1)),\
                        max(((s-1)%self.d)*self.d+t+self.d**2*(rounds+1),((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*(rounds+1))
                    p_infer = np.sum(pe_list[6:8])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')


                    # V2
                    stab1,stab2 = s*self.d+(t-1)%self.d+self.d**2*(rounds-1),((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*rounds
                    p_infer = pe_list[1]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # V3
                    stab1,stab2 = s*self.d+t+self.d**2*(rounds-1),((s-1)%self.d)*self.d+t+self.d**2*rounds
                    p_infer = np.sum(pe_list[1:3])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    #M2z
                    stab1,stab2 = ((s-1)%self.d)*self.d+(t-1)%self.d+rounds*self.d**2,\
                        ((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2+rounds*self.d**2
                    p_infer = pe_list[5]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    #M3z
                    stab1,stab2 = ((s-1)%self.d)*self.d+t+rounds*self.d**2,((s-1)%self.d)*self.d+t+self.d**2+rounds*self.d**2
                    p_infer = pe_list[6]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

        if self.locate_method == 'Trivial' or self.leakage_detection == 'Partial':
            for edge in m1.edges():
                stab1,stab2,dict = edge[0],edge[1],edge[2]
                w = m.get_edge_data(stab1,stab2)['weight']
                m.add_edge(stab1,stab2,fault_ids=dict['fault_ids'],weight=w\
                           ,error_probability=0.5,merge_strategy='replace') 
        return m

    def replace(self,m,dem_era,zsynlea,xsynlea,pe_list,pe_halflist):

        if type(m) is not pm.matching.Matching:
            m = Matching.from_detector_error_model(m)

        zsynlea_rounds,zsynlea_idx = np.nonzero(zsynlea)
        xsynlea_rounds,xsynlea_idx = np.nonzero(xsynlea)
        length1 = len(zsynlea_idx)
        length2 = len(xsynlea_idx)
        m1 = Matching.from_detector_error_model(dem_era)

        if self.logicals == "X1X2":
            logical_x1,logical_x2 = True,True
        elif self.logicals == "X1":
            logical_x1,logical_x2 = True,False
        elif self.logicals == "X2":
            logical_x1,logical_x2 = False,True

        if self.locate_method != 'Trivial':
            
            for i in range(length1):
                rounds,idx = zsynlea_rounds[i],zsynlea_idx[i]

                if idx in np.arange(0,self.d) and logical_x2:
                    faultidx2 = {1}
                else:
                    faultidx2 = set()

                if idx in np.arange(self.d,self.d*2) and logical_x2:
                    faultidx3 = {1}
                else:
                    faultidx3 = set()

                if idx in np.arange(self.d**2-self.d,self.d**2) and logical_x2:
                    faultidx4 = {1}
                else:
                    faultidx4 = set()

                if rounds == 0:
                    # M4z(2)
                    stab1,stab2 = idx,idx+self.d**2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=0,error_probability=p,merge_strategy='independent')

                    stab1,stab2 = min(idx+self.d**2,(idx-self.d)%(self.d**2)+self.d**2),max(idx+self.d**2,(idx-self.d)%(self.d**2)+self.d**2)
                    # Do(2)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=0,error_probability=p,merge_strategy='independent')

                    stab1,stab2 = (idx-self.d)%(self.d**2),(idx-self.d)%(self.d**2)+self.d**2
                    #M1z(2)
                    p_infer = pe_halflist[0]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                    if self.locate_method == 'Critical':
                        stab1,stab2 = min((idx-2*self.d)%(self.d**2)+self.d**2,(idx-self.d)%(self.d**2)+self.d**2),\
                            max((idx-2*self.d)%(self.d**2)+self.d**2,(idx-self.d)%(self.d**2)+self.d**2)
                        p_infer = pe_halflist[0]/2
                        # Do(2)-critical
                        if m1.has_edge(stab1,stab2):
                            p = 0.5
                            m.add_edge(stab1,stab2,fault_ids=faultidx4,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                        else:
                            p = 0
                            m.add_edge(stab1,stab2,fault_ids=faultidx4,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                elif rounds == self.d:

                    stab1,stab2 = self.d**2*(self.d-1)+idx,idx+self.d**3
                    # M4z(1)
                    p_infer = np.sum(pe_halflist[0:4])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                    stab1,stab2 = min(idx+self.d**3,(idx-self.d)%(self.d**2)+self.d**3),\
                        max(idx+self.d**3,(idx-self.d)%(self.d**2)+self.d**3)
                    # Do(1)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=0,error_probability=p,merge_strategy='independent')

                    if self.locate_method == 'Critical':
                        stab1,stab2 = min(idx+self.d**3,(idx+self.d)%(self.d**2)+self.d**3),\
                            max(idx+self.d**3,(idx+self.d)%(self.d**2)+self.d**3)
                        p_infer = pe_halflist[0]/2
                        # Do(1) - Critical
                        if m1.has_edge(stab1,stab2):
                            p = 0.5
                            m.add_edge(stab1,stab2,fault_ids=faultidx3,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                        else:
                            p = 0
                            m.add_edge(stab1,stab2,fault_ids=faultidx3,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

            
                else:
                    # M4z(1)
                    stab1,stab2 = (rounds-1)*self.d**2+idx,rounds*self.d**2+idx
                    p_infer = np.sum(pe_list[0:4])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # M4z(2)
                    stab1,stab2 = rounds*self.d**2+idx,(rounds+1)*self.d**2+idx
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=0,error_probability=p,merge_strategy='independent')
                    # M1z(2)
                    stab1,stab2 = rounds*self.d**2+(idx-self.d)%(self.d**2),(rounds+1)*self.d**2+(idx-self.d)%(self.d**2)
                    p_infer = np.sum(pe_list[0:6])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    
                    # Do(2)
                    stab1,stab2 = min(idx+(rounds+1)*self.d**2,(idx-self.d)%(self.d**2)+(rounds+1)*self.d**2),\
                        max(idx+(rounds+1)*self.d**2,(idx-self.d)%(self.d**2)+(rounds+1)*self.d**2)

                    p_infer = 1/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    
                    if self.locate_method == 'Critical':
                        stab1,stab2 = min((idx-2*self.d)%(self.d**2)+(rounds+1)*self.d**2,(idx-self.d)%(self.d**2)+(rounds+1)*self.d**2),\
                            max((idx-2*self.d)%(self.d**2)+(rounds+1)*self.d**2,(idx-self.d)%(self.d**2)+(rounds+1)*self.d**2)
                        p_infer = pe_list[5]/2
                        if m1.has_edge(stab1,stab2):
                            p = 0.5
                            m.add_edge(stab1,stab2,fault_ids=faultidx4,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                        else:
                            p = 0
                            m.add_edge(stab1,stab2,fault_ids=faultidx4,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                        stab1,stab2 = min(idx+rounds*self.d**2,(idx+self.d)%(self.d**2)+rounds*self.d**2),\
                            max(idx+rounds*self.d**2,(idx+self.d)%(self.d**2)+rounds*self.d**2)
                        p_infer = pe_list[0]/2
                        if m1.has_edge(stab1,stab2):
                            p = 0.5
                            m.add_edge(stab1,stab2,fault_ids=faultidx3,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                        else:
                            p = 0
                            m.add_edge(stab1,stab2,fault_ids=faultidx3,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

            for i in range(length2):
                rounds,idx = xsynlea_rounds[i],xsynlea_idx[i]
                s,t = idx//self.d,idx%self.d
                if s == 0 and logical_x2:
                    faultidx2 = {1}
                else:
                    faultidx2 = set()
                if t == 0 and logical_x1:
                    faultidx1 = {0}
                else:
                    faultidx1 = set()

                if rounds == 0:
                    # D4e(1)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t,((s-1)%self.d)*self.d+(t-1)%self.d),\
                        max(((s-1)%self.d)*self.d+t,((s-1)%self.d)*self.d+(t-1)%self.d)
                    p_infer = np.sum(pe_halflist[0:2])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # D4e(2)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t+self.d**2,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2),\
                        max(((s-1)%self.d)*self.d+t+self.d**2,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2)
                    p_infer = np.sum(pe_halflist[2:4])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                    stab1,stab2 = ((s-1)%self.d)*self.d+(t-1)%self.d,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2
                    p_infer = pe_halflist[1]/2
                    # M2(2)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                    stab1,stab2 = ((s-1)%self.d)*self.d+t,((s-1)%self.d)*self.d+t+self.d**2
                    p_infer = pe_halflist[2]/2
                    # M3(2)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                elif rounds == self.d:
                    # D4e(1)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t+self.d**3,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**3),\
                        max(((s-1)%self.d)*self.d+t+self.d**3,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**3)
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=0,error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=0,error_probability=p,merge_strategy='independent')

                    # D1e(0)
                    stab1,stab2 = min(s*self.d+t+self.d**2*(self.d-1),s*self.d+(t-1)%self.d+self.d**2*(self.d-1)),\
                        max(s*self.d+t+self.d**2*(self.d-1),s*self.d+(t-1)%self.d+self.d**2*(self.d-1))
                    p_infer = pe_halflist[0]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # V2
                    stab1,stab2 = s*self.d+(t-1)%self.d+self.d**2*(self.d-1),((s-1)%self.d)*self.d+(t-1)%self.d+self.d**3
                    p_infer = pe_halflist[1]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # V3    
                    stab1,stab2 = s*self.d+t+self.d**2*(self.d-1),((s-1)%self.d)*self.d+t+self.d**3
                    p_infer = np.sum(pe_halflist[1:3])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')

                else:
                    # D1e(0)
                    stab1,stab2 = min(s*self.d+t+self.d**2*(rounds-1),s*self.d+(t-1)%self.d+self.d**2*(rounds-1)),\
                        max(s*self.d+t+self.d**2*(rounds-1),s*self.d+(t-1)%self.d+self.d**2*(rounds-1))
                    p_infer = pe_list[0]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # D4e(1)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t+self.d**2*rounds,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*rounds),\
                        max(((s-1)%self.d)*self.d+t+self.d**2*rounds,((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*rounds)
                    p_infer = np.sum(pe_list[0:7])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # D4e(2)
                    stab1,stab2 = min(((s-1)%self.d)*self.d+t+self.d**2*(rounds+1),((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*(rounds+1)),\
                        max(((s-1)%self.d)*self.d+t+self.d**2*(rounds+1),((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*(rounds+1))
                    p_infer = np.sum(pe_list[7:9])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx1,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')


                    # V2
                    stab1,stab2 = s*self.d+(t-1)%self.d+self.d**2*(rounds-1),((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2*rounds
                    p_infer = pe_list[1]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    # V3
                    stab1,stab2 = s*self.d+t+self.d**2*(rounds-1),((s-1)%self.d)*self.d+t+self.d**2*rounds
                    p_infer = np.sum(pe_list[1:3])/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=faultidx2,weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    #M2z
                    stab1,stab2 = ((s-1)%self.d)*self.d+(t-1)%self.d+rounds*self.d**2,\
                        ((s-1)%self.d)*self.d+(t-1)%self.d+self.d**2+rounds*self.d**2
                    p_infer = pe_list[6]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    #M3z
                    stab1,stab2 = ((s-1)%self.d)*self.d+t+rounds*self.d**2,((s-1)%self.d)*self.d+t+self.d**2+rounds*self.d**2
                    p_infer = pe_list[7]/2
                    if m1.has_edge(stab1,stab2):
                        p = 0.5
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
                    else:
                        p = 0
                        m.add_edge(stab1,stab2,fault_ids=set(),weight=np.log((1-p_infer)/p_infer),error_probability=p,merge_strategy='independent')
        
        if self.locate_method == 'Trivial' or self.leakage_detection == 'Partial':
            for edge in m1.edges():
                stab1,stab2,dict = edge[0],edge[1],edge[2]
                w = m.get_edge_data(stab1,stab2)['weight']
                m.add_edge(stab1,stab2,fault_ids=dict['fault_ids'],weight=w\
                           ,error_probability=0.5,merge_strategy='replace')          
        return m
    
    def logical_error_shots(self,nshot):
        count = 0
        dictx,dictz = self.stab_to_qidx()
        cnotsex,cnotsez = self.cnot_sequence()
        pe_list,pe_halflist = self.pelist()


        if self.feed_forward == False:
            if self.leakage_detection == 'Perfect' and self.locate_method != 'Trivial':
                dem_pauli = self.swap_dem_pauli()
                m1 = Matching.from_detector_error_model(dem_pauli)
                for _ in range(nshot):
                    if self.pe != 0:
                        daera,synera,vhook,zsynlea,xsynlea = self.cnotgate_perfect(dictx,dictz,cnotsex,cnotsez)
                        dem_era = self.swap_dem_era(daera,synera,vhook)
                        m = self.replace(dem_pauli,dem_era,zsynlea,xsynlea,pe_list,pe_halflist)
                    else:
                        m = m1
                    q,s = m.add_noise()
                    predition = m.decode(s)
                    if np.any(np.not_equal(predition,q)):
                        count += 1
                return count
            else:    
                for _ in range(nshot):
                    m = self.swap_dem_pauli()
                    if self.pe != 0:
                        daera,synera,vhook,zsynlea,xsynlea = self.cnotgate(dictx,dictz,cnotsex,cnotsez)
                        dem_era = self.swap_dem_era(daera,synera,vhook)
                        m = self.replace(m,dem_era,zsynlea,xsynlea,pe_list,pe_halflist)
                    q,s = m.add_noise()
                    predition = m.decode(s)
                    if np.any(np.not_equal(predition,q)):
                        count += 1
                return count
            
        if self.feed_forward == True:
            if self.leakage_detection == 'Perfect' and self.locate_method != 'Trivial':
                dem_pauli_feedforward = self.swap_dem_pauli_feedforward()
                m1 = Matching.from_detector_error_model(dem_pauli_feedforward)
                for _ in range(nshot):
                    if self.pe != 0:
                        daera,synera,vhook,zsynlea,xsynlea = self.cnotgate_perfect_feedforward(dictx,dictz,cnotsex,cnotsez)
                        dem_era = self.swap_dem_era(daera,synera,vhook)
                        m = self.replace(dem_pauli,dem_era,zsynlea,xsynlea,pe_list,pe_halflist)
                    else:
                        m = m1
                    q,s = m.add_noise()
                    predition = m.decode(s)
                    if np.any(np.not_equal(predition,q)):
                        count += 1
                return count
            else:
                for _ in range(nshot):
                    m = self.swap_dem_pauli_feedforward()
                    if self.pe != 0:
                        daera,synera,vhook,zsynlea,xsynlea = self.cnotgate_feedforward(dictx,dictz,cnotsex,cnotsez)
                        dem_era = self.swap_dem_era(daera,synera,vhook)
                        m = self.replace(m,dem_era,zsynlea,xsynlea,pe_list,pe_halflist)
                    q,s = m.add_noise()
                    predition = m.decode(s)
                    if np.any(np.not_equal(predition,q)):
                        count += 1
                return count

    def logical_parallel(self,Nshot):
        result_list = Parallel(n_jobs=-1)(delayed(self.logical_error_shots)(int(Nshot/Ncpu)) for _ in range(Ncpu))
        failure_counts = np.sum(np.array(result_list))
        return (failure_counts,Nshot)
 
    def logical_error_shots_mixed(self,nshot1,nshot2):
        count = 0
        dictx,dictz = self.stab_to_qidx()
        cnotsex,cnotsez = self.cnot_sequence()
        pe_list,pe_halflist = self.pelist()
        if self.feed_forward == False:
            if self.leakage_detection == 'Perfect' and self.locate_method != 'Trivial':
                dem_pauli = self.swap_dem_pauli()
                m1 = Matching.from_detector_error_model(dem_pauli)
                for _ in range(nshot1):
                    if self.pe != 0:
                        daera,synera,vhook,zsynlea,xsynlea = self.cnotgate_perfect(dictx,dictz,cnotsex,cnotsez)
                        dem_era = self.swap_dem_era(daera,synera,vhook)
                        m = self.replace(dem_pauli,dem_era,zsynlea,xsynlea,pe_list,pe_halflist)
                    else:
                        m = m1
                    for _ in range(nshot2):
                        q,s = m.add_noise()
                        predition = m.decode(s)
                        if np.any(np.not_equal(predition,q)):
                            count += 1
                return count
            else:
                for _ in range(nshot1):
                    m = self.swap_dem_pauli()
                    if self.pe != 0:
                        daera,synera,vhook,zsynlea,xsynlea = self.cnotgate(dictx,dictz,cnotsex,cnotsez)
                        dem_era = self.swap_dem_era(daera,synera,vhook)
                        m = self.replace(m,dem_era,zsynlea,xsynlea,pe_list,pe_halflist)
                    for _ in range(nshot2):
                        q,s = m.add_noise()
                        predition = m.decode(s)
                        if np.any(np.not_equal(predition,q)):
                            count += 1
                return count
            
        if self.feed_forward == True:
            if self.leakage_detection == 'Perfect' and self.locate_method != 'Trivial':
                dem_pauli_feedforward = self.swap_dem_pauli_feedforward()
                m1 = Matching.from_detector_error_model(dem_pauli_feedforward)
                for _ in range(nshot1):
                    if self.pe != 0:
                        daera,synera,vhook,zsynlea,xsynlea = self.cnotgate_perfect_feedforward(dictx,dictz,cnotsex,cnotsez)
                        dem_era = self.swap_dem_era(daera,synera,vhook)
                        m = self.replace(dem_pauli,dem_era,zsynlea,xsynlea,pe_list,pe_halflist)
                    else:
                        m = m1
                    for _ in range(nshot2):
                        q,s = m.add_noise()
                        predition = m.decode(s)
                        if np.any(np.not_equal(predition,q)):
                            count += 1
                return count
            else:
                for _ in range(nshot1):
                    m = self.swap_dem_pauli_feedforward()
                    if self.pe != 0:
                        daera,synera,vhook,zsynlea,xsynlea = self.cnotgate_feedforward(dictx,dictz,cnotsex,cnotsez)
                        dem_era = self.swap_dem_era(daera,synera,vhook)
                        m = self.replace(m,dem_era,zsynlea,xsynlea,pe_list,pe_halflist)
                    for _ in range(nshot2):
                        q,s = m.add_noise()
                        predition = m.decode(s)
                        if np.any(np.not_equal(predition,q)):
                            count += 1
                return count

    def logical_parallel_mixed(self,Nshot1,Nshot):
        result_list = Parallel(n_jobs=-1)(delayed(self.logical_error_shots_mixed)(int(Nshot1/Ncpu),int(Nshot/Nshot1))\
                                           for _ in range(Ncpu))
        failure_counts = np.sum(np.array(result_list))
        return (failure_counts,Nshot)

# Some code to process data

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
    dlist = np.array([9,11,13,15])
    y = np.outer(dlist**alpha,(p-pth))
    pl = a*y**2+b*y+c
    return pl

def fit_threshold_1(errorlist,logical_errorlist):
    thres_model = Model(threshold_model)
    params = thres_model.make_params(a=10,b=1,c=0,alpha = 1,pth = 0.02)
    result = thres_model.fit(logical_errorlist, params, p = errorlist)
    return result

def fit_threshold_2(errorlist,logical_errorlist):
    thres_model = Model(threshold_model)
    params = thres_model.make_params(a=10,b=1,c=0,alpha = 1,pth = 0.007)
    result = thres_model.fit(logical_errorlist, params, p = errorlist)
    return result