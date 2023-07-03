from AKLTTensor import AKLT_Physical_Basis,connect_physical_edges,tn_contract,optimizer
import time,gc,uuid
from tqdm.auto import tqdm
import tensornetwork as tn
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
import numpy as np



class ReusableTensorNetwork:
    def __init__(self,variable_tensors,constant_tensors,output_edges):
        self.variable_tensors=variable_tensors
        self.constant_tensors=constant_tensors
        self.output_edges=output_edges
    def generate_path():
        optimizer=oe.RandomGreedy()

def construct_projectors(spins,left,middle,right,connsL,connsM,connsR,connsLM,connsMR):

    left,middle,right=set(left),set(middle),set(right)
    assert all(all(i<j for i in left)for j in middle)
    assert all(all(i<j for i in middle)for j in right)
    assert len(left)+len(middle)+len(right)==max(right)+1

    UL=AKLT_Physical_Basis(spins,connsL,left,validate=False)
    UM=AKLT_Physical_Basis(spins,connsM,middle,validate=False)
    UR=AKLT_Physical_Basis(spins,connsR,right,validate=False)
    ULM=AKLT_Physical_Basis(spins,connsLM,left.union(middle),validate=False)
    UMR=AKLT_Physical_Basis(spins,connsMR,middle.union(right),validate=False)

    dimL,dimM,dimR=UL.virtual_dimension,UM.virtual_dimension,UR.virtual_dimension
    dimLMR=dimL*dimM*dimR
    
    token=str(uuid.uuid1())

    def E(v):
        n0=tn.Node(v.reshape((UL.virtual_dimension,UM.virtual_dimension,UR.virtual_dimension)))
        n1,n2=UL.copy(),UM.copy()
        n3=ULM.copy()
        n4=ULM.copy()
        n5,n6=UL.copy(),UM.copy()
        n0[0]^n1.virtual_edge
        n0[1]^n2.virtual_edge
        connect_physical_edges(n1.physical_edges,n3.physical_edges)
        connect_physical_edges(n2.physical_edges,n3.physical_edges)
        n3.virtual_edge^n4.virtual_edge
        connect_physical_edges(n5.physical_edges,n4.physical_edges)
        connect_physical_edges(n6.physical_edges,n4.physical_edges)

        nodes=[n0]+[n for N in [n1,n2,n3,n4,n5,n6] for n in N.all_nodes]
        edges=[n5.virtual_edge,n6.virtual_edge,n0[2]]
        
        #with optimizer.cache_path(token+'_E'):
        n7=tn_contract(nodes,edges)
        rtval=n7.tensor.reshape(-1)
        return rtval

    def F(v):
        n0=tn.Node(v.reshape((UL.virtual_dimension,UM.virtual_dimension,UR.virtual_dimension)))
        n1,n2=UM.copy(),UR.copy()
        n3=UMR.copy()
        n4=UMR.copy()
        n5,n6=UM.copy(),UR.copy()
        n0[1]^n1.virtual_edge
        n0[2]^n2.virtual_edge
        connect_physical_edges(n1.physical_edges,n3.physical_edges)
        connect_physical_edges(n2.physical_edges,n3.physical_edges)
        n3.virtual_edge^n4.virtual_edge
        connect_physical_edges(n5.physical_edges,n4.physical_edges)
        connect_physical_edges(n6.physical_edges,n4.physical_edges)

        nodes=[n0]+[n for N in [n1,n2,n3,n4,n5,n6] for n in N.all_nodes]
        edges=[n0[0],n5.virtual_edge,n6.virtual_edge]

        #with optimizer.cache_path(token+'_F'):
        n7=tn_contract(nodes,edges)
        rtval=n7.tensor.reshape(-1)
        return rtval

    print('dimL*dimM*dimR',dimL*dimM*dimR)
    print("begin apply EF");start_time = time.time()
    v=np.random.random(dimLMR)
    Ev=E(v)
    Fv=F(v)
    assert np.allclose(E(Ev),Ev)
    assert np.allclose(F(Fv),Fv)
    del v
    print("time used: ",time.time() - start_time)
    print('done')
    
    return E,F,dimLMR




#1-z eta>0 => gapped
#eta=sup(noninteger a)
#eta0= estimated upper bound of eta
#if result is near to 2+2 eta0 then reduce eta
#if result is larger than 2+2 eta0 then for checking double root, increase eta

def calc_eta(E,F,dimLMR,eta0=0.1,tol=0.01):
    
    threshold=2*(1+eta0)


    pbar=tqdm()
    oldeigs=[0]
    def OP(v):
        Ev=E(v)
        gc.collect(2)
        Fv=F(v)
        gc.collect(2)
        EFv=E(Fv)
        gc.collect(2)
        FEv=F(Ev)
        gc.collect(2)
        rtval=(2+eta0)*(Ev+Fv)-(EFv+FEv)
        pbar.update(1)
        del Ev,Fv,EFv,FEv
        gc.collect(2)
        eig=np.linalg.norm(rtval)/np.linalg.norm(v)
        pbar.set_postfix_str(str(oldeigs[-1])+'>'+str(eig))
        oldeigs.append(eig)
        return rtval
    op=LinearOperator(shape=(dimLMR,dimLMR),matvec=OP)
    
    # Avoid Overflow bug. see: https://github.com/scipy/scipy/issues/11261
    maxiter=min(dimLMR*10,np.iinfo(np.int32).max)
    eigenvalues, eigenvectors = eigsh(op, k=1,tol=tol,maxiter=maxiter)
    pbar.close()

    e=eigenvalues[0]
    if e<=threshold:
        print(e,'<=',threshold,'eta cannot be extracted, try increase eta0')
    else:
        print(e,'>',threshold,'eta can be extracted, please check which root is correct by increase eta0 (but eta0 < smaller a)')
        print("possible a:")
        print(-0.5*(-1-eta0+ np.sqrt(9 - 4 *e + 6 *eta0 + eta0**2)))
        print(-0.5*(-1-eta0- np.sqrt(9 - 4 *e + 6 *eta0 + eta0**2)))

    smallRoot=-0.5*(-1-eta0+ np.sqrt(9 - 4 *e + 6 *eta0 + eta0**2))
    print("eta0 ",eta0,"tol ",tol,"eta ",smallRoot)