import numpy as np
#from functools import reduce
import tensornetwork as tn
import time
#import math,gc,itertools
#from scipy.sparse.linalg import LinearOperator
#from scipy.sparse.linalg import eigsh
#from scipy.sparse.linalg import svds
#from tqdm import tqdm
#import opt_einsum
import sys
import gc
import opt_einsum as oe

def getCG(n):
    """shape:[n+1]+[2]*n"""
    if n==0:
        return np.eye(1)
    CG=np.zeros((n+1,)+(2,)*n)
    for i in range(2**n):
        indices=tuple(map(int,bin(i)[2:].zfill(n)))
        m=np.sum(indices)
        CG[(m,)+indices]=1
    CG=np.array(list(map(lambda x:x/np.linalg.norm(x),CG.reshape(n+1,-1)))).reshape(CG.shape)
    return CG

assert getCG(3).shape==(4,2,2,2)
CGs=[getCG(i) for i in range(7)]
singlet = np.array([[0,1],[-1,0]])
assert (singlet.dot(singlet)==-np.eye(2)).all()
assert (singlet==-singlet.T).all()

# TODO cache
# TODO static



def toN(t):
    return t.cpu().detach().numpy()
def eigh(MM):
    import torch
    MM=torch.tensor(MM,device='cuda:0',dtype=torch.float64)
    s,v=torch.linalg.eigh(MM)
    s,v=toN(s),toN(v)
    return s,v

def eigh(MM):
    return np.linalg.eigh(MM)

class CachedOptimalOptimizer(oe.paths.PathOptimizer):
    class withHelper:
        def __init__(self,parent,token):
            self.parent=parent
            self.token=token
        def __enter__(self):
            self.oldToken,self.parent.token=self.parent.token,self.token
            return self
        def __exit__(self, exc_type, exc_value, traceback):
            self.parent.token=self.oldToken
            return True
        
    def __init__(self):
        self.token=None
        self.cache={}
        
    def __call__(self, inputs, output, size_dict, memory_limit=None):
        if self.token in self.cache:
            print("use cached path: "+self.token)
            #print(path)
            return self.cache[self.token]
        else:
            start_time = time.time()
            #print('start planning path')
            #o=oe.path_random.RandomGreedy(parallel=True, max_repeats=128,max_time=2)
            #o=oe.paths.auto_hq
            o=oe.paths.auto
            
            path=o(inputs,output,size_dict,memory_limit)
            #print("time used for contract path planning: ",time.time() - start_time)
            #print('# trials:',o.costs)
            if self.token is not None:
                print("cache path as: "+self.token)
                self.cache[self.token]=path
            #print(path)
            return path
    
    def cache_path(self,token):
        return self.withHelper(self,token)
    
optimizer=CachedOptimalOptimizer()

def tn_contract(nodes,output_edge_order=None,ignore_edge_order=False):
    #return tn.contractors.auto(nodes,output_edge_order=output_edge_order,memory_limit,memory_limit,ignore_edge_order)
    memory_limit=1024*1024*1024*150/4 #number of elements
    return tn.contractors.custom(nodes,optimizer,output_edge_order=output_edge_order,memory_limit=memory_limit,ignore_edge_order=ignore_edge_order)

def connect_physical_edges(physical_edges1,physical_edges2):
    """
    Parameters
    ----------
    edges1,edges2:map vertice_id->edge
    """
    for k in physical_edges1:
        if k in physical_edges2:
            physical_edges1[k]^physical_edges2[k]
            
def contract_disconnected_tensor_network(nodes,edges):
    newNodes=[]
    nodes=set(nodes)
    while len(nodes)>0:
        newSet=set(tn.reachable(next(iter(nodes))))
        newEdges=list(filter(lambda x:x.get_nodes()[0] in newSet or x.get_nodes()[1] in newSet,edges))
        #print(newSet,[[x.get_nodes()[0],x.get_nodes()[1]] for x in newEdges])
        newNode=tn_contract(newSet,newEdges)
        gc.collect(2)
        nodes=nodes.difference(newSet)
        newNodes.append(newNode)
    result=tn.outer_product_final_nodes(newNodes,edges)
    return result 

def lName(i):
    return chr(ord('a')+i) if i<26 else '['+str(i-26)+']'

def uName(i):
    return chr(ord('A')+i) if i<26 else '('+str(i-26)+')'
    

assert sys.version_info>=(3,7) #will use dict ordering feature
class AKLT_Tensor:
    """
    Attributes
    ----------
    physical_edges:sorted dict vertice_id->edge
    virtual_edges:sorted dict vertice_id->edge
    nodes:sorted dict vertice_id->node or None after SVD
    physical_dimension:int
    virtual_dimension:int
    
    """
    def __init__(self,spins,conns,div):
        """
        Parameters
        ----------
        spins:[2,2,2,2]
            the spins for the vertices of the parent graph
        conns:[[0,1],[1,2],[2,3]]
            array of pair of vertice_ids for edges in the parent graph
        div:[1,2]
            the set of vertice_ids for vertices in the subgraph
        """        
        self.nodes=dict()
        self.physical_edges=dict()
        self.virtual_edges=dict()
        
        curIndice=[1]*len(spins)
        # create nodes from cg coefficient
        for vid in div:
            axis_names=[uName(vid)+'P']+[uName(vid)+lName(i) for i in range(spins[vid])]
            self.nodes[vid]=tn.Node(CGs[spins[vid]],name=uName(vid),axis_names=axis_names)# shape: [n+1]+[2]*n
        # connect nodes to singlets on each edge
        for vid1,vid2 in conns:
            if vid1 in div and vid2 in div:
                n1=self.nodes[vid1]
                n2=tn.Node(singlet)
                n3=self.nodes[vid2]

                edges=n1.get_all_edges()
                edges[curIndice[vid1]]=n2[1]
                n1=tn.contract(n1[curIndice[vid1]]^n2[0],name=n1.name).reorder_edges(edges)
                self.nodes[vid1]=n1

                (n1[curIndice[vid1]]^n3[curIndice[vid2]])#.name=n1.name+n3.name
                curIndice[vid1]+=1
                curIndice[vid2]+=1
        # combine virtual legs of boundary nodes and log them
        for vid in div:
            n1=self.nodes[vid]
            self.physical_edges[vid]=n1[0]
            num=spins[vid]+1-curIndice[vid]
            if num==0:
                pass
            elif num==1:
                self.virtual_edges[vid]=n1[curIndice[vid]]
            else:
                n2=tn.Node(CGs[num].conj())
                edges=n1.get_all_edges()[0:curIndice[vid]]+[n2[0]]
                for i in range(num):
                    n1[curIndice[vid]+i]^n2[1+i]
                n1=tn.contract_between(n1,n2,output_edge_order=edges)
                gc.collect(2)
                self.nodes[vid]=n1
                self.virtual_edges[vid]=edges[-1]
        
        self.nodes=dict(sorted(self.nodes.items()))
        self.physical_edges=dict(sorted(self.physical_edges.items()))
        self.virtual_edges=dict(sorted(self.virtual_edges.items()))
        
        self.physical_dimensions=[e.dimension for e in self.physical_edges.values()]
        self.virtual_dimensions=[e.dimension for e in self.virtual_edges.values()]
        self.physical_dimension=np.prod(self.physical_dimensions)
        self.virtual_dimension=np.prod(self.virtual_dimensions)
        

    
    def print_dimensions(self):
        print("node shapes:",[(i,n.shape) for i,n in self.nodes.items()])
        print("physical dimensions:",self.physical_dimensions)
        print("virtual dimensions:",self.virtual_dimensions)
        print("contracted size:",self.physical_dimension,"x",self.virtual_dimension,"=",self.physical_dimension*self.virtual_dimension)
    
    def print_topology(self):
        inm={n:i for i,n in self.nodes.items()}
        toPrint=[]
        for n in self.nodes.values():
            for e in n.edges:
                if e in self.physical_edges.values():
                    assert e.node2 is None
                    toPrint.append([inm[e.node1],'p',e.dimension])
                elif e in self.virtual_edges.values():
                    assert e.node2 is None
                    toPrint.append([inm[e.node1],'v',e.dimension])
                else:
                    toPrint.append([e.node1 and inm[e.node1],e.node2 and inm[e.node2],e.dimension])
        print(toPrint)
    def copy(self):
        node_dict,edge_dict=tn.copy(self.nodes.values())
        rtval=object.__new__(self.__class__)
        
        rtval.nodes={k:node_dict[v] for k,v in self.nodes.items()}
        rtval.physical_edges={k:edge_dict[v] for k,v in self.physical_edges.items()}
        rtval.virtual_edges={k:edge_dict[v] for k,v in self.virtual_edges.items()}
        rtval.physical_dimensions=self.physical_dimensions.copy()
        rtval.virtual_dimensions=self.virtual_dimensions.copy()
        rtval.physical_dimension=self.physical_dimension
        rtval.virtual_dimension=self.virtual_dimension
        
        return rtval
    
    def remap(self, edge_id_dict):
        self.physical_edges={edge_id_dict[k]:v for k,v in self.physical_edges.items()}
        self.physical_dimensions=[e.dimension for e in self.physical_edges.values()]
    
    def get_tensor(self):
        t=self.copy()
        return contract_disconnected_tensor_network(
            list(t.nodes.values()),
            list(t.physical_edges.values())+list(t.virtual_edges.values())
        ).tensor
class AKLT_Physical_Basis:
    def __init__(self,spins,conns,div,validate=True):
        """
        Parameters
        ----------
        spins:[2,2,2,2]
            the spins for the vertices of the parent graph
        conns:[[0,1],[1,2],[2,3]]
            array of pair of vertice_ids for edges in the parent graph
        div:[1,2]
            the set of vertice_ids for vertices in the subgraph
        """
        print("AKLT_Physical_Basis start construction")
        tensor=AKLT_Tensor(spins,conns,div) # it will be discarded later, left only the nodes
        # assert tn.check_correct(tensor.nodes.values())==None wrong for non-connected graph
        print("physical_dimensions",tensor.physical_dimensions)
        print("virtual_dimensions",tensor.virtual_dimensions)
        print("virtual_dimension",tensor.virtual_dimension)
        
        t1,t2=tensor.copy(),tensor.copy()
        connect_physical_edges(t1.physical_edges,t2.physical_edges)
        print("begin contract");start_time=time.time()
        mm=contract_disconnected_tensor_network(
            list(t1.nodes.values())+list(t2.nodes.values()),
            list(t1.virtual_edges.values())+list(t2.virtual_edges.values())
        ).tensor
        print('time used: ',time.time()-start_time)
        
        print("begin SVD for ",tensor.virtual_dimension," x ",tensor.virtual_dimension);start_time=time.time()
        s2,v=eigh(mm.reshape((tensor.virtual_dimension,tensor.virtual_dimension)))
        print('time used: ',time.time()-start_time)
        s=np.sqrt(s2)
        print("maxmin singular values: ",max(s),min(s)," total ",len(s))
        vh=v.conj().T
        #assert np.allclose(v@vh,np.eye(v.shape[0]))
        #assert np.allclose(vh@v,np.eye(vh.shape[0]))
        
        #new_tensor=v@np.diag(1/s)
        new_tensor=v@np.diag(1/s)@vh
        new_node=tn.Node(new_tensor.reshape(tensor.virtual_dimensions+[new_tensor.shape[1]]))
        
        for i,e in enumerate(tensor.virtual_edges.values()):
            new_node[i]^e
                
        self.all_nodes=[new_node]+list(tensor.nodes.values())
        self.physical_edges=tensor.physical_edges.copy()
        self.virtual_edge=new_node[-1]
        self.virtual_dimension=new_tensor.shape[1]
        self.physical_dimensions=tensor.physical_dimensions.copy()
        self.physical_dimension=tensor.physical_dimension
        self.singular_values=s
        self.v_tensor=v
        
        if validate:
            print('start validate');start_time=time.time()
            self.test_unitary()
            self.test_idempotent()
            print('validate passed')
            print('time used: ',time.time()-start_time)
        print("AKLT_Physical_Basis finish construction")
        
    def print_dimensions(self):
        print("node shapes:",[n.shape for n in self.all_nodes])
        print("physical dimensions:",self.physical_dimensions)
        print("virtual dimension:",self.virtual_dimension)
        print("contracted size:",self.physical_dimension,"x",self.virtual_dimension,"=",self.physical_dimension*self.virtual_dimension)
    
    def copy(self):
        node_dict,edge_dict=tn.copy(self.all_nodes)
        rtval=object.__new__(self.__class__)
        
        rtval.all_nodes=[node_dict[v] for v in self.all_nodes]
        rtval.physical_edges={k:edge_dict[v] for k,v in self.physical_edges.items()}
        rtval.virtual_dimension=self.virtual_dimension
        rtval.virtual_edge=edge_dict[self.virtual_edge]
        rtval.physical_dimensions=self.physical_dimensions.copy()
        rtval.physical_dimension=self.physical_dimension
        rtval.singular_values=self.singular_values.copy()
        rtval.v_tensor=self.v_tensor.copy()
        
        return rtval
        
    def remap(self, new_edge_dict):
        self.physical_edges={new_edge_dict[k]:edge_dict[v] for k,v in self.physical_edges.items()}
        self.physical_dimensions=[e.dimension for e in self.physical_edges.values()]
    
    def get_tensor(self):
        t=self.copy()
        return contract_disconnected_tensor_network(
            t.all_nodes,
            list(t.physical_edges.values())+[t.virtual_edge]
        ).tensor
    
    def test_unitary(self):
        a,b=self.copy(),self.copy()
        connect_physical_edges(a.physical_edges,b.physical_edges)
        t=contract_disconnected_tensor_network(
            a.all_nodes+b.all_nodes,
            [a.virtual_edge,b.virtual_edge]
        ).tensor
        assert np.allclose(t,np.eye(self.virtual_dimension))
    def test_idempotent(self):
        def E(v):
            n0=tn.Node(v.reshape(self.physical_dimensions))
            n0_physical_edges={k:n0[i] for i,k in enumerate(self.physical_edges)}
            a,b=self.copy(),self.copy()
            connect_physical_edges(n0_physical_edges,a.physical_edges)
            a.virtual_edge^b.virtual_edge
            nodes=a.all_nodes+b.all_nodes+[n0]
            edges=list(b.physical_edges.values())
            n1=contract_disconnected_tensor_network(nodes,edges)
            return n1.tensor.flatten()
        v=np.random.random(self.physical_dimension)
        assert np.allclose(E(v),E(E(v)))
        
#a=AKLT_Physical_Basis([4,2],[[0,1]],[0,1])
#b=AKLT_Tensor([4,2],[[0,1]],[0,1])
#pd=a.physical_dimension
#vd=a.virtual_dimension
#ua=a.get_tensor().reshape(pd,-1)@a.v_tensor
#sa=np.diag(a.singular_values)
#va=a.v_tensor
#tb=b.get_tensor().reshape(pd,-1)
#assert np.allclose(ua@sa@va.T,tb)
#assert np.allclose(va.T@va,np.eye(vd))
#assert np.allclose(va@va.T,np.eye(vd))
#assert np.allclose(ua.T@ua,np.eye(vd))
#assert np.allclose(ua@ua.T,ua@ua.T@ua@ua.T)
#
#a.test_unitary()
#a.test_idempotent()



