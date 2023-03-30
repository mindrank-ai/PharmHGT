import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset
import dgl
from dgl.dataloading import GraphDataLoader
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  
import os


fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond

def pharm_property_types_feats(mol,factory=factory): 
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result

def GetBricsBonds(mol):
    brics_bonds = list()
    brics_bonds_rules = list()
    bonds_tmp = FindBRICSBonds(mol)
    bonds = [b for b in bonds_tmp]
    for item in bonds:# item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0][0]), int(item[0][1])])
        brics_bonds_rules.append([[int(item[0][0]), int(item[0][1])], GetBricsBondFeature([item[1][0], item[1][1]])])
        brics_bonds.append([int(item[0][1]), int(item[0][0])])
        brics_bonds_rules.append([[int(item[0][1]), int(item[0][0])], GetBricsBondFeature([item[1][1], item[1][0]])])

    result = []
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])
            
    return result, brics_bonds_rules

def GetBricsBondFeature(action):
    result = []
    start_action_bond = int(action[0]) if (action[0] !='7a' and action[0] !='7b') else 7
    end_action_bond = int(action[1]) if (action[1] !='7a' and action[1] !='7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result

def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1) # aviod index 0
    return mol

def GetFragmentFeats(mol):
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    result_ap = {}
    result_p = {}
    pharm_id = 0
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
            emb_0 = maccskeys_emb(mol_pharm)
            emb_1 = pharm_property_types_feats(mol_pharm)
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]
            
        result_p[pharm_id] = emb_0 + emb_1

        pharm_id += 1
    return result_ap, result_p

ELEMENTS = [35, 6, 7, 8, 9, 15, 16, 17, 53]
ATOM_FEATURES = {
    'atomic_num': ELEMENTS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features


def Mol2HeteroGraph(mol):
    
    # build graphs
    edge_types = [('a','b','a'),('p','r','p'),('a','j','p'), ('p','j','a')]

    edges = {k:[] for k in edge_types}
    # if mol.GetNumAtoms() == 1:
    #     g = dgl.heterograph(edges, num_nodes_dict={'a':1,'p':1})
    # else:
    result_ap, result_p = GetFragmentFeats(mol)
    reac_idx, bbr = GetBricsBonds(mol)

    for bond in mol.GetBonds(): 
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('a','b','a')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])

    for r in reac_idx:
        begin = r[1]
        end = r[2]
        edges[('p','r','p')].append([result_ap[begin],result_ap[end]])
        edges[('p','r','p')].append([result_ap[end],result_ap[begin]])

    for k,v in result_ap.items():
        edges[('a','j','p')].append([k,v])
        edges[('p','j','a')].append([v,k])

    g = dgl.heterograph(edges)
    
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom
    dim_atom = len(f_atom[0])

    f_pharm = []
    for k,v in result_p.items():
        f_pharm.append(v)
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])
    
    
    dim_atom_padding = g.nodes['a'].data['f'].size()[0]
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]

    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)
    
    # features of edges

    f_bond = []
    src,dst = g.edges(etype=('a','b','a'))
    for i in range(g.num_edges(etype=('a','b','a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('a','b','a')].data['x'] = torch.FloatTensor(f_bond)

    f_reac = []
    src, dst = g.edges(etype=('p','r','p'))
    for idx in range(g.num_edges(etype=('p','r','p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr:
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
    g.edges[('p','r','p')].data['x'] = torch.FloatTensor(f_reac)

    return g



class MolGraphSet(Dataset):
    def __init__(self,df,target,log=print):
        self.data = df
        self.mols = []
        self.labels = []
        self.graphs = []
        for i,row in df.iterrows():
            smi = row['smiles']
            label = row[target].values.astype(float)
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    log('invalid',smi)
                else:
                    g = Mol2HeteroGraph(mol)
                    if g.num_nodes('a') == 0:
                        log('no edge in graph',smi)
                    else:
                        self.mols.append(mol)
                        self.graphs.append(g)
                        self.labels.append(label)
            except Exception as e:
                log(e,'invalid',smi)
                
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self,idx):
        
        return self.graphs[idx],self.labels[idx]
    
def create_dataloader(args,filename,shuffle=True,train=True):
    dataset = MolGraphSet(pd.read_csv(os.path.join(args['path'],filename)),args['target_names'])
    if train:
        batch_size = args['batch_size']
    else:
        batch_size = min(4200,len(dataset))
    
    dataloader = GraphDataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    
    return dataloader
    

    
def random_split(load_path,save_dir,num_fold=5,sizes = [0.7,0.1,0.2],seed=0):
    df = pd.read_csv(load_path)
    n = len(df)
    os.makedirs(save_dir,exist_ok=True)
    torch.manual_seed(seed)
    for fold in range(num_fold):

        df = df.loc[torch.randperm(n)].reset_index(drop=True)
        train_size = int(sizes[0] * n)
        train_val_size = int((sizes[0] + sizes[1]) * n)
        train = df[:train_size]
        val = df[train_size:train_val_size]
        test = df[train_val_size:]
        train.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_train.csv',index=False)
        val.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_valid.csv',index=False)
        test.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_test.csv',index=False)



if __name__=='__main__':
    for seed in [2022]:
        random_split('data_index/esol.csv','data_index/esol/',seed=seed)