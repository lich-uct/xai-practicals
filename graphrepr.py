import numpy as np
import torch
from torch_geometric.data import Data
from rdkit.Chem import rdMolDescriptors



def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)

    
def get_atom_features(atom, neighbours: bool = False,
                      total_num_hs: bool = False, formal_charge: bool = False,
                      is_in_ring=False, is_aromatic: bool = False):

    """
    Calculate feature vector for atom.
    :param atom: atom to featurise
    :param neighbours: bool: use number of neighbours as a feature?
    :param total_num_hs: bool: use total number of Hs as a feature?
    :param formal_charge: bool: use formal charage as a feature?
    :param is_in_ring: bool: use ringness as a feature?
    :param is_aromatic: bool: use aromaticity as a feature?
    :return: np.array of attributes - a vector representation of atom
    """

    # strict type checking
    for param in [neighbours, total_num_hs, formal_charge, is_in_ring, is_aromatic]:
        assert isinstance(param, bool), f"Param should be bool, is {type(param)} with value {param}."

    attributes = []
    attributes += one_hot_vector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    if neighbours:
        attributes += one_hot_vector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    if total_num_hs:
        attributes += one_hot_vector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if formal_charge:
        attributes.append(atom.GetFormalCharge())
    if is_in_ring:
        attributes.append(atom.IsInRing())
    if is_aromatic:
        attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def make_list(length, prefix):
  return [f'{prefix} {i}' for i in range(length)]

feature_meaning = ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other'] + make_list(6, 'N') + make_list(5, 'H') + ['charge', 'ringness', 'aroma']
feature_meaning = np.array(feature_meaning)


def featurise_data(dataset, node_level, device=None):
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
  print(f"Using {device}")

  data_list = []
  for index, row in dataset.iterrows():
    mol = row.ROMol
    # feature matrix
    node_features = np.array(
      [get_atom_features(atom, neighbours=True, total_num_hs=True, formal_charge=True, is_in_ring=True, is_aromatic=True)
      for atom in mol.GetAtoms()
      ])

    # adjacency matrix
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    if len(edges)==0:
      continue
    edges = np.array(edges)

    # labels
    if node_level:
      contribs = rdMolDescriptors._CalcCrippenContribs(mol)  # [(logp, mr), (...), ...]
      y = np.array(contribs)[:,0]
      y = y.reshape(-1, 1)
      raw_y = y
      y = torch.FloatTensor(y)
    else:
      threshold = -3
      raw_y = row['measured log solubility in mols per litre']
      y = np.array(raw_y>threshold)  # ta czy ta druga rozpuszczalność?
      y = torch.LongTensor(y)

    data_list.append(Data(x=torch.Tensor(node_features).to(device),
                          edge_index=torch.LongTensor(edges).t().to(device),
                          y=y.to(device), raw_y=raw_y,
                          smiles=row.smiles, mol=mol))
  return data_list


