# -*- coding: ms949 -*-
from pandas import read_csv
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

def convert_to_graph(Fname, maxNumAtoms, NumFeats):

    Data = read_csv(Fname, delimiter='\t', comment=None)
    Smi = Data["SMILES"].values.tolist()

    adj = []
    features = []

    cnt = 0

    for i in Smi:

        cnt += 1
        # Mol
        iMol = Chem.MolFromSmiles(i.rstrip())

        # Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)

        # Feature
        if (iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, NumFeats))
            iFeatureTmp = []

            for atom in iMol.GetAtoms():
                iFeatureTmp.append(atom_feature(atom))
            iFeature[0:len(iFeatureTmp), 0:NumFeats] = iFeatureTmp
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(np.asarray(iAdj))

    adj = np.array(adj)
    features = np.array(features)

    np.save('./' + str(Fname[:-4]) + '_adj_matrix', adj)
    np.save('./' + str(Fname[:-4]) + '_feature_matrix', features)


def get_ring_info(atom):

    ring_info_feature = []
    for i in range(3, 9):
        if atom.IsInRingSize(i):
            ring_info_feature.append(1)
        else:
            ring_info_feature.append(0)

    return ring_info_feature


def one_of_k_encoding(x, allowable_set):

    if x not in allowable_set:

        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):

    if x not in allowable_set:

        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))

def atom_feature(atom):

    return np.array(
                    one_of_k_encoding_unk(atom.GetSymbol(), #40
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + #7
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5]) + #6
                    one_of_k_encoding_unk(atom.GetHybridization(),
                                          ['S', 'SP', 'SP2', 'SP3', 'SP2D', 'SP3D', 'SP3D2']) + #7
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + #7
                    one_of_k_encoding(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) + #9
                    one_of_k_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]) + #10
                    one_of_k_encoding_unk(atom.GetChiralTag(), ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', #9
                                                                 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER', 'CHI_TETRAHEDRAL',
                                                                 'CHI_ALLENE', 'CHI_SQUAREPLANAR', 'CHI_TRIGONALBIPYRAMIDAL',
                                                                 'CHI_OCTAHEDRAL']) +
                    [atom.GetIsAromatic()] + get_ring_info(atom) #7 (1 + 6)
                    )