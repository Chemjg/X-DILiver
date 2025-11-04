# -*- coding: utf-8 -*-

import os
import pandas as pd
from rdkit import Chem, RDLogger
from mordred import Calculator, descriptors

RDLogger.DisableLog('rdApp.*')

def Mordred(smiles):

    try:
        descriptors_dict = {}
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return "Invalid SMILES: Cannot parse"

        calc = Calculator(descriptors, ignore_3D=False)
        des = calc(mol)
        des_dict = des.fill_missing().asdict()
        descriptors_dict.update(des_dict)

    except Exception as e:
        descriptors_dict = str(e)

    return descriptors_dict


def calculateMord(inputFilePath, work):

    data = {}
    smiles_dict = {}

    # 입력 파일 읽기
    with open(inputFilePath) as f:
        for line in f:
            molecule = line.strip().split('\t')[0].replace(' ', '')
            smiles_dict[molecule] = 1

    for smiles in smiles_dict.keys():
        descriptors = Mordred(smiles)

        if type(descriptors) != dict:
            continue

        data.update({smiles: descriptors})

    with open(work + '/Dsc_Mord.txt', 'w') as g:
        if data:
            desOrder = sorted(data[list(data.keys())[0]].keys())
            g.write('NAME\t' + '\t'.join(desOrder) + '\n')

            for smiles in data.keys():
                descriptors = data[smiles]
                if len(descriptors) == 1825:
                    text = smiles + '\t' + '\t'.join([str(descriptors[d]) for d in desOrder]) + '\n'
                    g.write(text)

    return True

def configGenerator(work, inputFilePath, filename='./Dat/Standard.drs'):


    with open(filename) as f:
        with open(work + '/config.drs', 'w') as g:

            dscFilePath = work + '/Dsc_Dra.txt'
            logFilePath = work + '/Log_Dra.txt'

            n = 0
            for line in f:
                n += 1

                if n == 50:
                    g.write(line.replace('Input_File', inputFilePath))
                elif n == 90:
                    g.write(line.replace('Dsc_File', dscFilePath))
                elif n == 92:
                    g.write(line.replace('Log_File', logFilePath))
                else:
                    g.write(line)


def callDragon(work):

    configPath = work + '/config.drs'
    os.system('/home/ssbio/flask/dragon/dragon7shell -s ' + configPath)


def featureFusionner(work):

    from functools import reduce

    Dsc_Mord = pd.read_csv(work + '/Dsc_Mord_fixed.txt', delimiter='\t')
    Dsc_Dra = pd.read_csv(work + '/Dsc_Dra_fixed.txt', delimiter='\t')

    # NAME을 기준으로 내부 조인
    Merge = reduce(
        lambda left, right: pd.merge(left, right, how="inner", on='NAME'),
        [Dsc_Dra, Dsc_Mord]
    )

    Merge.to_csv(work + '/Fusion_Dra_Mord.txt', sep="\t", index=False, na_rep="na")


def merge_descriptors(file_a, file_b, output_file):

    with open(file_a, 'r') as f_a:
        descriptors_a = f_a.read().strip().split('\t')

    with open(file_b, 'r') as f_b:
        descriptors_b = f_b.read().strip().split('\t')

    merged_descriptors = list(set(descriptors_a + descriptors_b))

    merged_descriptors.sort()

    with open(output_file, 'w') as f_out:
        f_out.write('\t'.join(merged_descriptors))


def saveSelectedWithToxicity(fset, ifile, ofile):

    with open(ifile, 'r') as f:
        headers = f.readline().strip().split('\t')

    feature_index = [headers.index(f) for f in fset if f in headers]

    with open(ofile, 'w') as f_out:

        selected_headers = [headers[i] for i in feature_index]
        f_out.write('\t'.join(selected_headers) + '\n')

        with open(ifile, 'r') as f_in:
            next(f_in)
            for line in f_in:
                data = line.strip().split('\t')
                if len(data) > max(feature_index):
                    selected_data = [data[i] for i in feature_index]
                    f_out.write('\t'.join(selected_data) + '\n')


def NA_rowChecker(work, Predictable, UnPredictable):

    MordDra_Fixed = pd.read_csv(
        work + "/DescFixedReduced.txt",
        sep='\t',
        na_values=['na', 'nan']
    )

    rows_with_na = MordDra_Fixed.index[
        MordDra_Fixed.isnull().any(axis=1)
    ].tolist()

    removed_rows_names = MordDra_Fixed.loc[rows_with_na, 'NAME'].tolist()

    data_ChK = MordDra_Fixed.drop(index=rows_with_na)

    data_ChK.to_csv(
        work + "/DescFixed_rowChecked.txt",
        sep='\t',
        index=False
    )

    with open(UnPredictable, "a") as Unpred:
        for new_unpred in removed_rows_names:
            Unpred.write(new_unpred + "\n")

    predictable_smiles = pd.read_csv(
        Predictable,
        sep="\t",
        header=None
    )

    filtered_smiles = predictable_smiles[
        ~predictable_smiles[0].isin(removed_rows_names)
    ]

    filtered_smiles.to_csv(
        work + "/Predictable_smi_checked.txt",
        index=False,
        header=False,
        sep="\t"
    )
