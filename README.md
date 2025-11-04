<div align="center">
  <h1>
    X-DILiver: an ensemble learning framework for predicting drug-induced liver injury
  </h1>
  <p><i>Multimodal machine learning framework for predicting drug-induced liver injury using ensemble methods and data augmentation</i></p>

  ![Python](https://img.shields.io/badge/python-3.8+-blue)
  ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
  ![rdkit - Version](https://img.shields.io/badge/rdkit-2022.3.3-blue)

</div>

<div align="center">
  <img src="https://github.com/Chemjg/X-DILiver/blob/main/img/workflow.png" width="140%">
</div>

Drug-induced liver injury (DILI) is a primary cause of drug attrition, associated with over 1000 medications and accounting for 32% of marketed drugs withdrawn due to toxicity. This has created an urgent demand for high-accuracy computational methods predicting DILI risk early in the drug development pipeline. We developed **X-DILiver, a predictive framework built on the largest DILI-annotated dataset to date. We used data augmentation to improve model robustness and address class imbalance**. A library of 312 machine learning models was created using various algorithms and molecular features, and then used an optimized ensemble strategy. The final model is an ensemble of two extreme gradient boosting models and seven recurrent neural networks. It achieved 0.68 accuracy and a 0.51 Matthews correlation coefficient on an external test set, outperforming all other publicly available DILI prediction models. X-DILiver is a reliable tool to predict DILI potential, thereby accelerating and improving drug discovery safety. To facilitate broad access X-DILiver is accessible at **http://ssbio.cau.ac.kr/software/X-DILiver/**. The training and test datasets are available at the web site. 

**Keywords:** Drug-induced liver injury, Drug discovery, Hepatotoxicity, ADME-Tox, Machine learning

## Installation

To install and use the package, first create the `conda` environment as follows:
```bash 
conda env create -f X-DILiver_env.yml
```

Then, activate the environment:
```bash
conda activate x-diliver
```

## Getting Started

To run `X-DILiver`, you first need to prepare your input data. The file should be in `.txt` format and include the following required columns:

* `smiles`: The SMILES string representation of each molecule in the dataset.

You can find our training dataset for model training on the X-DILiver webserver [Download](http://ssbio.cau.ac.kr/software/X-DILiver/).

### Local Installation Requirements

To use X-DILiver locally, you must:
1. Purchase the DRAGON7 software from the vendor
2. Place the DRAGON7 descriptor calculator files in the designated `dragon` folder within your project directory

DRAGON7-based molecular descriptors are required for DILI predictions.

**Note:** For users without access to the DRAGON7 software, we provide a web-based interface that requires no local installation: **[X-DILiver](http://ssbio.cau.ac.kr/software/X-DILiver/)**

## Usage

You can use X-DILiver in two ways:

### 1. Web Server

For a quick, browser-based prediction without installation, visit our web server:

**[X-DILiver Web Server](http://ssbio.cau.ac.kr/software/X-DILiver/)**

Simply upload your SMILES file and get predictions instantly.

### 2. Command Line Interface

Once the input data file has been prepared, you can run `X-DILiver` locally in the following way:

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Input SMILES file path | required |
| `--output` | Output file path | required |
| `--work_dir` | Working directory | `./` |
| `--data_dir` | Model data directory | `./Dat` |
| `--verbose` or `-v` | Enable verbose logging | disabled |

### Examples

**Basic usage:**
```bash
python main.py --input inputFILE.txt --output outputFILE.txt
```
**With verbose logging:**
```bash
python main.py --input inputFILE.txt --output outputFILE.txt -v
```

### Input Format

`inputFILE.txt` (SMILES format):

| SMILES |
|---------|
| O=C(CCBr)N1CCN(C(=O)CCBr)CC1 |
| CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1 |
| CCCN(CCC)CCc1cccc2c1CC(=O)N2 |
| NC(=O)CCC(N)C(=O)O |


### Output Format

`outputFILE.txt`:

| SMILES | Probability | Prediction |
|---------|----------|----------|
| O=C(CCBr)N1CCN(C(=O)CCBr)CC1 | 0.811 | Highly Toxic |
| CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1 | 0.781 | Highly Toxic |
| CCCN(CCC)CCc1cccc2c1CC(=O)N2 | 0.437 | safe |
| NC(=O)CCC(N)C(=O)O | 0.552 | Less DILI concern |
| ABCDEF | Invalid SMILES |          |      


### Advanced Usage: Using Pre-calculated Descriptors

If you have already calculated molecular descriptors externally, you can use X-DILiver's prediction module directly without recalculating descriptors:

**Basic usage:**
```bash
python test_scripts/test.py --input test_scripts/MolwithDes.txt --output test_scripts/outputFILE.txt
```

**Note:** The descriptor file must have the same format as the output from the descriptor calculation pipeline (Mordred + DRAGON7 features).


## License

`Chemjg` is licensed under the MIT License. See the [LICENSE](https://github.com/chemotargets/assay_inspector/blob/master/LICENSE) file.
