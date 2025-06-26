import os
import requests
import csv

import numpy as np
import pandas as pd

import torch
from rdkit.Chem import PandasTools

import zipfile
import csv
import requests

import xgboost

from sklearn.metrics import mean_squared_error, r2_score




def get_A2A_data(data_dir, out_data_dir):
  from qsprpred.data import QSPRDataset
  # extract dataset file
  with zipfile.ZipFile('QSPRpred_tutorial_data.zip', 'r') as zip_ref:
    zip_ref.extractall(data_dir)

  # load A2A dataset
  dataset = QSPRDataset.fromTableFile(
      filename= os.path.join(data_dir, "A2A_LIGANDS.tsv"),
      store_dir=out_data_dir,
      name="ClassificationTutorialDataset",
      target_props=[{"name": "pchembl_value_Mean", "task": "SINGLECLASS", "th": [6.5]}],
      random_state=42
  )
  return dataset


def get_esol_data():
  # download ESOL
  CSV_URL= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'

  with requests.Session() as s:
      download = s.get(CSV_URL)
      decoded_content = download.content.decode('utf-8')
      cr = csv.reader(decoded_content.splitlines(), delimiter=',')
      my_list = list(cr)

  esol = pd.DataFrame(my_list[1:], columns=my_list[0], dtype=None)
  esol = esol.astype({'Compound ID':str,
              'ESOL predicted log solubility in mols per litre':float,
              'Minimum Degree':int,
              'Molecular Weight':float,
              'Number of H-Bond Donors':int,
              'Number of Rings':int,
              'Number of Rotatable Bonds':int,
              'Polar Surface Area':float,
              'measured log solubility in mols per litre':float,  # probably this one?
              'smiles':str})

  PandasTools.AddMoleculeColumnToFrame(esol, smilesCol='smiles')  # returns None ;(
  return esol


def binary_accuracy(pred, true):
  pred_class = pred[:, 0] < pred[:, 1]
  return torch.sum(pred_class == true.to(bool))/true.shape[0]



def get_smarts_pats():
  SMARTS_URL = "https://github.com/shenwanxiang/bidd-molmap/raw/47ef99361ba6823a1d2eb1a74e672dedc78d3536/molmap/feature/fingerprint/smarts_maccskey.py"

  with requests.Session() as s:
      download = s.get(SMARTS_URL)
      decoded_content = download.content.decode('utf-8')

  with open('smarts_maccskey.py', 'w') as f:
      f.write(decoded_content)

  from smarts_maccskey import smartsPatts

  smarts = [('MACCSFP_MACCSFP_' + k[7:], v) for k, v in smartsPatts.items()]
  smarts = dict(smarts)
  return smarts


SMARTS_PATTS = get_smarts_pats()


def get_smarts(key):
  return SMARTS_PATTS[key]


def get_smarts_by_number(iterated_integers):
  return SMARTS_PATTS[f'MACCSFP_MACCSFP_{iterated_integers}'] if isinstance(iterated_integers, (int, np.int_)) else [SMARTS_PATTS[f'MACCSFP_MACCSFP_{i}'] for i in iterated_integers]



def score_model(model, dataset):
  scores = {
      'train_mse': mean_squared_error(model.predict(dataset.X), dataset.y),
      'train_r2': r2_score(model.predict(dataset.X), dataset.y),
      'test_mse':  mean_squared_error(model.predict(dataset.X_ind), dataset.y_ind),
      'test_r2':  r2_score(model.predict(dataset.X_ind), dataset.y_ind)
      }
  return scores


def feature_removal_score(k, dataset, shap_values, rank):
  # k = 5
  topk = rank[-k:]
  randk = np.random.randint(len(rank), size=k)
  lowk = rank[:k]

  scores = []
  for rem, method in zip([topk, randk, lowk], ['topk', 'random', 'lowk']):
    dataset.dropDescriptors(np.array(shap_values.feature_names)[rem])
    model = xgboost.XGBRFRegressor()  # train model
    model = model.fit(dataset.X, dataset.y)
    metrics = score_model(model, dataset)  # score model
    metrics.update({'k':k, 'method':method})
    scores.append(metrics)

    dataset.restoreDescriptorSets(dataset.descriptorSets)  # restore descriptors
  return scores


