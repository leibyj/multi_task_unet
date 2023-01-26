import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from models import MIL_classifier
from datasets.patch_datasets import FeatureBagDataset, collate_bag_batches

from sys import argv


def train_fold(fold, device, input_dim):
    print("_"*15, "FOLD: ", fold, "_"*15)

    tr_md = pd.read_csv(f"/home/jleiby/abdominal_ct/data/text_files/features_cv/fold_{fold}_ct_labels_TRAINING.csv", names=['ID', 'label'])
    tr_files = tr_md.ID.tolist()
    tr_labels = tr_md.label.tolist()

    tr_bag_data = FeatureBagDataset(data_path = data_path,
                         files = tr_files,
                         labels = tr_labels)

    tr_dl = DataLoader(tr_bag_data, batch_size=10, shuffle=True, collate_fn=collate_bag_batches)

    te_md = pd.read_csv(f"/home/jleiby/abdominal_ct/data/text_files/features_cv/fold_{fold}_ct_labels_TESTING.csv", names=['ID', 'label'])
    te_files = te_md.ID.tolist()
    te_labels = te_md.label.tolist()

    te_bag_data = FeatureBagDataset(data_path = data_path,
                         files = te_files,
                         labels = te_labels)

    te_dl = DataLoader(te_bag_data, batch_size=10, shuffle=True, collate_fn=collate_bag_batches)

    mod = MIL_classifier.MIL_model(dims=[input_dim, 512, 256], return_features=True).to(device)
    opt = torch.optim.AdamW(mod.parameters())

    criterion = nn.BCELoss().to(device)

    train_loss = []

    for j in range(50):
        mod.train()
        for i, (dat, lab, fn) in enumerate(tr_dl):
            opt.zero_grad()
            b_out = torch.empty(0).to(device)
            for d in dat:
                d = d.to(device)
                out, A, _ = mod(d)
                b_out = torch.cat((b_out, out))
            b_out = b_out.unsqueeze(1)
            lab = lab.to(device)
            loss = criterion(b_out, lab)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            # print(loss.item())
        # print(A)
        # print("Loss: ", loss.item())

        # performance metrics
        all_labels = torch.empty(0).to(device)
        all_out = torch.empty(0).to(device)
        with torch.no_grad():
            mod.eval()
            for i, (dat, lab, fn) in enumerate(te_dl):
                b_out = torch.empty(0).to(device)
                for d in dat:
                    d = d.to(device)
                    out, _, _ = mod(d)
                    b_out = torch.cat((b_out, out))
                b_out = b_out.unsqueeze(1)
                lab = lab.to(device)

                all_labels = torch.cat((all_labels, lab), 0)
                all_out = torch.cat((all_out, b_out), 0)
        print("EPOCH: ", j)
        auc = roc_auc_score(all_labels.detach().cpu().numpy(), all_out.detach().cpu().numpy())
        print(f"AUC: {auc:.5f}")
        aupr = average_precision_score(all_labels.detach().cpu().numpy(), all_out.detach().cpu().numpy())
        print(f"AUPRC: {aupr:.5f} \n")

    # save model...
    torch.save(mod, f"/home/jleiby/abdominal_ct/trained_models/frozen_trained_feature_model_fold_{fold}.pt")

    # get features
    tr_dl = DataLoader(tr_bag_data, batch_size=10, shuffle=False, collate_fn=collate_bag_batches)
    te_dl = DataLoader(te_bag_data, batch_size=10, shuffle=False, collate_fn=collate_bag_batches)
    mod.eval()
    tr_feats = torch.empty(0)
    fns = []
    for i, (dat, lab, fn) in enumerate(tr_dl):
        for d in dat:
            d = d.to(device)
            _, _, feats = mod(d)
            feats = feats.unsqueeze(0).detach().cpu()
            tr_feats = torch.cat((tr_feats, feats), axis=0)
        fns.extend(fn)

    tr_feats = pd.DataFrame(tr_feats, index=fns)
    tr_feats.to_csv(f"/home/jleiby/abdominal_ct/feat_model_out/train/frozen_train_features_fold_{fold}.csv")
    
    
    te_feats = torch.empty(0)
    te_out = torch.empty(0)
    fns = []
    for i, (dat, lab, fn) in enumerate(te_dl):
        for d in dat:
            d = d.to(device)
            out, _, feats = mod(d)
            feats = feats.unsqueeze(0).detach().cpu()
            out = out.unsqueeze(0).detach().cpu()
            te_feats = torch.cat((te_feats, feats), axis=0)
            te_out = torch.cat((te_out, out), axis=0)
        fns.extend(fn)

    te_feats = pd.DataFrame(te_feats, index=fns)
    te_feats.to_csv(f"/home/jleiby/abdominal_ct/feat_model_out/test/frozen_test_features_fold_{fold}.csv")
    te_out = pd.DataFrame(te_out, index=fns)
    te_out.to_csv(f"/home/jleiby/abdominal_ct/feat_model_out/test/frozen_test_out_fold_{fold}.csv")

    return(auc, aupr)


# fold = argv[1]
device = str(argv[1])

data_path = "/home/jleiby/abdominal_ct/data/encoder_only/overlap_data/"

folds = [1,2,3,4,5]

all_auc = []
all_auprc = []

for f in folds:
    fold_auc, fold_auprc = train_fold(f, device, 1120)
    all_auc.append(fold_auc)
    all_auprc.append(fold_auprc)


print("AUC: ", all_auc)
print("AUPRC: ", all_auprc)
