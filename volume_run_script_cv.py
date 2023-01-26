import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchio as tio
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from itertools import chain
from sys import argv

from models import MIL_classifier, unet
from datasets.patch_datasets import PatchBagDataset, collate_bag_batches

def load_models(pretrained=True):
    # encoder parameters
    num_input_channels = 1
    base_num_features = 32
    num_classes = 2
    net_num_pool_op_kernel_sizes = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
    conv_per_stage = 2
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    net_conv_kernel_sizes = [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],[3, 3, 3], [3, 3, 3]]

    enc = unet.Encoder(num_input_channels, base_num_features, num_classes, len(net_num_pool_op_kernel_sizes),
               net_num_pool_op_kernel_sizes, net_conv_kernel_sizes)

    if pretrained:
        pt_weights = torch.load("/home/jleiby/abdominal_ct/trained_models/pt_Encoder_weights_liver.pt")
        enc.load_state_dict(pt_weights)
    mod = MIL_classifier.MIL_model(dims=[1120, 512, 256], return_features=True)

    return enc, mod


def train_fold_single_gpu(fold, device, input_dim):
    print("_"*15, "FOLD: ", fold, "_"*15)

    tr_md = pd.read_csv(f"/home/jleiby/abdominal_ct/data/text_files/volume_cv/fold_{fold}_ct_labels_TRAINING.csv", names=['ID', 'label'])
    tr_files = tr_md.ID.tolist()
    tr_labels = tr_md.label.tolist()

    tr_bag_data = PatchBagDataset(data_path = data_path,
                         files = tr_files,
                         labels = tr_labels)

    tr_dl = DataLoader(tr_bag_data, batch_size=16, shuffle=True, collate_fn=collate_bag_batches)

    te_md = pd.read_csv(f"/home/jleiby/abdominal_ct/data/text_files/volume_cv/fold_{fold}_ct_labels_TESTING.csv", names=['ID', 'label'])
    te_files = te_md.ID.tolist()
    te_labels = te_md.label.tolist()

    te_bag_data = PatchBagDataset(data_path = data_path,
                         files = te_files,
                         labels = te_labels)

    te_dl = DataLoader(te_bag_data, batch_size=16, shuffle=True, collate_fn=collate_bag_batches)

    enc, mod = load_models(pretrained=True)
    enc.to(device)
    mod.to(device)

    opt = torch.optim.AdamW(chain(enc.parameters(), mod.parameters()))
    criterion = nn.BCELoss().to(device)

    # train_loss = []

    for j in range(50):
        mod.train()
        for _, (dat, lab, fn) in enumerate(tr_dl):
            # print("Next Batch")
            # print(lab.shape)
            opt.zero_grad()
            lab = lab.to(device)
            # model output for whole batch
            b_out = torch.empty(0).to(device)
            for sampler in dat:
                # print("Next sample")
                all_feats = torch.empty(0).to(device)
                for i, patch in enumerate(sampler):
                    print(f"Patch {i}")
                    print(torch.cuda.memory_allocated(device))
                    patch_tensor = patch.ct.data.unsqueeze(0).to(device)
                    skips, db = enc(patch_tensor)
                    print("model ran")
                    print(torch.cuda.memory_allocated(device))
                    feats = torch.empty(0).to(device)
                    for s in skips:
                        # global pooling
                        p = torch.mean(s, axis = [2,3,4])
                        # cat pooled features
                        feats = torch.cat((feats, p), axis =-1)
                        # print("skips")
                        # print(torch.cuda.memory_allocated(device))
                    # cat all patch feature embeddings to PxF  (num patches x features)
                    all_feats = torch.cat((all_feats, feats), dim=0)
                    print(all_feats.shape)

                out, _, _ = mod(all_feats)
                b_out = torch.cat([b_out, out])
                # print(b_out.shape)
            b_out = b_out.unsqueeze(1)
            # print(b_out.shape)

            loss = criterion(b_out, lab)
            loss.backward()
            opt.step()

        # performance metrics
        all_labels = torch.empty(0).to(device)
        all_out = torch.empty(0).to(device)
        with torch.no_grad():
            mod.eval()
            for _, (dat, lab, fn) in enumerate(te_dl):
                # print("Next Batch")
                # print(lab.shape)
                opt.zero_grad()
                lab.to(device)
                # model output for whole batch
                b_out = torch.empty(0).to(device)
                for sampler in dat:
                    # print("Next sample")
                    all_feats = torch.empty(0).to(device)
                    for i, patch in enumerate(sampler):
                        # print(f"Patch {i}")
                        # tell the patch ct tensor to require gradients ... 
                        skips, db = enc(patch.ct.data.requires_grad_().unsqueeze(0).to(device))
                        feats = torch.empty(0)
                        for s in skips:
                            # global pooling
                            p = torch.mean(s, axis = [2,3,4])
                            # cat pooled features
                            feats = torch.cat((feats, p), axis =-1)
                        # cat all patch feature embeddings to PxF  (num patches x features)
                        all_feats = torch.cat((all_feats, feats), dim=0)
                    out, _, _ = mod(all_feats)
                    b_out = torch.cat([b_out, out])
                b_out = b_out.unsqueeze(1)
                # print(b_out.shape)
                all_labels = torch.cat((all_labels, lab), 0)
                all_out = torch.cat((all_out, b_out), 0)
        print("EPOCH: ", j)
        auc = roc_auc_score(all_labels.detach().cpu().numpy(), all_out.detach().cpu().numpy())
        print(f"AUC: {auc:.5f}")
        aupr = average_precision_score(all_labels.detach().cpu().numpy(), all_out.detach().cpu().numpy())
        print(f"AUPRC: {aupr:.5f} \n")

    return(auc, aupr)

def train_fold_multi_gpu(fold, device, input_dim):
    print("_"*15, "FOLD: ", fold, "_"*15)

    tr_md = pd.read_csv(f"/home/jleiby/abdominal_ct/data/text_files/volume_cv/fold_{fold}_ct_labels_TRAINING.csv", names=['ID', 'label'])
    tr_files = tr_md.ID.tolist()
    tr_labels = tr_md.label.tolist()

    tr_bag_data = PatchBagDataset(data_path = data_path,
                         files = tr_files,
                         labels = tr_labels)

    tr_dl = DataLoader(tr_bag_data, batch_size=1, shuffle=True, collate_fn=collate_bag_batches)

    te_md = pd.read_csv(f"/home/jleiby/abdominal_ct/data/text_files/volume_cv/fold_{fold}_ct_labels_TESTING.csv", names=['ID', 'label'])
    te_files = te_md.ID.tolist()
    te_labels = te_md.label.tolist()

    te_bag_data = PatchBagDataset(data_path = data_path,
                         files = te_files,
                         labels = te_labels)

    te_dl = DataLoader(te_bag_data, batch_size=16, shuffle=True, collate_fn=collate_bag_batches)

    enc, mod = load_models(pretrained=True)
    # enc.to(device)
    mod.to(device[1])

    opt = torch.optim.Adam(chain(enc.parameters(), mod.parameters()))
    criterion = nn.BCELoss().to(device[1])

    # train_loss = []

    for j in range(10):
        batch_loss = 0.0
        mod.train()
        opt.zero_grad()
        for batch_idx, (dat, lab, fn) in enumerate(tr_dl):
            # print("Next Batch")
            # print(lab.shape)
            
            lab = lab.to(device[1])
            # model output for whole batch
            b_out = torch.empty(0).to(device[1])
            for sampler in dat:
                # print("Next sample")
                all_feats = torch.empty(0).to(device[1])
                for i, patch in enumerate(sampler):
                    if(i==40): break
                    # print(f"Patch {i}")
                    # print(torch.cuda.memory_allocated(device[0]))
                    patch_tensor = patch.ct.data.unsqueeze(0)
                    skips, db = enc(patch_tensor, device)
                    # print("model ran")
                    # print(torch.cuda.memory_allocated(device[0]))
                    feats = torch.empty(0).to(device[1])
                    for s in skips:
                        # global pooling
                        p = torch.mean(s, axis = [2,3,4])
                        # cat pooled features
                        feats = torch.cat((feats, p), axis =-1)
                    # cat all patch feature embeddings to PxF  (num patches x features)
                    all_feats = torch.cat((all_feats, feats), dim=0)
                    # print(all_feats.shape)
                    

                out, _, _ = mod(all_feats)
                b_out = torch.cat([b_out, out])
                # print(b_out.shape)
            b_out = b_out.unsqueeze(1)
            # print(b_out.shape)

            loss = criterion(b_out, lab)
            batch_loss += loss.item()
            loss.backward()
            # artifical batch size of 10
            if ((batch_idx + 1) % 10 == 0) or (batch_idx + 1 == len(tr_dl)): 
                print("Train loss: ", batch_loss/(10))
                opt.step()
                opt.zero_grad()
                batch_loss = 0.0

        # performance metrics
        all_labels = torch.empty(0).to(device[1])
        all_out = torch.empty(0).to(device[1])
        with torch.no_grad():
            mod.eval()
            for _, (dat, lab, fn) in enumerate(te_dl):
                # print("Next Batch")
                # print(lab.shape)
                opt.zero_grad()
                lab = lab.to(device[1])
                # model output for whole batch
                b_out = torch.empty(0).to(device[1])
                for sampler in dat:
                    # print("Next sample")
                    all_feats = torch.empty(0).to(device[1])
                    for i, patch in enumerate(sampler): 
                        patch_tensor = patch.ct.data.unsqueeze(0)
                        skips, db = enc(patch_tensor, device)
                        feats = torch.empty(0).to(device[1])
                        for s in skips:
                            # global pooling
                            p = torch.mean(s, axis = [2,3,4])
                            # cat pooled features
                            feats = torch.cat((feats, p), axis =-1)
                        # cat all patch feature embeddings to PxF  (num patches x features)
                        all_feats = torch.cat((all_feats, feats), dim=0)
                    out, _, _ = mod(all_feats)
                    b_out = torch.cat([b_out, out])
                b_out = b_out.unsqueeze(1)
                # print(b_out.shape)
                all_labels = torch.cat((all_labels, lab), 0)
                all_out = torch.cat((all_out, b_out), 0)
        print("EPOCH: ", j)
        auc = roc_auc_score(all_labels.detach().cpu().numpy(), all_out.detach().cpu().numpy())
        print(f"AUC: {auc:.5f}")
        aupr = average_precision_score(all_labels.detach().cpu().numpy(), all_out.detach().cpu().numpy())
        print(f"AUPRC: {aupr:.5f} \n")

    return(auc, aupr)


device = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2')]

data_path =  "/project/kimlab_genie/nnUNet_preprocess_full_data/nnUNet_preprocessed/Task998_LiverPreprocess/nnUNetData_plans_v2.1_stage1/"

folds = [1,2,3,4,5]

all_auc = []
all_auprc = []

for f in folds:
    fold_auc, fold_auprc = train_fold_multi_gpu(f, device, 1120)
    all_auc.append(fold_auc)
    all_auprc.append(fold_auprc)


print("AUC: ", all_auc)
print("AUPRC: ", all_auprc)
