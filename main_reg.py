import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
# from model_cls import BioEncoder, HyperGraphSynergy, HgnnEncoder, Decoder
from model_hetero2 import HHGNN
from sklearn.model_selection import KFold
import os
import copy
import glob
import tqdm
from drug_util import GraphDataset, collate
from utils import regression_metric, set_seed_all
from similarity import get_Cosin_Similarity, get_pvalue_matrix
from process_data import getData
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# drug_num = 87
# cline_num = 55
drug_num = 38
cline_num = 32


# def get_matrix(synergy, drug_g, cline_g):
#     synergy_ = synergy.copy()
#     num_node = drug_num + cline_num
#     num_type = 2
#     association_matrix_d = torch.zeros(drug_num, num_type, num_node)  # 38,2,70
#     association_matrix_c = torch.zeros(cline_num, num_type, num_node)  # 32,2,70
#     # edge_mask = torch.zeros(num_node, num_type, 256, dtype=torch.int64)  # 261,3,256
    
#     for row in synergy_[:, :3]:
#         association_matrix_d[row[0]][0][row[1:]] = 1
#         association_matrix_d[row[1]][0][row[0, 2]] = 1
#         association_matrix_c[row[2]-drug_num][0][row[:2]] = 1
    
#     for elem in drug_g:
#         for i in range(len(elem)-1):
#             for j in range(i+1, len(elem)):
#                 association_matrix_d[elem[i]][1][elem[j]] += 1
#                 association_matrix_d[elem[j]][1][elem[i]] += 1
                
#     for elem in cline_g:
#         for i in range(len(elem)-1):
#             for j in range(i+1, len(elem)):
#                 association_matrix_c[elem[i]][1][elem[j]+drug_num] += 1
#                 association_matrix_c[elem[j]][1][elem[i]+drug_num] += 1

#     return association_matrix_d.unsqueeze(dim=2), association_matrix_c.unsqueeze(dim=2)

def generate_G_from_H(H):
    # DV = np.sum(H * W.reshape(-1, 1), axis=1)
    DV = torch.sum(H, 1, dtype=torch.float32)
    DE = torch.sum(H, 0, dtype=torch.float32)

    # invDE = np.mat(np.diag(np.power(DE, -1)))
    invDE = torch.diag(torch.pow(DE, -1))

    S = invDE @ H.T
    return S


def get_among_cls_h(synergy_pos):
    for row in synergy_pos:
        row[3] = 1 if row[3] >= 30 else 0
    synergy_pos = pd.DataFrame([i for i in synergy_pos if i[3] == 1]).values
    synergy_pos1 = synergy_pos.copy()
    col01 = []
    for row in synergy_pos1[:, :2]:
        col01.append(sorted(set(row)))

    col01 = np.array(col01)
    synergy_pos1[:, :2] = col01
    unique_rows, indices, counts = np.unique(synergy_pos1[:, :2], axis=0, return_counts=True, return_index=True)
    cline_hyperedge_list = []
    num = 0
    for row in unique_rows:  # 327
        ind = np.where((synergy_pos1[:, 0] == row[0]) & (synergy_pos1[:, 1] == row[1]))[0]
        cline_set = np.array(sorted(set(synergy_pos1[ind][:, 2])))-drug_num
        if len(cline_set) > 1:
            cline_hyperedge_list.append(cline_set)  # 222
    H_c = torch.zeros([cline_num, len(cline_hyperedge_list)])
    for i, hyperedge in enumerate(cline_hyperedge_list):
        H_c[hyperedge, i] = 1.0

    arr_true = np.zeros((drug_num, cline_num))
    for line in synergy_pos:
        arr_true[line[0], line[2]-drug_num] = 1
        arr_true[line[1], line[2]-drug_num] = 1
    rows, cols = np.where(arr_true == 1)
    drug_hyperedge_list = []
    num = 0
    for row, col in zip(rows, cols):
        ind1 = np.where((synergy_pos[:, 0] == row) & (synergy_pos[:, 2]-drug_num == col))[0]
        ind2 = np.where((synergy_pos[:, 1] == row) & (synergy_pos[:, 2]-drug_num == col))[0]
        drug_set = np.array(sorted(set(list(synergy_pos[ind1][:, 1])+list(synergy_pos[ind2][:, 0]))))
        if len(drug_set) > 1:
            drug_hyperedge_list.append(drug_set)  # 570
    H_d = torch.zeros([drug_num, len(drug_hyperedge_list)])
    for i, hyperedge in enumerate(drug_hyperedge_list):
        H_d[hyperedge, i] = 1.0
        
    H_dc = torch.zeros([drug_num+cline_num, len(synergy_pos)])
    for i, hyperedge in enumerate(synergy_pos[:, :3]):
        H_dc[hyperedge, i] = 1.0
        
    H1 = torch.cat([H_d, torch.zeros((drug_num, len(cline_hyperedge_list)))], -1)  # 38, 570+222
    H2 = torch.cat([torch.zeros((cline_num, len(drug_hyperedge_list))), H_c], -1)  # 32, 570+222
    H3 = torch.cat([H1, H2], 0)
    DTH_cls = generate_G_from_H(H3).to(device)  # 792,70
    H = torch.cat([H3, H_dc], -1)  # 70,2486
    DTH_dc = generate_G_from_H(H_dc).to(device)  # 1694,70
    
    edge_count = torch.zeros(drug_num+cline_num)
    global_neigh_count = torch.zeros(drug_num+cline_num)
    hyperedge = dict()
    for i in range(len(H)):
        hyperedge[i] = torch.nonzero(H[i]).squeeze().to(device)
        edge_count[i] = len(torch.nonzero(H[i]))  # n,1
        neigh_list = []
        for j in torch.nonzero(H[i]).squeeze():
            neigh_list.append(torch.nonzero(H[:, j]).squeeze())
        neigh_list = torch.cat(neigh_list, -1)
        global_neigh_count[i] = len(torch.unique(neigh_list))-1
        
    input_weight = torch.cat([global_neigh_count.unsqueeze(1), edge_count.unsqueeze(1)], -1).to(device)
        
    return DTH_cls, DTH_dc, hyperedge, input_weight, len(drug_hyperedge_list)+len(cline_hyperedge_list)


def load_data(dataset):
    cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy = getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)
    # threshold = 30
    # for row in synergy:
    #     row[3] = 1 if row[3] >= threshold else 0
    
    drug_sim_matrix, cline_sim_matrix = get_sim_mat(drug_smiles_fea, np.array(gene_data, dtype='float32'))
    return drug_fea, cline_fea, synergy, drug_sim_matrix, cline_sim_matrix


def data_split(synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy])
    train_size = 0.95
    synergy_cv_data, synergy_test = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                             [int(train_size * len(synergy_pos))])
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
    test_ind = torch.from_numpy(np.array(synergy_test[:, 0:3])).long().to(device)
    return synergy_cv_data, test_ind, test_label


def get_sim_mat(drug_fea, cline_fea):
    drug_sim_matrix = np.array(get_Cosin_Similarity(drug_fea))
    cline_sim_matrix = np.array(get_pvalue_matrix(cline_fea))
    return torch.from_numpy(drug_sim_matrix).type(torch.FloatTensor).to(device), torch.from_numpy(
        cline_sim_matrix).type(torch.FloatTensor).to(device)


def train(index, label):
    loss_train = 0
    true_ls, pre_ls = [], []
    optimizer.zero_grad()
    for batch, (drug, cline) in enumerate(zip(drug_set, cline_set)):
        pred = model(drug, cline[0], DTH_cls, DTH_dc, hyperedge, input_weight, index)  # rec_drug, rec_cline
        loss = loss_func(pred, label)
        # loss_rec_1 = loss_func(rec_drug, drug_sim_mat)
        # loss_rec_2 = loss_func(rec_cline, cline_sim_mat)
        # loss = (1 - alpha) * loss + alpha * (loss_rec_1 + loss_rec_2)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        true_ls += label_train.cpu().detach().numpy().tolist()
        pre_ls += pred.cpu().detach().numpy().tolist()
    rmse, r2, pr = regression_metric(true_ls, pre_ls)
    return [rmse, r2, pr], loss_train


def test(index, label):
    model.eval()
    with torch.no_grad():
        for batch, (drug, cline) in enumerate(zip(drug_set, cline_set)):
            pred = model(drug, cline[0], DTH_cls, DTH_dc, hyperedge, input_weight, index)  # rec_drug, rec_cline
        loss = loss_func(pred, label)
        # loss_rec_1 = loss_func(rec_drug, drug_sim_mat)
        # loss_rec_2 = loss_func(rec_cline, cline_sim_mat)
        # loss = (1 - alpha) * loss + alpha * (loss_rec_1 + loss_rec_2)
        rmse, r2, pr = regression_metric(label.cpu().detach().numpy(),
                                         pred.cpu().detach().numpy())
        return [rmse, r2, pr], loss.item(), pred.cpu().detach().numpy()


if __name__ == '__main__':
    dataset_name = 'ONEIL'  # ONEIL or ALMANAC
    seed = 0
    cv_mode_ls = [1, 2, 3]
    epochs = 4000
    learning_rate = 0.004
    L2 = 1e-4
    alpha = 0.4
    for cv_mode in cv_mode_ls:
        path = 'result_reg/' + dataset_name + '_' + str(cv_mode) + '_'
        file = open(path + 'result.txt', 'w')
        set_seed_all(seed)
        drug_feature, cline_feature, synergy_data, drug_sim_mat, cline_sim_mat = load_data(dataset_name)
        
        synergy_data_ = copy.deepcopy(synergy_data)
        DTH_cls, DTH_dc, hyperedge, input_weight, num_cls = get_among_cls_h(synergy_data_)
        # drug_hypergraph, cline_hypergraph = get_matrix(synergy_data, drug_graph, cline_graph)
        
        drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature),
                                   collate_fn=collate, batch_size=len(drug_feature), shuffle=False)
        cline_set = Data.DataLoader(dataset=Data.TensorDataset(cline_feature),
                                    batch_size=len(cline_feature), shuffle=False)
        synergy_cv, index_test, label_test = data_split(synergy_data)
        if cv_mode == 1:
            cv_data = synergy_cv
        elif cv_mode == 2:
            cv_data = np.unique(synergy_cv[:, 2])
        else:
            cv_data = np.unique(np.vstack([synergy_cv[:, 0], synergy_cv[:, 1]]), axis=1).T
        final_metric = np.zeros(3)
        final_metric1 = np.zeros(3)
        fold_num = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, validation_index in kf.split(cv_data):
            if cv_mode == 1:
                synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
            elif cv_mode == 2:
                train_name, test_name = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array([i for i in synergy_cv if i[2] in train_name])
                synergy_validation = np.array([i for i in synergy_cv if i[2] in test_name])
            else:
                pair_train, pair_validation = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array(
                    [j for i in pair_train for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
                synergy_validation = np.array(
                    [j for i in pair_validation for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
            label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
            label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
            index_train = torch.from_numpy(synergy_train).long().to(device)
            index_validation = torch.from_numpy(synergy_validation).long().to(device)
            
            synergy_train_tmp = np.copy(synergy_train)
            
            for data in synergy_train_tmp:
                data[3] = 1 if data[3] >= 30 else 0
            
            pos_edge = np.array([t for t in synergy_train_tmp if t[3] != 0])
            edge_data = pos_edge[:, 0:3]
            synergy_edge = edge_data.reshape(1, -1)
            index_num = np.expand_dims(np.arange(len(edge_data)), axis=-1)
            synergy_num = np.concatenate((index_num, index_num, index_num), axis=1).reshape(1, -1)
            synergy_graph = np.concatenate((synergy_edge, synergy_num), axis=0)
            synergy_graph = torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)

            model = HHGNN(num_cls).to(device)
            loss_func = torch.nn.MSELoss()
            rec_loss = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)

            best_metric = [0, 0, 0]
            best_epoch = 0
            for epoch in tqdm.tqdm(range(epochs)):
                model.train()
                train_metric, train_loss = train(index_train, label_train)
                val_metric, val_loss, _ = test(index_validation, label_validation)

                if val_metric[1] > best_metric[1]:
                    model_path = f'Model_param_reg/{dataset_name}_mode{cv_mode}_fold{fold_num}_{best_epoch}.pth'
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                          'RMSE: {:.6f},'.format(val_metric[0]), 'R2: {:.6f},'.format(val_metric[1]),
                          'Pearson r: {:.6f},'.format(val_metric[2]))
                    best_metric = val_metric
                    best_epoch = epoch
                    model_path = f'Model_param_reg/{dataset_name}_mode{cv_mode}_fold{fold_num}_{best_epoch}.pth'
                    torch.save(model.state_dict(), model_path)
                                
            model.load_state_dict(torch.load(model_path))
            test_metric, _, y_test_pred = test(index_test, label_test)
            
            print('The results on test set, ',
                  'RMSE: {:.6f},'.format(test_metric[0]),
                  'R2: {:.6f},'.format(test_metric[1]), 
                  'Pearson r: {:.6f},'.format(test_metric[2]))
            
            file.write('val_metric:')
            for item in best_metric:
                file.write(str(item) + '\t')
            file.write('\ntest_metric:')
            for item in test_metric:
                file.write(str(item) + '\t')
            file.write('\n')
            
            final_metric += best_metric
            final_metric1 += test_metric
            fold_num = fold_num + 1
        
        final_metric /= 5
        final_metric1 /= 5
        
        file.write('average_val_metric:')
        for item in final_metric:
            file.write(str(item) + '\t')
        file.write('\naverage_test_metric:')
        for item in final_metric1:
            file.write(str(item) + '\t')
        file.write('\n')

