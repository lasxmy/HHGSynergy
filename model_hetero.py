import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
from utils import reset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# drug_num = 87
# cline_num = 55
drug_num = 38
cline_num = 32

# class HHEdgeCons(nn.Module):
#     def __init__(self, data_stat, recons_error_lambda=0.1, l2_lambda=0.2, recons_lambda=0.01):
#         super(HHEdgeCons, self).__init__()
#         self.l2_lambda = l2_lambda
#         self.recons_lambda = recons_lambda
#         self.num_node = data_stat['num_node']
#         self.num_type = data_stat['num_type']
#         self.num_fea = data_stat['num_fea']
#         self.recons_error_lambda = recons_error_lambda
#         self.linear = nn.Parameter(
#             torch.nn.init.xavier_uniform_(torch.rand(self.num_node, self.num_type, 1, self.num_node)))

#         self.recon_original_proj = nn.Parameter(torch.nn.init.xavier_uniform_(
#             torch.rand(self.num_type * self.num_type, self.num_fea, self.num_fea)))  # 9,256,256

#     def node_project(self, fea, proj, node_multi_mask):
#         return torch.vstack(
#             [torch.matmul(proj[node_multi_mask[i] * self.num_type + j], fea[i][j]) for i in range(self.num_node)
#              for j in range(self.num_type)]).reshape(self.num_node, self.num_type, self.num_fea)

#     def forward(self, feature, mask, node_multi_mask):
#         feature_frozen = feature.detach()
#         inp_4_recon = feature_frozen.expand(self.num_type, self.num_node, self.num_fea)  # 3,261,256
#         # different candidate slave nodes
#         linear_mask = self.linear * mask  # 261,3,1,261
#         # selected slave nodes with reconstruction value larger than 0
#         l0_mask = linear_mask > 0
#         linear_selected = linear_mask.masked_fill(~l0_mask, value=torch.tensor(0))  # 把小于等于0的位置填充为0
#         reconstruct_fea = torch.matmul(linear_selected, inp_4_recon).squeeze()  # 261,3,256

#         inp_original = torch.transpose(inp_4_recon, 1, 0)  # 261,3,256

#         inp_projected_selected = self.node_project(inp_original, self.recon_original_proj, node_multi_mask)  # 261,3,256
#         # reconstruction error
#         linear_comb_l1 = torch.sum(torch.norm(linear_selected.squeeze(), dim=-1, p=1))
#         linear_comb_l2 = torch.sum(torch.norm(linear_selected.squeeze(), dim=-1, p=2))
#         recon_error = torch.sum(torch.norm(inp_projected_selected - reconstruct_fea, dim=-1, p=2))
#         recon_loss = self.recons_lambda * recon_error + self.l2_lambda * linear_comb_l2 + linear_comb_l1
#         return linear_selected, self.recons_error_lambda * recon_loss


# class HHEdgeMP(nn.Module):
#     def __init__(self):
#         super(HHEdgeMP, self).__init__()

#     def forward(self, feature, linear_selected_cls, linear_selected_dc):
#         edge_fea = linear_selected @ feature  # 261,3,1,256
#         edge_fea = edge_fea.squeeze() + torch.transpose(feature.expand(self.num_type, self.num_node, self.num_fea), 1, 0)  # 261, 3, 256
#         edge_norm = torch.sum(linear_selected.squeeze(), dim=-1) + torch.ones(self.num_node, self.num_type)  # 261, 3
#         edge_fea = torch.div(edge_fea, edge_norm.unsqueeze(dim=2))  # 261, 3, 256
        
#         edge_fea_cls = linear_selected_cls @ feature  # (num_hyperedge,70)@(70,feat)
#         edge_fea_dc = linear_selected_dc @ feature
#         return edge_fea


class MultiheadWeight(nn.Module):
    def __init__(self, num_cls):
        super(MultiheadWeight, self).__init__()
        self.num_node = drug_num + cline_num
        self.edge_type = 4
        self.node_type = 2
        self.num_fea = 128
        self.num_head = 4
        self.num_cls = num_cls
        self.num_samples = 2

        self.multi_head_node_proj = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.rand(self.node_type, 128, 128)))
        self.multi_head_edge_proj = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.rand(self.edge_type, 128, 128)))  # 9,256,256

        self.act_1 = nn.Softmax(dim=1)
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        torch.nn.init.xavier_uniform_(self.fc[0].weight)

    def node_project(self, fea):
        fea1 = fea[:drug_num] @ self.multi_head_node_proj[0]
        fea2 = fea[drug_num:] @ self.multi_head_node_proj[1]
        return torch.cat([fea1, fea2], 0)

    def edge_project(self, fea, i, j):
        fea1 = fea[:, 0] @ self.multi_head_edge_proj[i]
        fea2 = fea[:, 1] @ self.multi_head_edge_proj[j]
        stacked_fea = torch.stack((fea1, fea2), dim=1).reshape(-1, self.num_fea)
        return stacked_fea

    def forward(self, feature, edge_fea, hyperedge_dict):
        drug_edge_feat = []
        cline_edge_feat = []
        for key in hyperedge_dict.keys():
            value = hyperedge_dict[key]
            mask = value < self.num_cls
            edge_fea_node = edge_fea[hyperedge_dict[key]]
            edge_fea_cls = edge_fea_node[mask]
            edge_fea_dc = edge_fea_node[~mask]
            # indices1 = torch.randperm(len(edge_fea_cls))[:len(edge_fea_cls)//2].to(device)
            # edge_fea1 = edge_fea_cls[indices1].mean(dim=0, keepdim=True)
            # indices2 = torch.randperm(len(edge_fea_dc))[:len(edge_fea_dc)//2].to(device)
            # edge_fea2 = edge_fea_dc[indices2].mean(dim=0, keepdim=True)
            edge_fea1 = edge_fea_cls.mean(dim=0, keepdim=True)
            edge_fea2 = edge_fea_dc.mean(dim=0, keepdim=True)
            if key < drug_num:
                drug_edge_feat.append(torch.cat((edge_fea1, edge_fea2), 0))
            else:
                cline_edge_feat.append(torch.cat((edge_fea1, edge_fea2), 0))
        
        drug_edge_feat = torch.stack(drug_edge_feat, 0)  # drug_num,2,128
        cline_edge_feat = torch.stack(cline_edge_feat, 0)  # cline_num,2,128
        edge_feat = torch.cat([drug_edge_feat, cline_edge_feat], 0)
        edge_feat = self.fc(edge_feat)
        
        node_m_projected = self.node_project(feature)  # 261, 256
        edge_m_projected_d = self.edge_project(drug_edge_feat, 0, 1)  # 76,128
        edge_m_projected_c = self.edge_project(cline_edge_feat, 2, 3)  # 64,128
        edge_m_projected = torch.cat([edge_m_projected_d, edge_m_projected_c], 0)

        node_multi = node_m_projected.reshape(self.num_node, 1, self.num_head, int(self.num_fea / self.num_head))  # 261,1,4,64
        edge_multi = edge_m_projected.reshape(self.num_node, 2, self.num_head,
                                              int(self.num_fea / self.num_head))  # 261,3,4,64

        edge_multi_in = edge_multi.permute(0, 2, 3, 1)  # 261,4,64,3
        node_multi_in = node_multi.permute(0, 2, 1, 3) / math.sqrt(  # 261,4,1,64
            int(self.num_fea / self.num_head))
        all_r_weight = self.act_1(torch.matmul(node_multi_in, edge_multi_in)).permute(0, 3, 1, 2)  # 261,3,4,1

        return all_r_weight, edge_feat


class HHNodeMP(nn.Module):
    def __init__(self, drop_rate):
        super(HHNodeMP, self).__init__()
        self.num_node = drug_num + cline_num
        self.num_fea = 128
        self.num_head = 4
        self.drop = nn.Dropout(drop_rate)
        self.act = nn.ReLU(inplace=True)

    def forward(self, edge_fea, all_r_weight):
        # print(edge_fea.shape)
        edge_fea_mp = edge_fea.reshape(self.num_node, 2, self.num_head, int(self.num_fea / self.num_head))  # 261,3,4,64
        edge_fea_weighted = torch.mul(all_r_weight, edge_fea_mp)  # 261,3,4,64

        node_rep = self.drop(self.act(edge_fea_weighted.reshape(self.num_node, 2, self.num_fea)))  # 261,3,256
        node_rep = torch.sum(node_rep, dim=1)
        return node_rep


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.num_fea = 128
        self.predict = nn.Linear(self.num_fea, self.num_cat)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        self.sigma = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, node_emb):
        n_cat = self.predict(node_emb)
        n_cat = self.sigma(n_cat)
        n_cat = self.softmax(n_cat).squeeze()
        return n_cat


class HHGNN(nn.Module):
    def __init__(self, num_cls):
        super(HHGNN, self).__init__()        
        # self.bio_encoder = BioEncoder(dim_drug=75, dim_cellline=651, output=128)
        self.dEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(drug_num, 128)))
        self.cEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(cline_num, 128)))
        self.weight_node = Weight_generator(2)
        self.mheadweight = MultiheadWeight(num_cls)
        self.hhnodemp = HHNodeMP(0.3)
        self.decoder = Decoder(128*3)
        # self.drug_rec_weight = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(128, 128)))
        # self.cline_rec_weight = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(128, 128)))

    def forward(self, drug_data, gexpr_data, DTH_cls, DTH_dc, hyperedge, input_weight, index):
        # feature = self.theta(feature)
        # linear_selected, recon_loss = self.hhedgecons(feature, data_stat['mask'], data_stat['node_multi_mask'])  # 261,3,1,261
        # edge_fea = self.hhedgemp(feature, linear_selected)  # 261, 3, 256
        # all_r_weight = self.mheadweight(feature, edge_fea, data_stat['node_multi_mask'])
        # node_rep = self.hhnodemp(edge_fea, all_r_weight)
        # predict = self.pred(node_rep[node_idx])
        
        # drug_embed, cellline_embed = self.bio_encoder(drug_data, gexpr_data)
        drug_embed, cellline_embed = self.dEmbeds, self.cEmbeds
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        # weight = self.weight_node(input_weight)
        # merge_embed = merge_embed * weight
        edge_fea_cls = DTH_cls @ merge_embed  # (num_hyperedge,70)@(70,feat)
        edge_fea_dc = DTH_dc @ merge_embed
        edge_fea = torch.cat([edge_fea_cls, edge_fea_dc], 0)
        all_r_weight, edge_feat = self.mheadweight(merge_embed, edge_fea, hyperedge)  # 70,2,4,1
        node_rep = self.hhnodemp(edge_feat, all_r_weight)
        res, node_rep = self.decoder(node_rep, index)
        
        # rec_drug = torch.sigmoid(torch.mm(torch.mm(node_rep[:drug_num], self.drug_rec_weight), node_rep[:drug_num].t()))
        # rec_cline = torch.sigmoid(torch.mm(torch.mm(node_rep[drug_num:], self.cline_rec_weight), node_rep[drug_num:].t()))
        # return res, rec_drug, rec_cline
        
        return res, node_rep

    
class Weight_generator(nn.Module):
    def __init__(self, in_channels):
        super(Weight_generator, self).__init__()
        self.fc1 = nn.Linear(in_channels, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)
        self.dropout = nn.Dropout(0.25)
        self.act = nn.ReLU()

    def forward(self, input_weight):
        out = self.act(self.fc1(input_weight))
        out = self.dropout(out)
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        return F.sigmoid(out)
    
    
class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        # -------drug_layer
        self.use_GMP = use_GMP
        self.conv1 = GCNConv(dim_drug, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)
        # -------cell line_layer
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.reset_para()
        self.act = nn.ReLU()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_data, gexpr_data):
        # -----drug_train
        drug_feature, drug_adj, ibatch = drug_data.x, drug_data.edge_index, drug_data.batch
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.act(x_drug)
        x_drug = self.batch_conv2(x_drug)
        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        # ----cellline_train
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline
    
    
class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(in_channels, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 1)
        self.batch1 = nn.BatchNorm1d(384)
        self.act = nn.LeakyReLU(negative_slope=0.5)
        
        # self.lin1 = nn.Linear(384, 384)
        # self.lin2 = nn.Linear(384, 192)
        # self.lin3 = nn.Linear(192, 1)
        
    def forward(self, graph_embed, index):
        embeds = torch.cat((graph_embed[index[:, 0], :], graph_embed[index[:, 1], :], graph_embed[index[:, 2], :]), 1)
        embeds = self.batch1(embeds)
        embeds = self.act(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.3, training=self.training)
        embeds = self.act(self.lin2(embeds))
        # embeds = F.dropout(embeds, p=0.3, training=self.training)
        ret = self.lin3(embeds)
        return torch.sigmoid(ret.squeeze(dim=1)), embeds