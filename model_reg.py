import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
from utils import reset

# drug_num = 87
# cline_num = 55
drug_num = 38
cline_num = 32


class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(in_channels, 256)
        self.batch1 = nn.BatchNorm1d(256)  # TODO HGNN 增加两层batch norm
        self.conv2 = HypergraphConv(256, out_channels)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge):
        x = self.act(self.conv1(x, edge))
        x = self.batch1(x)
        x = self.drop_out(x)
        x = self.act(self.conv2(x, edge))
        return x


class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        # -------drug_layer
        self.use_GMP = use_GMP
        self.conv1 = GCNConv(dim_drug, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)  # todo 新增一个batch norm
        # -------cell line_layer
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.ReLU()

    def forward(self, drug_data, gexpr_data):
        # -----drug_train
        drug_feature, drug_adj, ibatch = drug_data.x, drug_data.edge_index, drug_data.batch
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.drop_out(x_drug)
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.batch_conv2(self.act(x_drug))
        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        # ----cellline_train
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.drop_out(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, graph_embed, index):
        h = torch.cat((graph_embed[index[:, 0], :], graph_embed[index[:, 1], :], graph_embed[index[:, 2], :]), 1)
        h = self.act(self.fc1(h))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return h.squeeze(dim=1)


class HyperGraphSynergy(torch.nn.Module):
    def __init__(self):
        super(HyperGraphSynergy, self).__init__()
        self.bio_encoder = BioEncoder(dim_drug=75, dim_cellline=651, output=100)
        self.graph_encoder = HgnnEncoder(in_channels=100, out_channels=256)
        self.decoder = Decoder(in_channels=768)
        self.drug_rec_weight = nn.Parameter(torch.rand(256, 256))
        self.cline_rec_weight = nn.Parameter(torch.rand(256, 256))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.decoder)

    def forward(self, drug_data, gexpr_data, adj, index):
        drug_embed, cellline_embed = self.bio_encoder(drug_data, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)
        drug_emb, cline_emb = graph_embed[:drug_num], graph_embed[drug_num:]
        rec_drug = torch.sigmoid(torch.mm(torch.mm(drug_emb, self.drug_rec_weight), drug_emb.t()))
        rec_cline = torch.sigmoid(torch.mm(torch.mm(cline_emb, self.cline_rec_weight), cline_emb.t()))
        res = self.decoder(graph_embed, index)
        return res, rec_drug, rec_cline
