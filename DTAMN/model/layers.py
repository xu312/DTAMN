import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import (
    dense_to_sparse,
    get_laplacian,
    to_dense_adj,
    to_scipy_sparse_matrix,
)

try:
    from torch_geometric.utils import scatter
except ImportError:
    from torch_scatter import scatter


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps."""
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    if num_nodes is None:
        if edge_index.numel() == 0:
            num_nodes = 0
        else:
            num_nodes = int(edge_index.max().item()) + 1
    source = edge_index[0]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)

    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)
    return rw_landing


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """Implement different eigenvector normalizations."""
    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)
    elif normalization == "L2":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
    elif normalization == "abs-max":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
    elif normalization == "wavelength":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom * 2 / np.pi
    elif normalization == "wavelength-asin":
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = eigval_denom
    elif normalization == "wavelength-soft":
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom
    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom
    return EigVecs


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the graph."""
    num_nodes = len(evals)
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if num_nodes < max_freqs:
        eig_vecs = F.pad(evects, (0, max_freqs - num_nodes), value=float('nan'))
    else:
        eig_vecs = evects

    if num_nodes < max_freqs:
        eig_vals = F.pad(evals, (0, max_freqs - num_nodes), value=float('nan')).unsqueeze(0)
    else:
        eig_vals = evals.unsqueeze(0)
    eig_vals = eig_vals.repeat(num_nodes, 1).unsqueeze(2)

    return eig_vals, eig_vecs

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, num_node, kernel_size, conv_type):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.conv_type = conv_type
        
        if conv_type =='conv':
            if kernel_size == 1:
                self.spatial_proj = nn.Conv2d(num_node, num_node, kernel_size=kernel_size)
            else:
                self.spatial_proj = nn.Conv2d(
                    num_node, num_node, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)
                )    
        elif conv_type == 'mpnn':
            self.spatial_proj = GCNConv(d_ffn, d_ffn)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        
    def forward(self, x, residual=None, cirmat=None):
        # x= [B,T,N,D]
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        if self.conv_type == 'conv':
            x = self.spatial_proj(v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        elif self.conv_type == 'mpnn':
            edge_index = dense_to_sparse(torch.Tensor(cirmat))[0] 
            x = self.spatial_proj(v, edge_index)
                
        if residual is not None:
            out = (x + residual) * u
        else:
            out = u * x
        return out


class Cy2MixerBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_node, kernel_size, conv_type):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, num_node, kernel_size, conv_type)
        self.channel_proj_tinyatt = nn.Conv2d(d_ffn, 3 * d_ffn, kernel_size=1)
        self.out_proj_tinyatt = nn.Conv2d(d_ffn, d_ffn, kernel_size=1)

    def forward(self, x, residual=None, cirmat=None, use_tinyatt=False):
        residual = x
        x = self.norm(x)
        if use_tinyatt:
            qkv = self.channel_proj_tinyatt(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # [B,  N, T, 3*D]
            q, k, v = qkv.chunk(3, dim= -1) #[B, N, T, D] * 3
            w = q @ k.transpose(-1, -2)
            w = w.softmax(dim=-1)
            gate_res = w @ v
            gate_res = self.out_proj_tinyatt(gate_res.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        x = F.gelu(self.channel_proj1(x))
        if use_tinyatt:
            x = self.sgu(x, residual=gate_res, cirmat=cirmat)
        else:
            x = self.sgu(x, residual=None, cirmat=cirmat)
        x = self.channel_proj2(x)
        out = x + residual
        return out

rwse_cfg = {
    'hid': 16,
    'pe_dim' :28,
    'rwse_dim' : 20,
    'emb_dim' : 36
}

class GCN_RWSE(torch.nn.Module):
    def __init__(self, d_model, d_ffn, num_node, cfg):
        super().__init__()
        self.raw_norm = nn.BatchNorm1d(d_model)
        self.pe_encoder = nn.Linear(d_model, d_model)
        # self.conv1 = GCNConv(cfg['emb_dim'] + cfg['pe_dim'], cfg['hid'])
        self.conv1 = GCNConv(d_model * 2, d_ffn)
        self.conv2 = GCNConv(d_ffn, d_model)
        
    def forward(self, x, cirmat=None):
        B, T, N, D = x.shape
        time_list = [i for i in range(1, x.shape[-1]+1)]
        edge_index = dense_to_sparse(torch.Tensor(cirmat))[0].detach().cpu()
        pos_enc = get_rw_landing_probs(
            ksteps=time_list,
            edge_index=edge_index,
            num_nodes=x.shape[2]
        ).to(x.device)
        # x = self.feat_encoder(x[:, 0].long())
        pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)
        pos_enc = pos_enc.unsqueeze(0).unsqueeze(0)
        pos_enc = pos_enc.expand(B, T, N, D)

        x = torch.cat((x, pos_enc), dim=-1) # [N, hid + rwse_dim]
        # x = x + pos_enc # [N, hid + rwse_dim]
        edge_index = edge_index.to(x.device)
        x = self.conv1(x, edge_index) # [N, hid + rwse_dim] -> [N, hid]
        x = F.relu(x) 
        output = self.conv2(x, edge_index) #[N, hid] -> [N, 7]
        return output

lappe_cfg = {
    'hid': 16,
    # 'pe_dim' :28,
    'dim_pe' :28,
    'rwse_dim' : 20,
    'emb_dim' : 36,
    'training' : True,
    'raw_norm' : None,
    'n_layers' : 2,
    'post_mlp' : None,
    'expand_x' : False,
}


class LapPENodeEncoder(torch.nn.Module):
    def __init__(self, d_model, cfg):
        super().__init__()
        self.training = cfg['training']
        self.raw_norm = cfg['raw_norm']
        self.n_layers = cfg['n_layers']
        # self.dim_pe = cfg['dim_pe']
        self.dim_pe = d_model
        self.post_mlp = cfg['post_mlp']
        self.expand_x = cfg['expand_x']
        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        # DeepSet model for LapPE
        layers = []
        if self.n_layers == 1:
            layers.append(activation())
        else:
            self.linear_A = nn.Linear(2, 2 * self.dim_pe)
            layers.append(activation())
            for _ in range(self.n_layers - 2):
                layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                layers.append(activation())
            layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
            layers.append(activation())
        self.pe_encoder = nn.Sequential(*layers)

    
    def forward(self, x, EigVals, EigVecs):
        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        # if self.raw_norm:
        #     pos_enc = raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe
        pos_enc = self.pe_encoder(pos_enc)

        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        # if self.expand_x:
        #     h = linear_x(x)
        # else:
        #     h = x
        # h = x

        # data.x = torch.cat((h, pos_enc), 1)
        # x = torch.cat((h, pos_enc), 1)
        # x = x + pos_enc
        # return x
        return pos_enc

        
class GCN_LapPE(torch.nn.Module):
    def __init__(self, d_model, d_ffn, cfg):
        super().__init__()
        self.lap_encoder = LapPENodeEncoder(d_model, cfg)
        self.conv1 = GCNConv(d_model * 2, d_ffn)
        # self.conv1 = GCNConv(cfg['emb_dim'] + cfg['dim_pe'], cfg['hid'])
        self.conv2 = GCNConv(d_ffn, d_ffn)
        self.raw_norm = nn.BatchNorm1d(d_model)

    def forward(self, x, cirmat=None):
        B, T, N, D = x.shape
        edge_index = dense_to_sparse(torch.Tensor(cirmat))[0].detach().cpu()
        L = to_scipy_sparse_matrix(
            *get_laplacian(edge_index, normalization=None,num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())
        EigVals, EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=1,
            eigvec_norm="L2")
        EigVals, EigVecs = EigVals.to(x.device), EigVecs.to(x.device)
        #Laplacian PE
        pos_enc = self.lap_encoder(x, EigVals, EigVecs)
        pos_enc = pos_enc.unsqueeze(0).unsqueeze(0)
        pos_enc = pos_enc.expand(B, T, N, D)
        x = torch.cat((x, pos_enc), dim=-1) # [N, hid + rwse_dim]
        edge_index = edge_index.to(x.device)
        x = self.conv1(x, edge_index) # [N, hid + rwse_dim] -> [N, hid]
        x = F.relu(x) 
        output = self.conv2(x, edge_index) #[N, hid] -> [N, 7]
        return output

class Cy2Mixer_layer(nn.Module):
    def __init__(
        self, d_model, d_ffn, seq_len, num_node, output_dim=1, bias=False, use_tinyatt = False,
        dropout = 0.1, tgu_kernel_size = 3, pe = None
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.seq_len = seq_len
        self.num_node = num_node
        self.output_dim = output_dim
        self.use_tinyatt = use_tinyatt
        self.tgu_kernel_size = tgu_kernel_size
        
        
        self.tgu = Cy2MixerBlock(self.d_model, self.d_ffn, self.seq_len, kernel_size= self.tgu_kernel_size, conv_type = 'conv')
        self.sgu = Cy2MixerBlock(self.d_model, self.d_ffn, self.num_node, kernel_size=1, conv_type = 'mpnn')
        if pe == "RWSE":
            self.cgu = GCN_RWSE(self.d_model, self.d_ffn, self.num_node, rwse_cfg)
        elif pe == "LapPE":
            self.cgu = GCN_LapPE(self.d_model, self.d_ffn, lappe_cfg)
        else:
            self.cgu = Cy2MixerBlock(self.d_model, self.d_ffn, self.num_node, kernel_size=1, conv_type = 'mpnn')
            
        self.channel_proj_out = nn.Conv2d(3 * self.d_model, self.d_model, kernel_size=1)
        # self.channel_proj_out = nn.Conv2d(2 * self.d_model, self.d_model, kernel_size=1)
            
        
    def forward(
        self,
        x,
        cirmat_A=None,
        cirmat_C=None,
    ):
        # x=[B,T,N,D]
        residual = x
        x_t = self.tgu(x.permute(0, 2, 1, 3), use_tinyatt=self.use_tinyatt).permute(0, 2, 1, 3)
        x_t = self.norm1(x_t)
        x_t = self.dropout1(x_t)
        
        x_s = self.sgu(x, residual=residual, cirmat=cirmat_A, use_tinyatt=self.use_tinyatt)
        x_s = self.norm2(x_s)
        x_s = self.dropout2(x_s)
        
        # x_c = self.cgu(x, residual=residual, cirmat=cirmat_C, use_tinyatt=self.use_tinyatt)
        x_c = self.cgu(x, cirmat=cirmat_A)
        x_c = self.norm3(x_c)
        x_c = self.dropout3(x_c)
          
        # x=[B,T,N,3*D]
        x = torch.cat([x_t, x_s, x_c], dim=-1)
        # x = torch.cat([x_t, x_s], dim=-1)
        x = self.channel_proj_out(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) # x=[B,T,N,D]
        # x = self.norm(x)
        out = x + residual
        return out

