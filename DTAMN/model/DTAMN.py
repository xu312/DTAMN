import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import dense_to_sparse, get_laplacian, to_scipy_sparse_matrix

from model.layers import LapPENodeEncoder, get_lap_decomp_stats


class MPL(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(MPL, self).__init__()

        self.cheb_k = 1
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.weights_pool_s = nn.Parameter(torch.FloatTensor(embed_dim, self.cheb_k, dim_in, dim_out))
        self.bias_pool_s = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool_s)
        nn.init.xavier_normal_(self.bias_pool_s)

        self.weights_pool_t = nn.Parameter(torch.FloatTensor(time_dim, self.cheb_k, dim_in, dim_out))
        self.bias_pool_t = nn.Parameter(torch.FloatTensor(time_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool_t)
        nn.init.xavier_normal_(self.bias_pool_t)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, node_embeddings):
        emb_t = node_embeddings[0]
        emb_s = node_embeddings[1]

        x_g = x.unsqueeze(2)

        weights_s = torch.einsum('nd,dkio->nkio', emb_s, self.weights_pool_s)
        bias_s = torch.matmul(emb_s, self.bias_pool_s)
        x_gconv_s = torch.einsum('bnki,nkio->bno', x_g, weights_s) + bias_s

        if emb_t.dim() == 2:
            weights_t = torch.einsum('bd,dkio->bkio', emb_t, self.weights_pool_t)
            x_gconv_t = torch.einsum('bnki,bkio->bno', x_g, weights_t)
            bias_t = torch.matmul(emb_t, self.bias_pool_t).unsqueeze(1)
        elif emb_t.dim() == 3:
            weights_t = torch.einsum('bnd,dkio->bnkio', emb_t, self.weights_pool_t)
            x_gconv_t = torch.einsum('bnki,bnkio->bno', x_g, weights_t)
            bias_t = torch.matmul(emb_t, self.bias_pool_t)
        else:
            raise ValueError(f"Unexpected shape for emb_t: {emb_t.shape}")

        x_gconv_t = x_gconv_t + bias_t
        x_gconv = x_gconv_s + x_gconv_t
        return x_gconv


class DTMGRU(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(DTMGRU, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = MPL(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, time_dim)
        self.update = MPL(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, time_dim)

    def forward(self, x, state, node_embeddings):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


# -----------------------------------------------------------------
# 1. Encoder
# -----------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DPMGRU layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.hidden_dim = dim_out
        self.num_layers = num_layers
        self.PM_cells_conn = nn.ModuleList()
        self.PM_cells_topo = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()

        self.PM_cells_conn.append(DTMGRU(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        self.PM_cells_topo.append(DTMGRU(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        self.fusion_layers.append(nn.Sequential(nn.Linear(dim_out * 2, dim_out), nn.ReLU()))

        for _ in range(1, num_layers):
            self.PM_cells_conn.append(DTMGRU(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim))
            self.PM_cells_topo.append(DTMGRU(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim))
            self.fusion_layers.append(nn.Sequential(nn.Linear(dim_out * 2, dim_out), nn.ReLU()))

    def forward(self, x, init_state_conn, init_state_topo, node_embeddings_conn, node_embeddings_topo):
        assert x.shape[2] == self.node_num
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden_conn = []
        output_hidden_topo = []

        for i in range(self.num_layers):
            state_conn = init_state_conn[i]
            state_topo = init_state_topo[i]
            inner_states_fused = []

            for t in range(seq_length):
                current_input_t = current_inputs[:, t, :, :]

                # Path 1: Conn
                time_emb_t = node_embeddings_conn[0][:, t, :]
                static_emb_conn = node_embeddings_conn[1]
                h_conn_t = self.PM_cells_conn[i](current_input_t, state_conn, [time_emb_t, static_emb_conn])

                # Path 2: Topo
                time_emb_t_topo = node_embeddings_topo[0][:, t, :]
                static_emb_topo = node_embeddings_topo[1]
                h_topo_t = self.PM_cells_topo[i](current_input_t, state_topo, [time_emb_t_topo, static_emb_topo])

                state_conn = h_conn_t
                state_topo = h_topo_t

                # Fusion per step
                h_fused_t = self.fusion_layers[i](torch.cat([h_conn_t, h_topo_t], dim=-1))
                inner_states_fused.append(h_fused_t)

            output_hidden_conn.append(state_conn)
            output_hidden_topo.append(state_topo)
            current_inputs = torch.stack(inner_states_fused, dim=1)

        return current_inputs, (torch.stack(output_hidden_conn, dim=0), torch.stack(output_hidden_topo, dim=0))

    def init_hidden(self, batch_size):
        init_states_conn = []
        init_states_topo = []
        for i in range(self.num_layers):
            init_states_conn.append(self.PM_cells_conn[i].init_hidden_state(batch_size))
            init_states_topo.append(self.PM_cells_topo[i].init_hidden_state(batch_size))
        return torch.stack(init_states_conn, dim=0), torch.stack(init_states_topo, dim=0)


# -----------------------------------------------------------------
# 2. Decoder
# -----------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PM_cells = nn.ModuleList()
        self.PM_cells.append(DTMGRU(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.PM_cells.append(DTMGRU(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D)
        # init_state: list of states for each layer
        assert xt.shape[1] == self.node_num
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            # node_embeddings passed as [time_emb, static_emb]
            state = self.PM_cells[i](current_inputs, init_state[i], node_embeddings)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden  # current_inputs is final hidden state


# -----------------------------------------------------------------

# -----------------------------------------------------------------
class Recollection_Decoder(nn.Module):
    def __init__(self, args):
        super(Recollection_Decoder, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim  
        self.hidden_dim = args.rnn_units
        self.horizon = 12  

        
        self.decoder = Decoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                  args.embed_dim, args.time_dim, num_layers=1)

        
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.input_dim, bias=True))

    def forward(self, source_shape, h_n, node_embedding_hist, fused_static_embeddings):
        
        
        

        batch_size = source_shape[0]
        hist_len = source_shape[1]  

        
        go = torch.zeros((batch_size, self.num_node, self.input_dim), device=h_n.device)

        
        
        ht_list = [h_n]

        out = []
        
        for t in range(hist_len):
            # Step-by-step decoding
            
            
            

            # node_embedding_hist[:, t, :] shape: [B, Dim]
            state, ht_list = self.decoder(go, ht_list, [node_embedding_hist[:, t, :], fused_static_embeddings])

            
            prediction = self.proj(state)  # [B, N, Input_Dim]
            out.append(prediction)

            # Autoregressive input for next step
            go = prediction

        output = torch.stack(out, dim=1)  # [B, T, N, Input_Dim]
        return output


# -----------------------------------------------------------------

# -----------------------------------------------------------------
class DTAMN(nn.Module):
    def __init__(self, args, adj_conn, adj_topo):
        super(DTAMN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim

        # LapPE Config
        self.lappe_cfg = {'training': True, 'raw_norm': None, 'n_layers': 2, 'post_mlp': None, 'expand_x': False}
        self.lap_pos_dim = args.embed_dim

        
        def compute_eigen_info(adj_numpy, k_freqs):
            adj_th = torch.from_numpy(adj_numpy).float()
            edge_index, _ = dense_to_sparse(adj_th)
            L = to_scipy_sparse_matrix(*get_laplacian(edge_index, normalization=None, num_nodes=self.num_node))
            try:
                vals, vecs = np.linalg.eigh(L.toarray())
            except:
                vals, vecs = np.linalg.eigh(L.toarray())
            return get_lap_decomp_stats(evals=vals, evects=vecs, max_freqs=k_freqs, eigvec_norm="L2")

        vals_conn, vecs_conn = compute_eigen_info(adj_conn, self.lap_pos_dim)
        vals_topo, vecs_topo = compute_eigen_info(adj_topo, self.lap_pos_dim)
        self.register_buffer('eig_vals_conn', vals_conn)
        self.register_buffer('eig_vecs_conn', vecs_conn)
        self.register_buffer('eig_vals_topo', vals_topo)
        self.register_buffer('eig_vecs_topo', vecs_topo)

        self.conn_lappe = LapPENodeEncoder(self.embed_dim, self.lappe_cfg)
        self.topo_lappe = LapPENodeEncoder(self.embed_dim, self.lappe_cfg)
        self.conn_projector = nn.Linear(self.num_node, args.embed_dim)
        self.topo_projector = nn.Linear(self.num_node, args.embed_dim)

        
        adj_matrix_conn = torch.from_numpy(adj_conn).float().to(args.device)
        adj_matrix_topo = torch.from_numpy(adj_topo).float().to(args.device)
        self.conn_projector = self.conn_projector.to(args.device)
        self.topo_projector = self.topo_projector.to(args.device)
        with torch.no_grad():
            init_conn_embed = self.conn_projector(adj_matrix_conn)
            init_topo_embed = self.topo_projector(adj_matrix_topo)
        self.napl_embeddings_conn = nn.Parameter(init_conn_embed, requires_grad=True)
        self.napl_embeddings_topo = nn.Parameter(init_topo_embed, requires_grad=True)
        self.norm_fusion = nn.LayerNorm(args.embed_dim)

        # Time Embeddings
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.time_dim))

        # -----------------------------------------------------------------
        
        # -----------------------------------------------------------------
        self.encoder = Encoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                   args.embed_dim, args.time_dim, args.num_layers)

        self.decoder_fusion_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.decoder_fusion_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU()
            ))

        self.static_embed_fusion_layer = nn.Sequential(
            nn.Linear(args.embed_dim * 2, args.embed_dim),
            nn.ReLU()
        )

        # Prediction Decoder (Main Task)
        self.decoder = Prediction_decoder(args)

        # -----------------------------------------------------------------
        
        
        # -----------------------------------------------------------------
        self.recollection_decoder = Recollection_Decoder(args)

    def forward(self, source, traget=None, batches_seen=None):
        
        t_i_d_data1 = source[..., 0, -2]
        d_i_w_data1 = source[..., 0, -1]
        t_i_d_data2 = traget[..., 0, -2]
        d_i_w_data2 = traget[..., 0, -1]

        T_i_D_emb1 = self.T_i_D_emb[(t_i_d_data1 * 288).long()]
        D_i_W_emb1 = self.D_i_W_emb[(d_i_w_data1).long()]
        T_i_D_emb2 = self.T_i_D_emb[(t_i_d_data2 * 288).long()]
        D_i_W_emb2 = self.D_i_W_emb[(d_i_w_data2).long()]

        node_embedding1 = torch.mul(T_i_D_emb1, D_i_W_emb1)  # [B, T_hist, N, Dim]
        node_embedding2 = torch.mul(T_i_D_emb2, D_i_W_emb2)  # [B, T_pred, N, Dim]

        # LapPE + NAPL Fusion
        lappe_conn = self.conn_lappe(None, self.eig_vals_conn, self.eig_vecs_conn)
        lappe_topo = self.topo_lappe(None, self.eig_vals_topo, self.eig_vecs_topo)
        final_static_conn = self.norm_fusion(self.napl_embeddings_conn + lappe_conn)
        final_static_topo = self.norm_fusion(self.napl_embeddings_topo + lappe_topo)

        en_node_embeddings_conn = [node_embedding1, final_static_conn]
        en_node_embeddings_topo = [node_embedding1, final_static_topo]

        
        source_input = source[..., :self.input_dim]
        init_state_conn, init_state_topo = self.encoder.init_hidden(source_input.shape[0])
        state, h_n_tuple = self.encoder(source_input, init_state_conn, init_state_topo,
                                        en_node_embeddings_conn, en_node_embeddings_topo)

        h_n_conn, h_n_topo = h_n_tuple  # [Layers, B, N, Hidden]

        
        h_n_fused_list = []
        for i in range(self.num_layers):
            h_n_fused_layer_i = self.decoder_fusion_layers[i](
                torch.cat([h_n_conn[i], h_n_topo[i]], dim=-1)
            )
            h_n_fused_list.append(h_n_fused_layer_i)

        
        cat_static_embeds = torch.cat([final_static_conn, final_static_topo], dim=-1)
        fused_static_embed_for_decoder = self.static_embed_fusion_layer(cat_static_embeds)

        # 3. Prediction Decoder (Main Task: Future Prediction)
        decoder_input_h_n = h_n_fused_list

        output = self.decoder(source_input, traget, decoder_input_h_n, node_embedding1, node_embedding2,
                              fused_static_embed_for_decoder, batches_seen)

        # -----------------------------------------------------------------
        
        
        # -----------------------------------------------------------------
        recollection_output = None

        
        if self.training:
            
            
            # h_n_fused_list[-1] shape: [B, N, Hidden]
            h_rec_init = h_n_fused_list[-1]

            
            
            
            recollection_output = self.recollection_decoder(
                source_input.shape,
                h_rec_init,
                node_embedding1,
                fused_static_embed_for_decoder
            )

            
            return output, recollection_output

        
        return output

class Prediction_decoder(nn.Module):
    def __init__(self, args=None):
        super(Prediction_decoder, self).__init__()
        # self.cl_decay_steps = args.lr_decay_step
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.teacher_forcing = args.teacher_forcing
        self.teacher_decay_step = args.teacher_decay_step
        self.decoder = Decoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                  args.embed_dim, args.time_dim, args.num_layers)
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))

    def forward(self, source, traget, h_n, node_embedding1, node_embedding2,
                fused_static_embeddings, batches_seen):  
        ht_list = h_n
        go = torch.zeros((source.shape[0], self.num_node, self.output_dim), device=source.device)
        out = []
        for t in range(self.horizon):
            
            state, ht_list = self.decoder(go, ht_list, [node_embedding2[:, t, :], fused_static_embeddings])  
            go = self.proj(state)
            out.append(go)
            if self.training and self.teacher_forcing:  
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):  
                    # go = traget[:, t, :, :self.input_dim].unsqueeze(-1)
                    go = traget[:, t, :, :self.input_dim]
        output = torch.stack(out, dim=1)

        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.teacher_decay_step / (
                self.teacher_decay_step + np.exp(batches_seen / self.teacher_decay_step))
        return x
