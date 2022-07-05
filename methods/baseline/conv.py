"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
import torch
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair





class slotGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.,
                 num_ntype=None,n_type_mappings=False,res_n_type_mappings=False,etype_specified_attention=False,eindexer=None,inputhead=False,semantic_trans="False",semantic_trans_normalize="row",attention_average="False",attention_mse_sampling_factor=0,attention_mse_weight_factor=0,attention_1_type_bigger_constraint=0,attention_0_type_bigger_constraint=0,slot_attention="False",relevant_passing="False"):
        super(slotGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats) if edge_feats else None
        self.n_type_mappings=n_type_mappings
        self.res_n_type_mappings=res_n_type_mappings
        self.etype_specified_attention=etype_specified_attention
        self.eindexer=eindexer
        self.num_ntype=num_ntype 
        self.semantic_transition_matrix=nn.Parameter(th.Tensor(self.num_ntype , self.num_ntype))
        self.semantic_trans=semantic_trans
        self.semantic_trans_normalize=semantic_trans_normalize
        self.attentions=None
        self.attention_average=attention_average
        self.attention_mse_sampling_factor=attention_mse_sampling_factor
        self.attention_mse_weight_factor=attention_mse_weight_factor
        self.attention_1_type_bigger_constraint=attention_1_type_bigger_constraint
        self.attention_0_type_bigger_constraint=attention_0_type_bigger_constraint
        self.slot_attention=slot_attention
        self.relevant_passing=relevant_passing

        if isinstance(in_feats, tuple):
            raise Exception("!!!")
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            if not n_type_mappings:
                self.fc = nn.Parameter(th.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
            """else:
                self.fc =nn.ModuleList([nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)  for _ in range(num_ntype)] )
                raise Exception("!!!")"""
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False) if edge_feats else None
        if self.etype_specified_attention:
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats,num_etypes)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats,num_etypes)))
        elif self.slot_attention=="True":
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads,self.num_ntype, out_feats)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads,self.num_ntype, out_feats)))
            self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads,1, edge_feats))) if edge_feats else None
        else:
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats   *self.num_ntype)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats*self.num_ntype)))
            self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats))) if edge_feats else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                if not self.res_n_type_mappings:
                    self.res_fc =nn.Parameter(th.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
                    """self.res_fc = nn.Linear(
                        self._in_dst_feats, num_heads * out_feats, bias=False)"""
                else:
                    raise NotImplementedError()
                    self.res_fc =nn.ModuleList([nn.Linear(
                        self._in_dst_feats, num_heads * out_feats, bias=False)  for _ in range(num_ntype)] )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            raise NotImplementedError()
            if not self.n_type_mappings:
                self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
            else:
                self.bias_param=nn.ModuleList([ nn.Parameter(th.zeros((1, num_heads, out_feats)))  for _ in range(num_ntype) ])
        self.alpha = alpha
        self.inputhead=inputhead

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc, gain=gain)
            """if self.n_type_mappings:
                for m in self.fc:
                    nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.fc.weight, gain=gain)"""
        else:
            raise Exception("!!!")
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if not self.etype_specified_attention and self._edge_feats:
            nn.init.xavier_normal_(self.attn_e, gain=gain) 
        if isinstance(self.res_fc, nn.Linear):
            if self.res_n_type_mappings:
                for m in self.res_fc:
                    nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        elif isinstance(self.res_fc, Identity):
            pass
        elif isinstance(self.res_fc, nn.Parameter):
            nn.init.xavier_normal_(self.res_fc, gain=gain)
        if self._edge_feats:
            nn.init.xavier_normal_(self.fc_e.weight, gain=gain) 

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat,get_out=[""], res_attn=None):
        with graph.local_scope():
            node_idx_by_ntype=graph.node_idx_by_ntype
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                raise Exception("!!!")
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                if self.semantic_trans=="True":
                    if self.semantic_trans_normalize=="row":
                        dim_flag=1
                    elif self.semantic_trans_normalize=="col":
                        dim_flag=0
                    st_m= F.softmax( self.semantic_transition_matrix,dim=dim_flag ).unsqueeze(0).unsqueeze(0)
                    #st_m= F.softmax( torch.randn_like(self.semantic_transition_matrix),dim=dim_flag ).unsqueeze(0).unsqueeze(0)# ruin exp!!! 
                    #st_m= torch.zeros_like(self.semantic_transition_matrix).unsqueeze(0).unsqueeze(0)# ruin exp!!! 
                elif self.semantic_trans=="False":
                    st_m=torch.eye(self.num_ntype).to(self.semantic_transition_matrix.device)
                #feature transformation first
                h_src = h_dst = self.feat_drop(feat)   #num_nodes*(num_ntype*input_dim)
                if self.n_type_mappings:
                    raise Exception("!!!")
                    h_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        h_new.append(self.fc[type_count](h_src[idx,:]).view(
                        -1, self._num_heads, self._out_feats))
                    feat_src = feat_dst = torch.cat(h_new, 0)

                else:
                    if self.inputhead:
                        h_src=h_src.view(-1,1,self.num_ntype,self._in_src_feats)
                        h_src=torch.matmul(st_m,h_src)
                    else:
                        h_src=h_src.view(-1,self._num_heads,self.num_ntype,int(self._in_src_feats/self._num_heads))
                        h_src=torch.matmul(st_m,h_src)
                    h_dst=h_src=h_src.permute(2,0,1,3).flatten(2)  #num_ntype*num_nodes*(in_feat_dim)
                    if "getEmb" in get_out:
                        self.emb=h_dst.cpu().detach()
                    #self.fc with num_ntype*(in_feat_dim)*(out_feats * num_heads)
                    
                    feat_dst = torch.bmm(h_src,self.fc)  #num_ntype*num_nodes*(out_feats * num_heads)
                    
                    feat_src = feat_dst =feat_dst.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
                            -1,self.num_ntype ,self._num_heads, self._out_feats).permute(0,2,1,3).flatten(2)
                    



                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
        
            if self.etype_specified_attention:
                el = (feat_src.unsqueeze(-1) * self.attn_l).sum(dim=2).unsqueeze(2) #num_nodes*heads*dim*num_etype   1*heads*dim*1   
                er = (feat_dst.unsqueeze(-1) * self.attn_r).sum(dim=2).unsqueeze(2)
                graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.dstdata.update({'er': er})
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))  #  num_edges*heads*1*num_etype
                e=self.leaky_relu((graph.edata.pop('e')*self.eindexer).sum(-1))
            else:
                e_feat = self.edge_emb(e_feat) if self._edge_feats else None
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)  if self._edge_feats else None
                
                if self.slot_attention=="True":
                    feat_src=feat_src.view(-1, self._num_heads,self.num_ntype, self._out_feats)
                    feat_dst=feat_src.view(-1, self._num_heads,self.num_ntype, self._out_feats)
                    e_feat = e_feat.unsqueeze(2)  if self._edge_feats else None
                    if self.relevant_passing=="True":
                        node_self_slot_flag= graph.node_ntype_indexer   #self slot
                        graph.srcdata.update({"self_slot_indexer_in":node_self_slot_flag})
                        graph.dstdata.update({"self_slot_indexer_out":node_self_slot_flag})
                        graph.apply_edges(fn.u_add_v('self_slot_indexer_in', 'self_slot_indexer_out', 'relevant_slot_flag'))
                        relevant_slot_flag=graph.edata.pop('relevant_slot_flag')
                        relevant_slot_flag=relevant_slot_flag.unsqueeze(1).unsqueeze(-1)
                        relevant_slot_flag=(relevant_slot_flag>0).int()
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1) if self._edge_feats else 0  #(-1, self._num_heads, 1) 
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.dstdata.update({'er': er})
                graph.edata.update({'ee': ee}) if self._edge_feats else None
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e_=graph.edata.pop('e')
                ee=graph.edata.pop('ee') if self._edge_feats else 0
                e=e_+ee
                
                e = self.leaky_relu(e)
            # compute softmax
            a=self.attn_drop(edge_softmax(graph, e))
            """if self.attention_average=="True" or self.attention_mse_sampling_factor>0:
                graph.apply_edges(fn.u_add_v('er', 'el', 'e_reverse'))
                e_reverse=graph.edata.pop('e_reverse')
                e_reverse=e_reverse+ee
                e_reverse = self.leaky_relu(e_reverse)
                a_reverse=self.attn_drop(edge_softmax(graph, e_reverse,norm_by='src'))
                cor_vec=torch.stack([a[graph.etype_ids[1]].flatten(),a_reverse[graph.etype_ids[1]].flatten()],dim=0)
                self.attn_correlation=np.corrcoef(cor_vec.detach().cpu())[0,1]
            if self.attention_average=="True":
                

                a[graph.etype_ids[1]]=(a[graph.etype_ids[1]]+a_reverse[graph.etype_ids[1]])/2
            if self.attention_mse_sampling_factor>0:
                #choosed_nodes_indices= torch.randperm(len(graph.etype_ids[1]))[:int(len(graph.etype_ids[1])*self.attention_mse_sampling_factor)]
                choosed_edges=np.random.choice(graph.etype_ids[1],int(len(graph.etype_ids[1])*self.attention_mse_sampling_factor), replace=False)
                #choosed_nodes=graph.etype_ids[1][choosed_nodes_indices]
                mse=torch.mean((a[choosed_edges]-a_reverse[choosed_edges])**2,dim=[0,1,2])
                self.mse=mse
            else:
                self.mse=torch.tensor(0)
            if self.attention_1_type_bigger_constraint>0:
                self.t1_bigger_mse=torch.mean((1-a[graph.etype_ids[1]])**2,dim=[0,1,2])
            else:
                self.t1_bigger_mse=torch.tensor(0)
            if self.attention_0_type_bigger_constraint>0:
                self.t0_bigger_mse=torch.mean((1-a[graph.etype_ids[0]])**2,dim=[0,1,2])
            else:
                self.t0_bigger_mse=torch.tensor(0)"""
            
            #print("a mean",[round(a[graph.etype_ids[i]].mean().item(),3) for i in range(7)],"a_reverse mean",[round(a_reverse[graph.etype_ids[i]].mean().item(),3) for i in range(7)])
            #graph.edata['a'] = a
            if res_attn is not None:
                a=a * (1-self.alpha) + res_attn * self.alpha
            if self.relevant_passing=="True":
                degs_in = graph.in_degrees().float().clamp(min=1)
                norm_in = th.pow(degs_in, -0.5)
                degs_out = graph.out_degrees().float().clamp(min=1)
                norm_out = th.pow(degs_out, -0.5)
                graph.srcdata.update({'norm_in': norm_in})
                graph.dstdata.update({'norm_out': norm_out})
                graph.apply_edges(fn.u_mul_v('norm_in', 'norm_out', 'gcn_passing'))
                gcn_passing=graph.edata.pop('gcn_passing').unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                
                a=a*relevant_slot_flag+gcn_passing*(1-relevant_slot_flag)

            graph.edata['a'] = a
            # then message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
                             
            rst = graph.dstdata['ft']
            if self.slot_attention=="True":
                rst=rst.flatten(2)
            # residual
            if self.res_fc is not None:
                if not self.res_n_type_mappings:
                    if self._in_dst_feats != self._out_feats:
                        resval =torch.bmm(h_src,self.res_fc)
                        resval =resval.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
                            -1,self.num_ntype ,self._num_heads, self._out_feats).permute(0,2,1,3).flatten(2)
                        #resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                    else:
                        resval = self.res_fc(h_src).view(h_dst.shape[0], -1, self._out_feats*self.num_ntype)  #Identity
                else:
                    raise NotImplementedError()
                    res_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        res_new.append(self.res_fc[type_count](h_dst[idx,:]).view(
                        h_dst[idx,:].shape[0], -1, self._out_feats))
                    resval = torch.cat(res_new, 0)
                rst = rst + resval
            # bias
            if self.bias:
                if self.n_type_mappings:
                    rst_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        rst_new.append(        rst[idx]+ self.bias_param[type_count]    )
                    rst = torch.cat(rst_new, 0)
                else:

                    rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            self.attentions=graph.edata.pop('a').detach()
            torch.cuda.empty_cache()
            return rst, self.attentions










# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()

