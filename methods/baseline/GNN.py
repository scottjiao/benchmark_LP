import torch
import torch as th
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv,slotGATConv

class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()

class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()
    def forward(self, left_emb, right_emb, r_id,slot_num=None,prod_aggr=None,sigmoid="after",logitsRescale="None"):
        if not prod_aggr:
            left_emb = torch.unsqueeze(left_emb, 1)
            right_emb = torch.unsqueeze(right_emb, 2)
            return torch.bmm(left_emb, right_emb).squeeze()
        else:
            left_emb = left_emb.view(-1,slot_num,int(left_emb.shape[1]/slot_num))
            right_emb = right_emb.view(-1,int(right_emb.shape[1]/slot_num),slot_num)
            x=torch.bmm(left_emb, right_emb)# num_sampled_edges* num_slot*num_slot
            if prod_aggr=="all":
                x=x.flatten(1)
                x=x.sum(1)
                return x
            x=torch.diagonal(x,0,1,2) # num_sampled_edges* num_slot
            if sigmoid=="before":
                x=F.sigmoid(x)
            
            if prod_aggr=="mean":
                x=x.mean(1)
                
            elif prod_aggr=="max":
                x=x.max(1)[0]
            elif prod_aggr=="sum":
                x=x.sum(1)
            else:
                raise Exception()
            if logitsRescale=="slotNum":
                x= x/(slot_num*2)+1/2
                if x.max()>1+3e-1 or x.min()<0-3e-1:
                    raise Exception()
                #print(f"max: {x.max()} min: {x.min()}")
                x=torch.clamp( x,0,1)
            return x


class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 decode='distmult',inProcessEmb="True",l2use="True",dataRecorder=None):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.inProcessEmb=inProcessEmb
        self.l2use=l2use
        self.dataRecorder=dataRecorder
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        if decode == 'distmult':
            self.decoder = DistMult(num_etypes, num_classes*(num_layers+2))
        elif decode == 'dot':
            self.decoder = Dot()

    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))

    def forward(self, features_list, e_feat, left, right, mid):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        emb = [self.l2_norm(h)]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            emb.append(self.l2_norm(h.mean(1)))
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=res_attn)#None)
        logits = logits.mean(1)
        logits = self.l2_norm(logits)
        emb.append(logits)
        if self.inProcessEmb=="True":
            emb.append(logits)
        else:
            emb=[logits]
        logits = torch.cat(emb, 1)
        left_emb = logits[left]
        right_emb = logits[right]
        return F.sigmoid(self.decoder(left_emb, right_emb, mid))


       
class slotGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 num_ntype,
                 n_type_mappings,
                 res_n_type_mappings,
                 etype_specified_attention,
                 eindexer,
                 ae_layer=False,aggregator="average",semantic_trans="False",semantic_trans_normalize="row",attention_average="False",attention_mse_sampling_factor=0,attention_mse_weight_factor=0,attention_1_type_bigger_constraint=0,attention_0_type_bigger_constraint=0,predicted_by_slot="None",
                 addLogitsEpsilon=0,addLogitsTrain="None",get_out=[""],slot_attention="False",relevant_passing="False",
                 decode='distmult',inProcessEmb="True",l2BySlot="False",prod_aggr=None,sigmoid="after",l2use="True",logitsRescale="None",HANattDim=128,dataRecorder=None,targetTypeAttention="False",target_edge_type=None):
        super(slotGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.heads=heads
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.ae_layer=ae_layer
        self.num_ntype=num_ntype
        self.num_classes=num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attention_mse_sampling_factor=attention_mse_sampling_factor
        self.attention_mse_weight_factor=attention_mse_weight_factor
        self.attention_1_type_bigger_constraint=attention_1_type_bigger_constraint
        self.attention_0_type_bigger_constraint=attention_0_type_bigger_constraint
        self.predicted_by_slot=predicted_by_slot
        self.addLogitsEpsilon=addLogitsEpsilon
        self.addLogitsTrain=addLogitsTrain
        self.slot_attention=slot_attention
        self.relevant_passing=relevant_passing
        self.inProcessEmb=inProcessEmb
        if relevant_passing=="True":
            assert slot_attention=="True"
        self.l2BySlot=l2BySlot
        self.prod_aggr=prod_aggr
        self.sigmoid=sigmoid
        self.l2use=l2use
        self.logitsRescale=logitsRescale
        self.HANattDim=HANattDim
        self.dataRecorder=dataRecorder
        self.targetTypeAttention=targetTypeAttention
        self.target_edge_type=target_edge_type
        #self.ae_drop=nn.Dropout(feat_drop)
        #if ae_layer=="last_hidden":
            #self.lc_ae=nn.ModuleList([nn.Linear(num_hidden * heads[-2],num_hidden, bias=True),nn.Linear(num_hidden,num_ntype, bias=True)])
        self.last_fc = nn.Parameter(th.FloatTensor(size=(num_classes*self.num_ntype, num_classes))) ;nn.init.xavier_normal_(self.last_fc, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,inputhead=True,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize,attention_average=attention_average,attention_mse_sampling_factor=attention_mse_sampling_factor,attention_mse_weight_factor=attention_mse_weight_factor,attention_1_type_bigger_constraint=attention_1_type_bigger_constraint,attention_0_type_bigger_constraint=attention_0_type_bigger_constraint,slot_attention=slot_attention,relevant_passing=relevant_passing))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
                num_hidden* heads[l-1] , num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize,attention_average=attention_average,attention_mse_sampling_factor=attention_mse_sampling_factor,attention_mse_weight_factor=attention_mse_weight_factor,attention_1_type_bigger_constraint=attention_1_type_bigger_constraint,attention_0_type_bigger_constraint=attention_0_type_bigger_constraint,slot_attention=slot_attention,relevant_passing=relevant_passing))
        # output projection
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden* heads[-2] , num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize,attention_average=attention_average,attention_mse_sampling_factor=attention_mse_sampling_factor,attention_mse_weight_factor=attention_mse_weight_factor,attention_1_type_bigger_constraint=attention_1_type_bigger_constraint,attention_0_type_bigger_constraint=attention_0_type_bigger_constraint,slot_attention=slot_attention,relevant_passing=relevant_passing))
        self.aggregator=aggregator
        if aggregator=="HAN":
            if self.inProcessEmb=="True":
                last_dim=num_hidden*(2+num_layers)
            else:
                last_dim=num_hidden
            self.macroLinear=nn.Linear(last_dim, self.HANattDim, bias=True);nn.init.xavier_normal_(self.macroLinear.weight, gain=1.414)
            self.macroSemanticVec=nn.Parameter(torch.FloatTensor(self.HANattDim,1));nn.init.normal_(self.macroSemanticVec,std=1)
        
        if self.targetTypeAttention=="True":
            assert self.aggregator=="HAN"
            tnt0,tnt1=self.target_edge_type[0]
            self.targetTypeFilter=torch.zeros(num_ntype)
            self.targetTypeFilter[tnt0]=1;self.targetTypeFilter[tnt1]=1  #(1,1,0) or others like this

        self.by_slot=[f"by_slot_{nt}" for nt in range(g.num_ntypes)]
        assert aggregator in (["onedimconv","average","last_fc","slot_majority_voting","max","None","HAN"]+self.by_slot)
        if self.aggregator=="onedimconv":
            self.nt_aggr=nn.Parameter(torch.FloatTensor(1,1,self.num_ntype,1));nn.init.normal_(self.nt_aggr,std=1)
        #self.get_out=get_out
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        if decode == 'distmult':
            if self.aggregator=="None":
                num_classes=num_classes*num_ntype
            self.decoder = DistMult(num_etypes, num_classes*(num_layers+2))
        elif decode == 'dot':
            self.decoder = Dot()


    def forward(self, features_list,e_feat, left, right, mid, get_out="False"): 
        encoded_embeddings=None
        h = []
        for nt_id,(fc, feature) in enumerate(zip(self.fc_list, features_list)):
            nt_ft=fc(feature)
            emsen_ft=torch.zeros([nt_ft.shape[0],nt_ft.shape[1]*self.num_ntype]).to(feature.device)
            emsen_ft[:,nt_ft.shape[1]*nt_id:nt_ft.shape[1]*(nt_id+1)]=nt_ft
            h.append(emsen_ft)   # the id is decided by the node types
        h = torch.cat(h, 0)        #  num_nodes*(num_type*hidden_dim)
        emb = [self.aggr_func(self.l2_norm(h,l2BySlot=self.l2BySlot))]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat,get_out=get_out, res_attn=res_attn)   #num_nodes*num_heads*(num_ntype*hidden_dim)
            emb.append(self.aggr_func(self.l2_norm(h.mean(1),l2BySlot=self.l2BySlot)))
            h = h.flatten(1)#num_nodes*(num_heads*num_ntype*hidden_dim)
            #if self.ae_layer=="last_hidden":
            encoded_embeddings=h
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat,get_out=get_out, res_attn=None)   #num_nodes*num_heads*num_ntype*hidden_dim
        #average across the ntype info

        
        logits = logits.mean(1)
        if self.predicted_by_slot!="None" and self.training==False:
            logits=logits.view(-1,1,self.num_ntype,self.num_classes)
            self.scale_analysis=torch.std_mean(logits.squeeze(1).mean(dim=-1).detach().cpu(),dim=0) if get_out!=[""] else None
            if self.predicted_by_slot in ["majority_voting","majority_voting_max"] :
                logits=logits.squeeze(1)           # num_nodes * num_ntypes*num_classes
                with torch.no_grad():
                    slot_votings=torch.argmax(logits,dim=-1)   # num_nodes * num_ntypes
                    if "majorityVoting" in get_out:
                        slot_votings_onehot=F.one_hot(slot_votings)## num_nodes * num_ntypes *num_classes
                        votings_count=slot_votings_onehot.sum(1) ## num_nodes  *num_classes
                        votings_max_count=votings_count.max(1)[0] ## num_nodes 
                        ties_flags_pos=(votings_max_count.unsqueeze(-1)==votings_count)   ## num_nodes  *num_classes
                        ties_flags=ties_flags_pos.sum(-1)>1   ## num_nodes 
                        ties_ids=ties_flags.int().nonzero().flatten().tolist()   ## num_nodes 
                        voting_patterns=torch.sort(votings_count,descending=True,dim=-1)[0]  #num_nodes  *num_classes
                        pattern_counts={}
                        ties_labels={}
                        ties_first_labels={}
                        ties_second_labels={}
                        ties_third_labels={}
                        ties_fourth_labels={}
                        for i in range(voting_patterns.shape[0]):
                            if i in self.g.node_idx_by_ntype[0]:
                                pattern=tuple(voting_patterns[i].flatten().tolist())
                                if pattern not in pattern_counts.keys():
                                    pattern_counts[pattern]=0
                                pattern_counts[pattern]+=1

                        for i in ties_ids:
                            ties_labels[i]=ties_flags_pos[i].nonzero().flatten().tolist()
                            ties_first_labels[i]=ties_labels[i][0]
                            ties_second_labels[i]=ties_first_labels[i] if len(ties_labels[i])<2 else ties_labels[i][1]
                            ties_third_labels[i]=ties_second_labels[i] if len(ties_labels[i])<3 else ties_labels[i][2]
                            ties_fourth_labels[i]=ties_third_labels[i] if len(ties_labels[i])<4 else ties_labels[i][3]
                                
                        self.majority_voting_analysis={"pattern_counts":pattern_counts,"ties_first_labels":ties_first_labels,"ties_second_labels":ties_second_labels,"ties_third_labels":ties_third_labels,"ties_fourth_labels":ties_fourth_labels,"ties_labels":ties_labels,"ties_ids":ties_ids}

                    ## num_nodes *num_classes
                    votings=torch.argmax(F.one_hot(torch.argmax(logits,dim=-1)).sum(1),dim=-1)  #num_nodes
                    #num_nodes*1
                    votings_int=(slot_votings==(votings.unsqueeze(1))).int().unsqueeze(-1)   # num_nodes *num_ntypes *1
                    self.votings_int=votings_int
                    self.voting_patterns=voting_patterns  


                if self.predicted_by_slot=="majority_voting_max":
                    logits=(logits*votings_int).max(1,keepdim=True)[0] #num_nodes *  1 *num_classes
                else:
                    logits=(logits*votings_int).sum(1,keepdim=True) #num_nodes *  1 *num_classes
            elif self.predicted_by_slot=="max":
                if "getMaxSlot" in  get_out:
                    maxSlotIndexesWithLabels=logits.max(2)[1].squeeze(1)
                    logits_indexer=logits.max(2)[0].max(2)[1]
                    self.maxSlotIndexes=torch.gather(maxSlotIndexesWithLabels,1,logits_indexer)
                logits=logits.max(2)[0]
            elif self.predicted_by_slot=="all":
                if "getSlots" in get_out:
                    self.logits=logits.detach()
                logits=logits.view(-1,1,self.num_ntype,self.num_classes).mean(2)

            else:
                target_slot=int(self.predicted_by_slot)
                logits=logits[:,:,target_slot,:].squeeze(2)
        else:
            logits=self.aggr_func(logits)
            
        #average across the heads
        ### logits = [num_nodes *  num_of_heads *num_classes]
        #self.logits_mean=logits.flatten().mean()


        #if self.addLogitsTrain=="True" or (self.addLogitsTrain=="False" and self.training==False):
        #    logits+=self.addLogitsEpsilon
        
        logits = self.l2_norm(logits,l2BySlot=self.l2BySlot)
        if self.inProcessEmb=="True":
            emb.append(logits)
        else:
            emb=[logits]
        if self.aggregator=="None" and self.inProcessEmb=="True":
            emb=[ x.view(-1, self.num_ntype,int(x.shape[1]/self.num_ntype))   for x in emb]
            o = torch.cat(emb, 2).flatten(1)
        else:
            o = torch.cat(emb, 1)
        if self.aggregator=="HAN" :
            o=o.view(-1, self.num_ntype,int(o.shape[1]/self.num_ntype))
            if self.targetTypeAttention=="True":
                slot_scores=(F.tanh( self.macroLinear(o))  @  self.macroSemanticVec).mean(0,keepdim=True) 
                toSoftmax=slot_scores[:,[self.target_edge_type[0][0],self.target_edge_type[0][1]],:]
                toSoftmax=F.softmax(toSoftmax,dim=1)
                self.slot_scores=torch.zeros_like(slot_scores)
                self.slot_scores[:,[self.target_edge_type[0][0],self.target_edge_type[0][1]],:]=toSoftmax
            else:
                slot_scores=(F.tanh( self.macroLinear(o))  @  self.macroSemanticVec).mean(0,keepdim=True)  #num_slots
                self.slot_scores=F.softmax(slot_scores,dim=1)
            o=(o*self.slot_scores).sum(1)

        left_emb = o[left]
        right_emb = o[right]
        if self.sigmoid=="after":
            logits=self.decoder(left_emb, right_emb, mid,slot_num=self.num_ntype,prod_aggr=self.prod_aggr,logitsRescale=self.logitsRescale)
            if "Test" in self.dataRecorder["status"] and self.dataRecorder["meta"]["getLogitsDistBeforeSigmoid"]=="True":
                self.dataRecorder["data"][f"{self.dataRecorder['status']}_logits"]=logits.cpu() #count dist
            logits=F.sigmoid(logits)
        elif self.sigmoid=="before":
            
            logits=self.decoder(left_emb, right_emb, mid,slot_num=self.num_ntype,prod_aggr=self.prod_aggr,sigmoid=self.sigmoid,logitsRescale=self.logitsRescale)
        elif self.sigmoid=="None":
            left_emb=self.l2_norm(left_emb,l2BySlot=self.l2BySlot)
            right_emb=self.l2_norm(right_emb,l2BySlot=self.l2BySlot)
            logits=self.decoder(left_emb, right_emb, mid,slot_num=self.num_ntype,prod_aggr=self.prod_aggr,logitsRescale=self.logitsRescale)
        else:
            raise Exception()
        return logits


    def l2_norm(self, x,l2BySlot="False"):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        if self.l2use=="True":
            if l2BySlot=="False":
                return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))
            elif l2BySlot=="True":
                x=x.view(-1, self.num_ntype,int(x.shape[1]/self.num_ntype))
                x=x / (torch.max(torch.norm(x, dim=2, keepdim=True), self.epsilon))
                x=x.flatten(1)
                return x
        elif self.l2use=="False":
            return x
        else:
            raise Exception()


    def aggr_func(self,logits):
        if self.aggregator=="average":
            logits=logits.view(-1, self.num_ntype,self.num_classes).mean(1)
        #elif self.aggregator=="onedimconv":
            #logits=(logits.view(-1,self.num_ntype,self.num_classes)*F.softmax(self.leaky_relu(self.nt_aggr),dim=2)).sum(2)
        elif self.aggregator=="last_fc":
            logits=logits.view(-1,self.num_ntype,self.num_classes)
            logits=logits.flatten(1)
            logits=logits.matmul(self.last_fc).unsqueeze(1)
        elif self.aggregator=="max":
            logits=logits.view(-1,self.num_ntype,self.num_classes).max(1)[0]
        
        elif self.aggregator=="None" or "HAN":
            logits=logits.view(-1, self.num_ntype,self.num_classes).flatten(1)



        else:
            raise NotImplementedError()
        
        return logits

