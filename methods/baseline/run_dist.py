import sys
sys.path.append('../../')
import time
import argparse
import json
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tools import writeIntoCsvLogger,vis_data_collector
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from GNN import myGAT,slotGAT
import dgl
import os

torch.set_num_threads(4)


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


torch.random.manual_seed(2022)

def run_model_DBLP(args):
    dataRecorder={"meta":{
        "getLogitsDistBeforeSigmoid":args.getLogitsDistBeforeSigmoid,
    },"data":{},"status":"None"}
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    exp_info=f"exp setting: {vars(args)}"
    vis_data_saver=vis_data_collector()
    vis_data_saver.save_meta(exp_info,"exp_info")
    try:
        torch.cuda.set_device(int(args.gpu))
    except :
        pass
    feats_type = args.feats_type
    features_list, adjM, dl = load_data(args.dataset)
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu!="cpu") else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    
    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    total = len(list(dl.links_test['data'].keys()))
    num_ntypes=len(features_list)
    #num_layers=len(hiddens)-1
    num_nodes=dl.nodes['total']

    g.node_idx_by_ntype=[]

    g.node_ntype_indexer=torch.zeros(num_nodes,num_ntypes).to(device)
    ntype_dims=[]
    idx_count=0
    ntype_count=0
    for feature in features_list:
        temp=[]
        for _ in feature:
            temp.append(idx_count)
            g.node_ntype_indexer[idx_count][ntype_count]=1
            idx_count+=1

        g.node_idx_by_ntype.append(temp)
        ntype_dims.append(feature.shape[1])
        ntype_count+=1
    ntypes=g.node_ntype_indexer.argmax(1)
    res_2hops=[]
    res_randoms=[]
    
    toCsvRepetition=[]
    for re in range(args.repeat):
        print(f"re : {re} starts!\n\n\n")
        res_2hop = defaultdict(float)
        res_random = defaultdict(float)
        train_pos, valid_pos = dl.get_train_valid_pos()#edge_types=[test_edge_type])
        # train_pos {etype0: [[...], [...]], etype1: [[...], [...]]}
        # valid_pos {etype0: [[...], [...]], etype1: [[...], [...]]}
        num_classes = args.hidden_dim
        heads = [args.num_heads] * args.num_layers + [args.num_heads]
        num_ntype=len(features_list)
        g.num_ntypes=num_ntype
        n_type_mappings=False
        res_n_type_mappings=False
        etype_specified_attention=False
        eindexer=None
        prod_aggr=args.prod_aggr if args.prod_aggr!="None" else None
        if args.net=="myGAT":
            net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False, 0., decode=args.decoder,dataRecorder=dataRecorder)
        elif args.net=="slotGAT":
            net = slotGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05, num_ntype,
                 n_type_mappings,
                 res_n_type_mappings,
                 etype_specified_attention,
                 eindexer,decode=args.decoder,aggregator=args.slot_aggregator,inProcessEmb=args.inProcessEmb,l2BySlot=args.l2BySlot,prod_aggr=prod_aggr,sigmoid=args.sigmoid,l2use=args.l2use,logitsRescale=args.logitsRescale,HANattDim=args.HANattDim,dataRecorder=dataRecorder,targetTypeAttention=args.targetTypeAttention,target_edge_type=dl.links_test['meta'])
        print(net) if args.verbose=="True" else None
        

            
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        try:
            t=time.localtime()
            str_t=f"{t.tm_year:0>4d}{t.tm_mon:0>2d}{t.tm_hour:0>2d}{t.tm_min:0>2d}{t.tm_sec:0>2d}{int(time.time()*1000)%1000}"
            ckp_dname=os.path.join('checkpoint',str_t)
            os.mkdir(ckp_dname)
        except:
            time.sleep(1)
            t=time.localtime()
            str_t=f"{t.tm_year:0>4d}{t.tm_mon:0>2d}{t.tm_hour:0>2d}{t.tm_min:0>2d}{t.tm_sec:0>2d}{int(time.time()*1000)%1000}"
            ckp_dname=os.path.join('checkpoint',str_t)
            os.mkdir(ckp_dname)
        ckp_fname=os.path.join(ckp_dname,'checkpoint_{}_{}_re_{}_feat_{}_heads_{}_{}.pt'.format(args.dataset, args.num_layers,re,args.feats_type,args.num_heads,net.__class__.__name__))
        
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=ckp_fname)
        loss_func = nn.BCELoss()
    
        for epoch in range(args.epoch):
            train_pos_head_full = np.array([])
            train_pos_tail_full = np.array([])
            train_neg_head_full = np.array([])
            train_neg_tail_full = np.array([])
            r_id_full = np.array([])
            for test_edge_type in dl.links_test['data'].keys():
                train_neg = dl.get_train_neg(edge_types=[test_edge_type])[test_edge_type]
                # train_neg [[0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [9294, 8277, 4484, 7413, 1883, 5117, 9256, 3106, 636, ...]]
                train_pos_head_full = np.concatenate([train_pos_head_full, np.array(train_pos[test_edge_type][0])])
                train_pos_tail_full = np.concatenate([train_pos_tail_full, np.array(train_pos[test_edge_type][1])])
                train_neg_head_full = np.concatenate([train_neg_head_full, np.array(train_neg[0])])
                train_neg_tail_full = np.concatenate([train_neg_tail_full, np.array(train_neg[1])])
                r_id_full = np.concatenate([r_id_full, np.array([test_edge_type]*len(train_pos[test_edge_type][0]))])
            train_idx = np.arange(len(train_pos_head_full))
            np.random.shuffle(train_idx)
            batch_size = args.batch_size
            epoch_val_loss=0
            c=0
            val_res_RocAucRandom=0
            val_res_MRRRandom=0
            for step, start in enumerate(range(0, len(train_pos_head_full), args.batch_size)):

            

                t_start = time.time()
                # training
                net.train()
                train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
                train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
                train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
                train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
                r_id = r_id_full[train_idx[start:start+batch_size]]
                left = np.concatenate([train_pos_head, train_neg_head])  #to get heads embeddings
                right = np.concatenate([train_pos_tail, train_neg_tail])   #to get tail embeddings
                mid = np.concatenate([r_id, r_id])  #specify edge types
                labels = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

                logits = net(features_list, e_feat, left, right, mid)
                logp = logits
                train_loss = loss_func(logp, labels)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t_end = time.time()

                # print training info
                print('Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, step, train_loss.item(), t_end-t_start))  if args.verbose=="True" else None

                t_start = time.time()
                # validation
                net.eval()
                with torch.no_grad():
                    valid_pos_head = np.array([])
                    valid_pos_tail = np.array([])
                    valid_neg_head = np.array([])
                    valid_neg_tail = np.array([])
                    valid_r_id = np.array([])
                    for test_edge_type in dl.links_test['data'].keys():
                        valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[test_edge_type]
                        valid_pos_head = np.concatenate([valid_pos_head, np.array(valid_pos[test_edge_type][0])])
                        valid_pos_tail = np.concatenate([valid_pos_tail, np.array(valid_pos[test_edge_type][1])])
                        valid_neg_head = np.concatenate([valid_neg_head, np.array(valid_neg[0])])
                        valid_neg_tail = np.concatenate([valid_neg_tail, np.array(valid_neg[1])])
                        valid_r_id = np.concatenate([valid_r_id, np.array([test_edge_type]*len(valid_pos[test_edge_type][0]))])
                    left = np.concatenate([valid_pos_head, valid_neg_head])
                    right = np.concatenate([valid_pos_tail, valid_neg_tail])
                    mid = np.concatenate([valid_r_id, valid_r_id])
                    labels = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                    logits = net(features_list, e_feat, left, right, mid)
                    logp = logits
                    val_loss = loss_func(logp, labels)

                    pred = logits.cpu().numpy()
                    edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
                    labels = labels.cpu().numpy()
                    val_res = dl.evaluate(edge_list, pred, labels)

                    val_res_RocAucRandom+=val_res["roc_auc"]*left.shape[0]
                    val_res_MRRRandom+=val_res["MRR"]*left.shape[0]
                    epoch_val_loss+=val_loss.item()*left.shape[0]
                    c+=left.shape[0]
                
                t_end = time.time()
                # print validation info
                print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                    epoch, val_loss.item(), t_end - t_start))  if args.verbose=="True" else None
                # early stopping
                early_stopping(val_loss, net)
                if early_stopping.early_stop:
                    print('Early stopping!')  if args.verbose=="True" else None
                    break
            epoch_val_loss=epoch_val_loss/c
            val_res_RocAucRandom=val_res_RocAucRandom/c
            val_res_MRRRandom=val_res_MRRRandom/c
        

        first_flag = True
        for test_edge_type in dl.links_test['data'].keys():
            # testing with evaluate_results_nc
            net.load_state_dict(torch.load(ckp_fname))
            net.eval()
            test_logits = []
            with torch.no_grad():
                test_neigh, test_label = dl.get_test_neigh()
                test_neigh = test_neigh[test_edge_type]
                test_label = test_label[test_edge_type]
                # save = np.array([test_neigh[0], test_neigh[1], test_label])
                # print(save)
                # np.savetxt(f"{args.dataset}_{test_edge_type}_label.txt", save, fmt="%i")
                #save = np.loadtxt(os.path.join(dl.path, f"{args.dataset}_ini_{test_edge_type}_label.txt"), dtype=int)
                #test_neigh = [save[0], save[1]]
                #test_label = np.random.randint(2, size=save[0].shape[0])
                # test_label = save[2]
                left = np.array(test_neigh[0])
                right = np.array(test_neigh[1])
                mid = np.zeros(left.shape[0], dtype=np.int32)
                mid[:] = test_edge_type
                labels = torch.FloatTensor(test_label).to(device)
                net.dataRecorder["status"]="Test2Hop"
                logits = net(features_list, e_feat, left, right, mid)
                net.dataRecorder["status"]="None"
                pred = logits.cpu().numpy()
                edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
                labels = labels.cpu().numpy()
                dl.gen_file_for_evaluate(test_neigh, pred, test_edge_type, file_path=f"{args.dataset}_{args.run}.txt", flag=first_flag)
                first_flag = False
                res = dl.evaluate(edge_list, pred, labels)
                #print(f"res {res}")
                for k in res:
                    res_2hop[k] += res[k]
            with torch.no_grad():
                test_neigh, test_label = dl.get_test_neigh_w_random()
                test_neigh = test_neigh[test_edge_type]
                test_label = test_label[test_edge_type]
                left = np.array(test_neigh[0])
                right = np.array(test_neigh[1])
                mid = np.zeros(left.shape[0], dtype=np.int32)
                mid[:] = test_edge_type
                labels = torch.FloatTensor(test_label).to(device)
                net.dataRecorder["status"]="TestRandom"
                logits = net(features_list, e_feat, left, right, mid)
                net.dataRecorder["status"]="None"
                pred = logits.cpu().numpy()
                edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
                labels = labels.cpu().numpy()
                res = dl.evaluate(edge_list, pred, labels)
                #print(f"res {res}")
                for k in res:
                    res_random[k] += res[k]
        for k in res_2hop:
            res_2hop[k] /= total
        for k in res_random:
            res_random[k] /= total
        res_2hops.append(res_2hop)
        res_randoms.append(res_random)

        toCsv={ "1_featType":feats_type,
            "1_numLayers":args.num_layers,
            "1_hiddenDim":args.hidden_dim,
            "1_numOfHeads":args.num_heads,
            "1_numOfEpoch":args.epoch,
            "1_Lr":args.lr,
            "1_Wd":args.weight_decay,
            "1_decoder":args.decoder,
            "1_batchSize":args.batch_size,
            #"2_valAcc":val_results["acc"],
            #"2_valMiPre":val_results["micro-pre"],
            "2_valLossNeg_mean":-epoch_val_loss,
            "2_valRocAucRandom_mean":val_res_RocAucRandom,
            "2_valMRRRandom_mean":val_res_MRRRandom,
            "3_testRocAuc2hop_mean":res_2hop["roc_auc"],
            "3_testMRR2hop_mean":res_2hop["MRR"],
            "3_testRocAucRandom_mean":res_random["roc_auc"],
            "3_testMRRRandom_mean":res_random["MRR"],
            "2_valLossNeg_std":-epoch_val_loss,
            "2_valRocAucRandom_std":val_res_RocAucRandom,
            "2_valMRRRandom_std":val_res_MRRRandom,
            "3_testRocAuc2hop_std":res_2hop["roc_auc"],
            "3_testMRR2hop_std":res_2hop["MRR"],
            "3_testRocAucRandom_std":res_random["roc_auc"],
            "3_testMRRRandom_std":res_random["MRR"],}
        toCsvRepetition.append(toCsv)


        #net.dataRecorder["data"]["Test2Hop_logits"]
        if args.getLogitsDistBeforeSigmoid=="True":
            test2HopDist= torch.histogram(net.dataRecorder["data"]["Test2Hop_logits"],bins=60,range=(-30,30))
            test2HopDist=[(f"{test2HopDist[1][i].item()}-{test2HopDist[1][i+1].item()}",test2HopDist[0][i].item()) for i in range(len(test2HopDist[0]))]
            vis_data_saver.collect_in_run(test2HopDist,"test2Hopdist",re=re)
            testRandomdist= torch.histogram(net.dataRecorder["data"]["TestRandom_logits"],bins=60,range=(-30,30))
            testRandomdist=[(f"{testRandomdist[1][i].item()}-{testRandomdist[1][i+1].item()}",testRandomdist[0][i].item()) for i in range(len(testRandomdist[0]))]
            vis_data_saver.collect_in_run(testRandomdist,"testRandomdist",re=re)
            
        #if early_stopping.early_stop:
            #print('Early stopping!')  if args.verbose=="True" else None
            #break
    print(f"res_2hops {res_2hops}")  if args.verbose=="True" else None
    print(f"res_randoms {res_randoms}")  if args.verbose=="True" else None

    toCsvAveraged={}
    for tocsv in toCsvRepetition:
        for name in tocsv.keys():
            if name.startswith("1_"):
                toCsvAveraged[name]=tocsv[name]
            else:
                if name not in toCsvAveraged.keys():
                    toCsvAveraged[name]=[]
                toCsvAveraged[name].append(tocsv[name])
    print(toCsvAveraged)
    for name in toCsvAveraged.keys():
        if not name.startswith("1_") :
            if type(toCsvAveraged[name][0]) is str:
                toCsvAveraged[name]=toCsvAveraged[name][0]
            else:
                
                if "_mean" in name:
                    toCsvAveraged[name]=np.mean(np.array(toCsvAveraged[name])) 
                elif "_std" in name:
                    toCsvAveraged[name]= np.std(np.array(toCsvAveraged[name])) 
    #toCsvAveraged["5_expInfo"]=exp_info

    writeIntoCsvLogger(toCsvAveraged,f"./log/{args.study_name}.csv")



    #########################################
    #####        data preprocess
    #########################################


    if not os.path.exists(f"./analysis"):
        os.mkdir("./analysis")
    if not os.path.exists(f"./analysis/{args.study_name}"):
        os.mkdir(f"./analysis/{args.study_name}")
    vis_data_saver.save(os.path.join(f"./analysis/{args.study_name}",args.study_name+".visdata"))

    


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=2, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=40, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.01)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--net', type=str, default='myGAT')
    ap.add_argument('--gpu', type=str, default="0")
    ap.add_argument('--verbose', type=str, default='False')
    ap.add_argument('--edge-feats', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--decoder', type=str, default='distmult')  
    ap.add_argument('--inProcessEmb', type=str, default='True')
    ap.add_argument('--l2BySlot', type=str, default='True')
    ap.add_argument('--l2use', type=str, default='True')
    ap.add_argument('--prod_aggr', type=str, default='None')
    ap.add_argument('--sigmoid', type=str, default='after')
    ap.add_argument('--logitsRescale', type=str, default='None')
    ap.add_argument('--getLogitsDistBeforeSigmoid', type=str, default='False')
    ap.add_argument('--targetTypeAttention', type=str, default='False')
    #ap.add_argument('--macroAggr', type=str, default='None')

    


    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--cost', type=int, default=1)
    ap.add_argument('--repeat', type=int, default=10, help='Repeat the training and testing for N times. Default is 1.')
    
    ap.add_argument('--task_property', type=str, default="notSpecified")
    ap.add_argument('--study_name', type=str, default="temp")
    ap.add_argument('--study_storage', type=str, default="sqlite:///db/temp.db")
    ap.add_argument('--residual', type=str, default="True")


    ap.add_argument('--HANattDim', type=int, default=128)
    ap.add_argument('--slot_aggregator', type=str, default="HAN")
    ap.add_argument('--slot_attention', type=str, default="False") #
    ap.add_argument('--predicted_by_slot', type=str, default="None")
    ap.add_argument('--relevant_passing', type=str, default="False") #
    ap.add_argument('--get_test_for_online', default="False")  

    args = ap.parse_args()
    run_model_DBLP(args)
