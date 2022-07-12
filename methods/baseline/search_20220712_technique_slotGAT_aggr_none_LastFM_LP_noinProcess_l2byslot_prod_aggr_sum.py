

import time
import subprocess
import multiprocessing
from threading import main_thread
from pipeline_utils import get_best_hypers,run_command_in_parallel,config_study_name,Run
import os
import copy
#time.sleep(60*60*4)


#dataset_to_evaluate=[("IMDB_corrected",1,10),("ACM_corrected",1,10),("DBLP_corrected",1,10),("pubmed_HNE_complete",1,20),]   # dataset,worker_num,repeat

dataset_to_evaluate=[("LastFM_corrected",1,10)]   # dataset,worker_num,repeat

prefix="technique_newCsv";specified_args=["dataset",   "net",      "slot_aggregator","inProcessEmb","l2BySlot","prod_aggr"]


fixed_info={"task_property":prefix,"net":"slotGAT","slot_aggregator":"None","inProcessEmb":"False","l2BySlot":"True","prod_aggr":"sum"}
task_space={"hidden-dim":"[64,128]","num-layers":"[2,3,4]","lr":"[1e-4,5e-4,1e-3]","weight-decay":"[1e-4,5e-4,1e-3]","feats-type":[3],"num-heads":[2,4],"epoch":[50,300],"decoder":["dot"],"batch-size":[8192]}

gpus=["1"]
total_trial_num=96 #useless

def get_tasks(task_space):
    tasks=[{}]
    for k,v in task_space.items():
        tasks=expand_task(tasks,k,v)
    return tasks

def expand_task(tasks,k,v):
    temp_tasks=[]
    if type(v) is str and type(eval(v)) is list:
        for value in eval(v):
            if k.startswith("search_"):
                value=str([value])
            for t in tasks:
                temp_t=copy.deepcopy(t)
                temp_t[k]=value
                temp_tasks.append(temp_t)
    elif type(v) is list:
        for value in v:
            for t in tasks:
                temp_t=copy.deepcopy(t)
                temp_t[k]=value
                temp_tasks.append(temp_t)
    else:
        for t in tasks:
            temp_t=copy.deepcopy(t)
            temp_t[k]=v
            temp_tasks.append(temp_t)
    return temp_tasks


        
task_to_evaluate=get_tasks(task_space)
print(task_to_evaluate)
start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
tc=0
for dataset,worker_num,repeat in dataset_to_evaluate:
    for task in task_to_evaluate:
        args_dict={}
        for dict_to_add in [task,fixed_info]:
            for k,v in dict_to_add.items():
                args_dict[k]=v
        net=args_dict['net']
        ##################################
        ##edit yes and no for filtering!##
        ##################################
        #yes=["technique",f"feat_type_{args_dict['feats-type']}",f"aggr_{args_dict['slot_aggregator']}"]
        #yes=[]
        #no=["attantion_average","attention_average","attention_mse","edge_feat_0","oracle"]
        #best_hypers=get_best_hypers(dataset,net,yes,no)
        #for dict_to_add in [best_hypers]:
        #    for k,v in dict_to_add.items():
        #        args_dict[k]=v
        #trial_num=int(total_trial_num/ (len(gpus)*worker_num) )
        #if trial_num<=1:
        #    trial_num=1

        args_dict['dataset']=dataset
        #args_dict['trial_num']=trial_num
        args_dict['repeat']=repeat

        study_name,study_storage=config_study_name(prefix=prefix,specified_args=specified_args,extract_dict=args_dict)
        #study_name=f"get_embeddings_{dataset}_net_{task['net']}_feats_type_{task['feats-type']}_slot_aggregator{task['slot_aggregator']}"
        #study_storage=f"sqlite:///db/{study_name}.db"
        



        args_dict['study_name']=study_name
        args_dict['study_storage']=study_storage



        run_command_in_parallel(args_dict,gpus,worker_num)


            
        end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Start time: {start_time}\nEnd time: {end_time}\nwith {tc}/{len(task_to_evaluate)*len(dataset_to_evaluate)} tasks")
        tc+=1

end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f"Start time: {start_time}\nEnd time: {end_time}\nwith {len(task_to_evaluate)*len(dataset_to_evaluate)} tasks")