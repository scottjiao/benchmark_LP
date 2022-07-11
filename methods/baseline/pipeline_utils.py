

import time
import subprocess
import multiprocessing
from threading import main_thread
import os
import pandas as pd

import copy

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
    else:
        for t in tasks:
            temp_t=copy.deepcopy(t)
            temp_t[k]=v
            temp_tasks.append(temp_t)
    return temp_tasks
    #if k.startswith("search_"):
        ##list


class Run( multiprocessing.Process):
    def __init__(self,command):
        super().__init__()
        self.command=command
    def run(self):
        
        subprocess.run(self.command,shell=True)

def proc_yes(yes,args_dict):
    temp_yes=[]
    for name in yes:
        temp_yes.append(f"{name}_{args_dict[name]}")
    return temp_yes

def get_best_hypers_from_csv(dataset,net,yes,no,metric="2_valAcc"):
    print(f"yes: {yes}, no: {no}")
    #get search best hypers
    fns=[]
    for root, dirs, files in os.walk("./log", topdown=False):
        for name in files:
            FLAG=1
            if "old" in root:
                continue
            if ".py" in name:
                continue
            if ".txt" in name:
                continue
            if ".csv" not in name:
                continue
            for n in no:
                if n in name:
                    FLAG=0
            for y in yes:
                if y not in name:
                    FLAG=0
            if FLAG==0:
                continue

            if dataset in name:
                name0=name.replace("_GTN","",1) if "kdd" not in name else name
                if net in name0 :

                    fn=os.path.join(root, name)
                    fns.append(fn)
    score_max=0
    print(fns)
    if fns==[]:
        raise Exception
    for fn in fns:

        param_data=pd.read_csv(fn)
        param_data_sorted=param_data.sort_values(by=metric,ascending=False).head(1)
        #print(param_data_sorted.columns)
        param_mapping={"1_Lr":"search_lr",
        "1_Wd":"search_weight_decay",
        "1_featType":"feats-type",
        "1_hiddenDim":"search_hidden_dim",
        "1_numLayers":"search_num_layers",
        "1_numOfHeads":"search_num_heads",}
        score=param_data_sorted[metric].iloc[0]
        if score>score_max:
            print(   f"score:{score}\t {param_data_sorted} bigger than current score {score_max} "  )
            best_hypers={}
            score_max=score
            best_param_data_sorted=param_data_sorted
            for col_name in param_data_sorted.columns:
                if col_name.startswith("1_"):
                    if param_mapping[col_name].startswith("search_"):
                        best_hypers[param_mapping[col_name]]=f"[{param_data_sorted[col_name].iloc[0]}]"
                    else:
                        best_hypers[param_mapping[col_name]]=f"{param_data_sorted[col_name].iloc[0]}"
        print(f"Best Score:{score_max}\t {best_param_data_sorted}")
        

    return best_hypers

def get_best_hypers(dataset,net,yes,no):
    print(f"yes: {yes}, no: {no}")
    #get search best hypers
    best={}
    fns=[]
    for root, dirs, files in os.walk("./log", topdown=False):
        for name in files:
            FLAG=1
            if "old" in root:
                continue
            if ".py" in name:
                continue
            if ".txt" in name:
                continue
            for n in no:
                if n in name:
                    FLAG=0
            for y in yes:
                if y not in name:
                    FLAG=0
            if FLAG==0:
                continue

            if dataset in name:
                name0=name.replace("_GTN","",1) if "kdd" not in name else name
                if net in name0 :

                    fn=os.path.join(root, name)
                    fns.append(fn)
    score_max=0
    print(fns)
    if fns==[]:
        raise Exception
    for fn in fns:
        path=fn
        FLAG0=False
        FLAG1=False
        with open(fn,"r") as f:
            for line in f:
                if "Best trial" in line and FLAG0==False:
                    FLAG0=True
                    FLAG1=False
                    continue
                if FLAG0==True:
                    if "Value" in line:
                        _,score=line.strip("\n").replace(" ","").split(":")
                        score=float(score)
                        continue
                    if "Params:" in line:
                        FLAG1=True
                        count=0
                        continue
                if FLAG1==True and score>=score_max and "    " in line and count<=5:

                    param,value=line.strip("\n").replace(" ","").split(":")
                    best[param]=value
                    score_max=score
                    FLAG0=False
                    count+=1
        print(best)
        best_hypers={}
        for key in best.keys():
            best_hypers["search_"+key]=f"""[{best[key]}]"""
    return best_hypers

def run_command_in_parallel(args_dict,gpus,worker_num):


    command='python -W ignore run_dist.py  '
    for key in args_dict.keys():
        command+=f" --{key} {args_dict[key]} "


    process_queue=[]
    for gpu in gpus:
        
        command+=f" --gpu {gpu} "
        command+=f"   > ./log/{args_dict['study_name']}.txt  "
        for _ in range(worker_num):
            
            print(f"running: {command}")
            p=Run(command)
            p.daemon=True
            p.start()
            process_queue.append(p)
            time.sleep(5)

    for p in process_queue:
        p.join()

def config_study_name(prefix,specified_args,extract_dict):
    study_name=prefix
    for k in specified_args:
        v=extract_dict[k]
        study_name+=f"_{k}_{v}"
    if study_name[0]=="_":
        study_name=study_name.replace("_","",1)
    study_storage=f"sqlite:///db/{study_name}.db"
    return study_name,study_storage

if __name__ == '__main__':
    
    
    dataset_to_evaluate=[("IMDB_corrected",1,10),]  # dataset,worker_num,repeat

    prefix="get_results";specified_args=["dataset",   "net",    "feats-type",     "slot_aggregator",     "predicted_by_slot"]


    fixed_info={"task_property":prefix,"net":"slotGAT","slot_aggregator":"average"}
    task_to_evaluate=[
    {"feats-type":"0","predicted_by_slot":"0"},
    {"feats-type":"0","predicted_by_slot":"1"},
    {"feats-type":"1","predicted_by_slot":"0"},
    {"feats-type":"1","predicted_by_slot":"1"},
    ]
    gpus=["0"]
    total_trial_num=1













    for dataset,worker_num,repeat in dataset_to_evaluate:
        for task in task_to_evaluate:
            args_dict={}
            for dict_to_add in [task,fixed_info]:
                for k,v in dict_to_add.items():
                    args_dict[k]=v
            net=args_dict['net']
            yes=["technique",f"feat_type_{args_dict['feats-type']}",f"aggr_{args_dict['slot_aggregator']}"]
            no=["attantion_average","attention_average","attention_mse","edge_feat_0","oracle"]
            best_hypers=get_best_hypers(dataset,net,yes,no)
            for dict_to_add in [best_hypers]:
                for k,v in dict_to_add.items():
                    args_dict[k]=v
            trial_num=int(total_trial_num/ (len(gpus)*worker_num) )
            if trial_num<=1:
                trial_num=1

            args_dict['dataset']=dataset
            args_dict['trial_num']=trial_num
            args_dict['repeat']=repeat

            study_name,study_storage=config_study_name(prefix=prefix,specified_args=specified_args,extract_dict=args_dict)
            #study_name=f"get_embeddings_{dataset}_net_{task['net']}_feats_type_{task['feats-type']}_slot_aggregator{task['slot_aggregator']}"
            #study_storage=f"sqlite:///db/{study_name}.db"
            
            args_dict['study_name']=study_name
            args_dict['study_storage']=study_storage



            run_command_in_parallel(args_dict,gpus,worker_num)
