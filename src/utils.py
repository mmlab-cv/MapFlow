import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

import wandb

from src.data_loader_multiagent import *

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

def get_data(FLAGS):
    dataset = TrajectoryDataset(FLAGS)
    if FLAGS.train:
        if len(FLAGS.loo_file) != 0:
            trainlen = int(len(dataset)*FLAGS.tv_split_value)
            trainset, valset = random_split(dataset,[trainlen,len(dataset)-trainlen])

            trainloader = DataLoader(dataset=trainset,batch_size=FLAGS.batch,shuffle=True, num_workers=4)
            valloader = DataLoader(dataset=valset,batch_size=FLAGS.batch,shuffle=False, num_workers=4)
            
            testset = TrajectoryDataset(FLAGS, test=True)
            testloader = DataLoader(dataset=testset,batch_size=FLAGS.batch,shuffle=False, num_workers=4)
            return trainloader, valloader, testloader

        trainlen = int(len(dataset)*FLAGS.tt_split_value)
        trainset, testset = random_split(dataset,[trainlen,len(dataset)-trainlen])

        trainlen = int(len(trainset)*FLAGS.tv_split_value)
        trainset, valset = random_split(trainset,[trainlen,len(trainset)-trainlen])

        trainloader = DataLoader(dataset=trainset,batch_size=FLAGS.batch,shuffle=True, num_workers=4)
        valloader = DataLoader(dataset=valset,batch_size=FLAGS.batch,shuffle=False, num_workers=4)
        testloader = DataLoader(dataset=testset,batch_size=FLAGS.batch,shuffle=False, num_workers=4)
        return trainloader, valloader, testloader
    else:
        trainloader = 0
        valloader = 0
        testset = TrajectoryDataset(FLAGS, test=True)
        testloader = DataLoader(dataset=testset,batch_size=FLAGS.batch,shuffle=False, num_workers=4)
        return trainloader, valloader, testloader

def save_checkpoints(FLAGS, model):
    if FLAGS.save_checkpoints:
        output_dir = 'weights/normalizingflow'
        if not(os.path.exists(output_dir)):
            os.makedirs(output_dir)

        if len(FLAGS.loo_file)==1:
            current_test, _ = os.path.splitext(FLAGS.loo_file[0])
        dest = current_test + ".pth"
        path = os.path.join(output_dir, dest)
        torch.save({
            "model": model.state_dict(),
        }, path)

def plot_traj(FLAGS, batch_idx, i, x, y_sort, y_true, epoch):
    hid_str = str(FLAGS.hidden_dim)+'/'
    history_enc_size_str = str(FLAGS.encoding_size)+'/'
    cond_ped_str = str(FLAGS.conditioning_peds)+'/'
    if FLAGS.ae_enabled:
        ae_str = 'ae_'+str(FLAGS.ae_enc)+'/'
    else:
        ae_str = 'no_ae/'
    if len(FLAGS.loo_file)==1:
        loo_file, _ = os.path.splitext(FLAGS.loo_file[0])
    output_dir = loo_file+"/"+ae_str+hid_str+history_enc_size_str+cond_ped_str+'trajectories/'+str(epoch)+'/'
    output_dir = os.path.join(FLAGS.results_folder,output_dir)
    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)
    output_fig=output_dir+str(batch_idx)+'_'+str(i)+'.png'

    fig = plt.figure()
    plt.ion()
    plt.plot(x[i,0,:,0].cpu(),x[i,0,:,1].cpu(), label='history', linestyle='dashdot')
    for j in range(FLAGS.conditioning_peds):
        other_ped=j+1
        plt.plot(x[i,other_ped,x[i,other_ped,:,0]!=0,0].cpu(),x[i,other_ped,x[i,other_ped,:,0]!=0,1].cpu(), linestyle='dashdot', alpha=0.4, color='r')
        if len(x[i,other_ped,x[i,other_ped,:,0]!=0,0]) != 0:
            plt.plot(x[i,other_ped,x[i,other_ped,:,0]!=0,0][-1].cpu(),x[i,other_ped,x[i,other_ped,:,0]!=0,1][-1].cpu(), alpha=0.4, color='r',marker='*')
    plt.plot(y_true.squeeze(1)[i,:,0].cpu(),y_true.squeeze(1)[i,:,1].cpu(), label='GT', linestyle='dashdot')
    for j in range(len(y_sort)):
        if j == 0:
            plt.plot(y_sort[j,:,0].cpu(),y_sort[j,:,1].cpu(), label=('traj'+str(j)), linestyle='dashdot')
        else:
            plt.plot(y_sort[j,:,0].cpu(),y_sort[j,:,1].cpu(), alpha=0.2)
    plt.legend()
    plt.axis([-15, 15, -15, 15])
    plt.ioff()
    fig.savefig(output_fig, dpi=360)
    plt.close()
    return fig

def plot_ade_fde(FLAGS, ade_tot, fde_tot, num_samples, epoch):
    fig=plt.figure()
    plt.plot((ade_tot/num_samples).to('cpu'), label="ade")
    plt.plot((fde_tot/num_samples).to('cpu'), label="fde")
    plt.grid()
    plt.legend()
    plt.xlim(-1,21)
    plt.ylim(0,5)
    plt.xticks([0, 3, 6, 9, 12, 15, 18])
    plt.yticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

    hid_str = str(FLAGS.hidden_dim)+'/'
    history_enc_size_str = str(FLAGS.encoding_size)+'/'
    cond_ped_str = str(FLAGS.conditioning_peds)+'/'
    if FLAGS.ae_enabled:
        ae_str = 'ae_'+str(FLAGS.ae_enc)+'/'
    else:
        ae_str = 'no_ae/'
    if len(FLAGS.loo_file)==1:
        loo_file, _ = os.path.splitext(FLAGS.loo_file[0])
    output_dir = loo_file+"/"+ae_str+hid_str+history_enc_size_str+cond_ped_str+'ade_fde/'
    output_dir = os.path.join(FLAGS.results_folder,output_dir)
    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)
    output_fig=output_dir+"ade_fde_"+str(epoch)+".png"

    fig.savefig(output_fig)
    return fig



def save_z(FLAGS, prob):
    fnm = "no_loo"
    if len(FLAGS.loo_file)==1:
        fnm, ext = os.path.splitext(FLAGS.loo_file[0])
    fnm = "prob/"+fnm+"/"
    # if FLAGS.sketched:
    #     fnm = fnm+"sketched_"
    if not(os.path.exists(fnm)):
        os.makedirs(fnm)
    ae = "no_ae"
    if FLAGS.ae_enabled:
        ae = "ae"+str(FLAGS.ae_hidden)

    var = prob.unsqueeze(1)

    filename = fnm+ae+"_p"+str(FLAGS.conditioning_peds)+".csv"
    with open(filename, "a", encoding="UTF8") as f:
        writer = csv.writer(f)
        for row in var:
            writer.writerow(row.detach().cpu().numpy())




def plotting(FLAGS, batch, number, y, x):
    output_dir = "res"
    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)

    fig = plt.figure()
    plt.ion()
    
    for i in range(y.size(0)):
        output_fig=output_dir+str(batch)+'_'+str(number)+'.png'
        if i == 0 :
            plt.plot(y[i,0,:,0].cpu(),y[i,0,:,1].cpu(), label='my_ped', linestyle='dashdot')
        else:
            plt.plot(y[i,0,:,0].cpu(),y[i,0,:,1].cpu(), linestyle='dashdot', alpha=0.4)
        for k in range(3):
            if i == 0 and k == 0:
                plt.plot(x[i,k,x[i,k,:,0]!=0,0].cpu(),x[i,k,x[i,k,:,0]!=0,1].cpu(), label='SCV', linestyle='dashdot', color='g')
            else:
                plt.plot(x[i,k,x[i,k,:,0]!=0,0].cpu(),x[i,k,x[i,k,:,0]!=0,1].cpu(), linestyle='dashdot', color='g')

    plt.legend()
    plt.axis([-15, 15, -15, 15])
    plt.ioff()
    fig.savefig(output_fig, dpi=360)
    plt.close()





def check(FLAGS, dataset, model, z=None, epoch="test"):
    model.eval()
    if str(epoch) == "test":
        model.training = False
    ade_tot = 0
    min_ade = 0
    fde_tot = 0
    min_fde = 0
    num_samples = 0
    epoch_loss = 0

    with tqdm(dataset,unit="val_batch") as tbatch:
        for batch_idx, (x, y_true) in enumerate(tbatch):
            tbatch.set_description(f"Batch {batch_idx}")
            x = x.to(FLAGS.device)
            y_true = y_true.to(FLAGS.device)

            prob, g_prob, y_enc, x_t, angle = model.log_prob(y_true, x)
            epoch_loss += -prob.sum().item()
            num_samples += x.size(0)
            
            if z is not None:
                y, probs = model.sample(FLAGS.num_traj, x, z[:(x.size(0)*20)])
            else:
                y, probs, a = model.sample(FLAGS.num_traj, x)
                
            for i in range(probs.size(0)):
                _, index = torch.sort(probs[i],descending=True)

                y_sort = y[i,index]
                ade = (y_true.squeeze(1)[i]-y_sort).pow(2).sum(-1).sqrt().mean(-1)
                fde = (y_true.squeeze(1)[i,-1]-y_sort[:,-1]).pow(2).sum(-1).sqrt()

                if str(epoch) == "test":
                    valid_indices = [i for i, point in enumerate(y_true.squeeze(1)[i]) if (point!=0).sum() != 0]
                    ade = (y_true.squeeze(1)[i][valid_indices]-y_sort[:,valid_indices]).pow(2).sum(-1).sqrt().mean(-1)
                    fde = (y_true.squeeze(1)[i,valid_indices][-1]-y_sort[:,valid_indices][:,-1]).pow(2).sum(-1).sqrt()

                min_ade += min(ade)
                min_fde += min(fde)

                ade_tot += ade
                fde_tot += fde
                
        ade_fde_fig = plot_ade_fde(FLAGS, ade_tot, fde_tot, num_samples, epoch)

    if FLAGS.wandb:
        if str(epoch) == "test":
            wandb.log({"test/loss": (epoch_loss/num_samples)})
            wandb.log({"test/ade_medio": (ade_tot/num_samples).mean()})
            wandb.log({"test/fde_medio": (fde_tot/num_samples).mean()})
            wandb.log({"test/min_ade_medio": (min_ade/num_samples).mean()})
            wandb.log({"test/min_fde_medio": (min_fde/num_samples).mean()})
            wandb.log(({"test_results/ade_fde": wandb.Image(ade_fde_fig)}))
        else:
            wandb.log({"val/loss": (epoch_loss/num_samples)}, step=epoch)
            wandb.log({"val/ade_medio": (ade_tot/num_samples).mean()}, step=epoch)
            wandb.log({"val/fde_medio": (fde_tot/num_samples).mean()}, step=epoch)
            wandb.log({"val/min_ade_medio": (min_ade/num_samples).mean()}, step=epoch)
            wandb.log({"val/min_fde_medio": (min_fde/num_samples).mean()}, step=epoch)
            wandb.log(({"results/ade_fde": wandb.Image(ade_fde_fig)}))

    return (ade_tot/num_samples), (fde_tot/num_samples), (min_ade/num_samples), (min_fde/num_samples)