import numpy as np
import torch
import pandas as pd

from agp import *
from model_utils import *


class ClosedLoop:
    def __init__(self, y_star, y_mean, y_std, x_mean, x_std, best_GWP, agp_model, agp_dict, agp=False):
    
        F, G = self.create_form_space()

        self.F = F
        self.G = G
        self.n_samples = F.shape[0]

        self.y_star = torch.tensor(y_star - y_mean)/y_std
        self.y_mean = y_mean
        self.y_std = y_std
        
        self.x_mean = x_mean
        self.x_std = x_std
        
        self.best_GWP = best_GWP
        
        self.agp_model = agp_model
        self.agp_dict = agp_dict
        
        self.exp_idx = []
        
        if agp:
            self.F_scaled = (self.F - self.x_mean)/self.x_std
    
    def prob_contraint(self, pred_mean, pred_std):
        pc = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            pc[i] = 1 - torch.distributions.Normal(torch.tensor(pred_mean[i]), 
                                                                torch.tensor(pred_std[i])).cdf(self.y_star)
        return pc
    
    def calc_rmse(self, model, samples, strength):
        
        pred, _ = self.make_predictions(model, samples=samples)
        return np.sqrt(np.mean((pred - strength)**2))
        
    
    def make_predictions(self, model, samples=None, t_scale=50):
        
        if samples is None:
            samples =  torch.from_numpy(self.F_scaled)
        
        mean = np.zeros(len(samples))
        stddev = np.zeros(len(samples))
        t_28 = np.array([28])[:, None]/t_scale
        
        for i in range(len(samples)):
            
            z_i = samples[i]
            
            with torch.no_grad():
                mn = model.weight_net(z_i).numpy()
                b = model.bias_net(z_i).numpy()
                ln = model.lengthscale_net(z_i).numpy()
                sc = model.scale_net(z_i).numpy()
                noise1 = model.noise_net_1(z_i).numpy()
                noise2 = model.noise_net_2(z_i).numpy()

            k = calc_k_rbf(t_28, ln, sc)    
            m = calc_m_exp(t_28, mn, b)  
            total_noise = noise1 * (1 - np.exp(-t_28/noise2)) + 1e-6
            
            mean[i] = m
            stddev[i] = np.sqrt(k + total_noise)
            
        
        return mean, stddev

    def create_form_space(self):
        df = pd.read_excel('../Ulva_Exp_Plans_v2.xlsx', sheet_name='Sheet1')

        # define full formulation space
        F = []
        G = []
        for i in range(len(df)):
            w = df.iloc[i]['W/C']
            b = df.iloc[i]['Concentration']
            p = df.iloc[i]['PSD']
            h = df.iloc[i]['Humidity']
            F.append([w, b, p, h])
            G.append(-df.iloc[i]['GWP_gCO2_gBiocomp'])
        F = np.stack(F)
        G = np.stack(G) 
        return F, G

    def propose_experiments(self, prob_const, pred_mean, n_experiments=4):
        
        prob_const[self.exp_idx] = 0 #don't allow for re-testing of the same setting
        
        round_exp = []
        total_exp = 0
        while total_exp < n_experiments:

            if max(0, np.max(np.multiply(self.G - self.best_GWP, np.multiply(prob_const, pred_mean >=  self.y_star.numpy())))) > 0:
                print('improving qualifying formulation')
                proposed_idx = np.argmax(np.multiply(self.G - self.best_GWP, np.multiply(prob_const, pred_mean >=  self.y_star.numpy())))
            else:
                print('best probability of improvement')
                proposed_idx = np.argmax(np.multiply(prob_const, (self.G - self.best_GWP) > 0))
            if total_exp > 0:
                #check if humidity matches
                if self.F[proposed_idx, -1] == humidity:
                    round_exp.append(proposed_idx)
                    self.exp_idx.append(proposed_idx)
                    total_exp += 1
            if total_exp == 0:
                humidity = self.F[proposed_idx, -1]
                total_exp += 1
                round_exp.append(proposed_idx)
                self.exp_idx.append(proposed_idx)
            # set the probability of meeting the constraint to zero so we don't pick the same experiment twice
            prob_const[proposed_idx] = 0
            
        return round_exp

    def check_formulation(self, round_exp, y_scale = 50):

        check = np.zeros((2, len(round_exp)))
        for j, z in enumerate(round_exp):

            z_i = (torch.from_numpy(self.F[z,[1,0,2,3]]) - self.agp_dict['z_mean']) / self.agp_dict['z_std']
            with torch.no_grad():
                mn = self.agp_model.weight_net(z_i).numpy()
                b = self.agp_model.bias_net(z_i).numpy()

            m = calc_m_exp(np.array([28./self.agp_dict['t_scale']]), mn, b)
            check[0,j] = m*y_scale
            check[1,j] = self.G[z]

            if (m*y_scale >= (self.y_star.numpy()*self.y_std + self.y_mean)) & (self.G[z] > self.best_GWP):
                self.best_GWP = self.G[z]

            if (self.F[z][1] == 11) & (self.F[z][0] == 0.5):
                print('check', self.F[z])

        return check



    def simulate_data(self, round_exp, t_scale=50, y_scale=50, noise=False, agp=False):
        # simulate data to append to training 
        x_new = []
        y_new = []
        if agp:
            z_new = []
            
        t_new = np.array([2, 4, 7, 28])[:, None]/t_scale
        t_append = (np.array([2, 4, 7, 28])[:, None] - self.x_mean[-1])/self.x_std[-1]


        for j, z in enumerate(round_exp):


            z_i = (torch.from_numpy(self.F[z,[1,0,2,3]]) - self.agp_dict['z_mean']) / self.agp_dict['z_std']
            with torch.no_grad():
                mn = self.agp_model.weight_net(z_i).numpy()
                b = self.agp_model.bias_net(z_i).numpy()
                ln = self.agp_model.lengthscale_net(z_i).numpy()
                sc = self.agp_model.scale_net(z_i).numpy()
                noise1 = self.agp_model.noise_net_1(z_i).numpy()
                noise2 = self.agp_model.noise_net_2(z_i).numpy()

            k = calc_k_rbf(t_new, ln, sc)    
            m = calc_m_exp(t_new, mn, b)    
            total_noise = noise1 * (1 - np.exp(-t_new/noise2)) + 1e-6
            mvn = create_mvn(torch.tensor(m), torch.tensor(k) + torch.tensor(total_noise)*torch.eye(k.shape[0]))

            if agp:
                z_i = self.F_scaled[z]
                
            else:
                z_i = (self.F[z] - self.x_mean[:-1])/self.x_std[:-1]
            
            if noise:
                
                    
                if agp:
                    
                    _x = []
                    _y = []
                    for _ in range(5):
                        _x.append(t_new)
                        _y.append(mvn.sample().flatten())
                    
                    x_new.append(np.hstack(_x))
                    y_new.append(np.hstack(_y))
                    z_new.append(z_i) 
                        
                else:
                    for _ in range(5):
                    
                        x_new.append(np.hstack((np.repeat(z_i[np.newaxis,:],repeats=4,axis=0),t_append)))
                        #y_new.append((m*y_scale - self.y_mean)/self.y_std + 0.5*np.random.randn())
                        y_new.append((mvn.sample()*y_scale - self.y_mean)/self.y_std)
            else:
                x_new.append(np.hstack((np.repeat(z_i[np.newaxis,:],repeats=4,axis=0),t_append)))
                y_new.append((m*y_scale - self.y_mean)/self.y_std)



        if agp:
            return np.vstack(z_new), np.hstack(x_new), np.vstack(y_new)
        else:
                
            return np.vstack(x_new), np.stack(y_new).flatten()
    
   
