r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np
from collections import Counter
from .loss import BPRLoss

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import activation_layer
from recbole.utils import InputType


class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset,sim_users,usrprf_embeds,itemprf_embeds,age,dataset_index=1,datasetname='beer',sensitive_name='age'):
        super(BPR, self).__init__(config, dataset)

        self.gamma =-256#---pbe -32 -64 -128 -256
        self.beta=64#--ua 32 64 128 256
        self.d =0.1#--ip 32 64 128 256

        self.a = 1
        self.b = 1
        self.c = 1
        self.data_name = datasetname
        self.num_items = len(dataset.item_counter)
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        #item_provider = pd.Series(dataset.item_feat.interaction['publisher'].numpy())
        if datasetname=='beer':
            item_provider = pd.Series(dataset.item_feat.interaction['brewer_id'].numpy())
        if datasetname=='brand':
            item_provider = pd.Series(dataset.item_feat.interaction['brand'].numpy())
        if datasetname=='book':
            item_provider = pd.Series(dataset.item_feat.interaction['publisher'].numpy())
        self.user_provider_matrix=user_provider_interaction(self.interaction_matrix,item_provider)
        self.dataset_index = dataset_index
        self.sensitive_name = sensitive_name
        self.sensitive=torch.tensor(age)
        #dataset.user_feat.age=self.sensitive
        #self.sensitive=dataset.user_feat.age

        self.sensitive = self.sensitive.to(torch.int64).to('cuda:0') # Convert to int64 tensor

        self.global_nov,self.global_pop_provider= compute_novelty_popularity(self.user_provider_matrix)

        self.local_nov,self.local_pop_provider= compute_local_nov_pop(self.data_name,sim_users,self.interaction_matrix)

        self.global_user_pop = compute_novelty_popularity_user(self.interaction_matrix)[2]

        #self.global_user_pop = compute_novelty_popularity_user(self.interaction_matrix)

        self.global_item_nov,self.global_item_pop = compute_novelty_popularity(self.interaction_matrix)

        self.global_pop = F.normalize(self.global_pop_provider.float(), dim=0).to(self.device)

        self.global_item_pop = F.normalize(self.global_item_pop.float(), dim=0).to(self.device)

        self.local_pop = F.normalize(self.local_pop_provider.float(), dim=1).to(self.device)

        self.global_user_pop = F.normalize(self.global_user_pop.float(), dim=0).to(self.device)
        self.f = nn.Sigmoid()
        self.reg_loss_fn = nn.MSELoss()

        self.reg_coe=1
        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.loss = BPRLoss()
        self.other_parameter()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.local_pred = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size)
        )
        self.global_pred = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size* 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0))

        self.kd_temperature = 0.01
        self.kd_weight=1.0e-2

        # semantic-embeddings
        self.usrprf_embeds = torch.tensor(usrprf_embeds).float().cuda()
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size ) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size ) // 2, self.embedding_size )
        )
        self.itmprf_embeds = torch.tensor(itemprf_embeds).float().cuda()
        self.mlp_i = nn.Sequential(
            nn.Linear(self.itmprf_embeds.shape[1], (self.itmprf_embeds.shape[1] + self.embedding_size ) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.itmprf_embeds.shape[1] + self.embedding_size ) // 2, self.embedding_size )
        )

        self.global_item_pred = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size* 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
       )

        self.global_user_pred = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size* 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )
    def cal_infonce_loss(self,embeds1, embeds2, all_embeds2, temp=1.0):
        normed_embeds1 = embeds1 / (torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True)))
        normed_embeds2 = embeds2 / (torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True)))
        normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
        nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
        deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
        cl_loss = (nume_term + deno_term).sum()
        return cl_loss
    def mlp_layers(self, layer_dims, activations, dropouts):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            mlp_modules.append(activation_layer(activations[i]))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Dropout(p=dropouts[i]))
        return nn.Sequential(*mlp_modules)

    def get_rating_matrix(self, dataset):
        history_item_id, history_item_value, _ = dataset.history_item_matrix()

        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = history_item_id.flatten()
        row_indices = torch.arange(self.n_users).repeat_interleave(
            history_item_id.shape[1], dim=0
        )
        rating_matrix = torch.zeros(1).repeat(self.n_users, self.n_items)
        rating_matrix.index_put_(
            (row_indices, col_indices), history_item_value.flatten()
        )
        return rating_matrix.to(self.device)

    def get_all_weights(self):
        h_encode = self.encoder(self.rating_matrix)
        return h_encode

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]
            user_e, pos_e = self.forward(user, pos_item)
            neg_e = self.get_item_embedding(neg_item)
            pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)

            usr_po_emb=self.global_user_pred(user_e)
            usr_ci_emb = self.local_pred(user_e)
            pos_ci_emb = self.local_pred(pos_e)
            neg_ci_emb = self.local_pred(neg_e)
            pos_ci_local = self.f(torch.mul(usr_ci_emb, pos_ci_emb).sum(1))
            neg_ci_local = self.f(torch.mul(usr_ci_emb, neg_ci_emb).sum(1))
            # pos_ci_global = self.global_pred(pos_e)
            # neg_ci_global = self.global_pred(neg_e)
            pos_ci_item_global = self.global_item_pred(pos_e)
            neg_ci_item_global = self.global_item_pred(neg_e)
            #pos_scores = pos_item_score
            #neg_scores = neg_item_score
            # pos_scores = pos_item_score*pos_ci_local* pos_ci_global
            # neg_scores = neg_item_score*neg_ci_local* neg_ci_global
            # pos_scores = pos_item_score *(pos_ci_local * pos_ci_global* pos_ci_item_global)
            # neg_scores = neg_item_score *(neg_ci_local * neg_ci_global* neg_ci_item_global)
            pos_scores = pos_item_score *pos_ci_local * pos_ci_item_global*usr_po_emb
            neg_scores = neg_item_score *neg_ci_local * neg_ci_item_global*usr_po_emb
            loss1 = self.loss(pos_scores, neg_scores)
            local_reg_loss = (self.reg_loss_fn(pos_ci_local, self.local_pop[user.cpu(), pos_item.cpu()].to(self.device)) +
                              self.reg_loss_fn(neg_ci_local, self.local_pop[user.cpu(), neg_item.cpu()].to(self.device))) / 2
            # global_reg_loss = (self.reg_loss_fn(pos_ci_global, self.global_pop[pos_item.cpu()].to(self.device)) +
            #            self.reg_loss_fn(neg_ci_global, self.global_pop[neg_item.cpu()].to(self.device))) / 2
            global_item_reg_loss=(self.reg_loss_fn(pos_ci_item_global, self.global_item_pop[pos_item.cpu()].to(self.device)) +
                          self.reg_loss_fn(neg_ci_item_global, self.global_item_pop[neg_item.cpu()].to(self.device))) / 2
            global_pop_reg_loss=(self.reg_loss_fn(usr_po_emb, self.global_user_pop[user.cpu()].to(self.device)))
            #linreg_loss = self.a*local_reg_loss+self.b*global_reg_loss
            # linreg_loss = self.a*local_reg_loss + self.b*global_reg_loss+ self.c*global_item_reg_loss
            linreg_loss = self.a*local_reg_loss+ self.c*global_item_reg_loss+global_pop_reg_loss

            usrprf_embeds = self.mlp(self.usrprf_embeds)
            itmprf_embeds = self.mlp_i(self.itmprf_embeds)
            ancprf_embeds=usrprf_embeds[user]
            posprf_embeds=itmprf_embeds[pos_item]
            negprf_embeds=itmprf_embeds[neg_item]
            kd_loss =self.cal_infonce_loss(user_e,ancprf_embeds, usrprf_embeds, self.kd_temperature)+ \
                     self.cal_infonce_loss(pos_e, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                     self.cal_infonce_loss(neg_e , negprf_embeds, negprf_embeds, self.kd_temperature)
            kd_loss /= user_e.shape[0]
            kd_loss *= self.kd_weight
            loss=loss1 + linreg_loss * self.reg_coe+kd_loss
            print(loss1)
            print(linreg_loss * self.reg_coe)
            print(loss)
            return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        scores = torch.mul(user_e, item_e).sum(dim=1)

        user_ci_emb = self.local_pred(user_e)
        item_ci_emb = self.local_pred(item_e)
        pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
        #pred_global = self.global_pred(item_e).expand(scores.shape)
        pred_user_pop=self.global_user_pred(user_e)
        real_user_pop=self.global_user_pop.expand(scores.shape).to(self.device)
        pred_item_global= self.global_item_pred(item_e).expand(scores.shape)
        real_local = self.local_pop[user.cpu()].to(self.device)
        # real_global = self.global_pop.expand(scores.shape).to(self.device)
        real_item_global= self.global_item_pop.expand(scores.shape).to(self.device)
        # scores = scores* pred_local*pred_global-(self.gamma * real_local+self.beta2 * real_global)
       # scores = scores* (pred_local*pred_global*pred_item_global)-(self.gamma * real_local + self.beta2 * real_global+self.d*real_item_global)
        scores = scores* (pred_local*pred_item_global*pred_user_pop)-(self.gamma * real_local+self.d*real_item_global+self.beta*real_user_pop)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))

        user_ci_emb = self.local_pred(user_e)
        item_ci_emb = self.local_pred(all_item_e)
        pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
        #pred_global = self.global_pred(all_item_e).expand(score.shape)
        pred_item_global= self.global_item_pred(all_item_e).expand(score.shape)
        real_local = self.local_pop[user.cpu()].to(self.device)
        # real_global = self.global_pop.expand(score.shape).to(self.device)
        real_item_global= self.global_item_pop.expand(score.shape).to(self.device)
        pred_user_pop=self.global_user_pred(user_e)
        pred_user_pop = pred_user_pop.unsqueeze(1)
        pred_user_pop = pred_user_pop.expand(-1, score.shape[1])
        real_user_pop=self.global_user_pop[user].unsqueeze(1).expand(-1,score.shape[1]).to(self.device)
        #score = score* pred_local*pred_global-(self.gamma * real_local+self.beta2 * real_global)
        #score = score* (pred_local*pred_global*pred_item_global)-(self.gamma * real_local + self.beta2 * real_global+self.d*real_item_global)
        score = score* (pred_local*pred_item_global*pred_user_pop)-(self.gamma * real_local +self.d*real_item_global+self.beta*real_user_pop)
        return score.view(-1)