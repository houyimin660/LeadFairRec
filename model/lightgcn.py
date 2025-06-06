import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
from utils import *
from .loss import BPRLoss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.layers import activation_layer
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
import json
import pickle
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#这里注意 之后用LLM生成sensitive之后要注意修改这个文件！！！！！
with open('dataset/dataset_info.json', 'r', encoding='utf-8') as file:
    dataset_info = json.load(file)
#GeneralRecommender用于实现通用的推荐系统模型，GeneralRecommender 提供了一些基本的功能和结构，供具体的推荐模型继承和实现
class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset,sim_users,usrprf_embeds,itemprf_embeds,sensitive,datasetname='brand',dataset_index=1,sensitive_name='age'):
        super(LightGCN, self).__init__(config, dataset)

        self.gamma =-16#0.05 0.1 0.2 0.3 0.4 0.5
        self.beta2=4 #-0.03 -0.3 -0.1 -0.5 -0.6 -0.8 0.5 0.8 1
        self.d = 32

        self.a = 1
        self.b = 1
        self.c = 1
        self.data_name = datasetname

        self.kd_temperature = 0.2
        self.kd_weight=1.0e-3

        self.reverse_weight=0.25
        #provider_total_weight
        self.reg_coe = 0.1

        self.beta_weight=0.5

        self.eo_weight=0.5
        self.ud_eo_loss=0.5

        self.dataset_index = dataset_index
        self.sensitive_name = sensitive_name
        #self.sensitive=dataset.user_feat.age
        self.sensitive=torch.tensor(sensitive)

        self.sensitive = self.sensitive.to(torch.int64).to('cuda:0') # Convert to int64 tensor
        self.hidden_size = 512
        self.fisced = nn.Embedding(6, 1)

        self.mlp_combine = nn.Sequential(
            nn.Linear(128, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 64),
            nn.BatchNorm1d(64),
        )

        self.mlp_sensitive = nn.Sequential(
            nn.Linear(64, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 64),
            nn.BatchNorm1d(64),
        )

        self.mlp_sensitive_dense = nn.Sequential(
            nn.Linear(64, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 64),
            nn.BatchNorm1d(64),
        )
        self.binary_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            ) for i in range(5)
        ])
        if self.sensitive_name == 'escs':
            self.mlp_sensitive_reverse = nn.Sequential(
                nn.Linear(64, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size//2),
                nn.BatchNorm1d(self.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.hidden_size//2, 64),
                nn.BatchNorm1d(64)
        )
        else:
            self.mlp_sensitive_reverse = nn.Sequential(
                nn.Linear(64, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size//2),
                nn.BatchNorm1d(self.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.hidden_size//2, 6),
                nn.Softmax(dim=1)
            )
        self.last_combine = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 1),
            nn.BatchNorm1d(1),
        )
        self.congfig=config
        self.num_items = len(dataset.item_counter)
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        if datasetname=='beer':
            item_provider = pd.Series(dataset.item_feat.interaction['brewer_id'].numpy())
        if datasetname=='brand':
           item_provider = pd.Series(dataset.item_feat.interaction['brand'].numpy())
        if datasetname=='book':
           item_provider = pd.Series(dataset.item_feat.interaction['publisher'].numpy())
        self.user_provider_matrix=user_provider_interaction(self.interaction_matrix , item_provider)

        self.global_nov,self.global_pop_provider= compute_novelty_popularity(self.user_provider_matrix)
        self.local_nov,self.local_pop_provider= compute_local_nov_pop(self.data_name,sim_users,self.interaction_matrix)

        self.global_item_nov,self.global_item_pop = compute_novelty_popularity(self.interaction_matrix)

        self.global_user_pop = compute_novelty_popularity_user(self.interaction_matrix)[2]

        #self.global_pop = F.normalize(self.global_pop_provider.float(), dim=0)

        self.global_item_pop = F.normalize(self.global_item_pop.float(), dim=0)

        self.local_pop = F.normalize(self.local_pop_provider.float(), dim=1)


        self.global_user_pop = F.normalize(self.global_user_pop.float(), dim=0).to(self.device)

        self.f = nn.Sigmoid()
        self.reg_loss_fn = nn.MSELoss()
        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN

        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]
        self.alpha = config["alpha"]
        self.fairness_type = config["fairness_type"]
        self.fairness_weight = None
        self.local_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim)
        )
        self.global_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)

        )
        self.global_item_pred = nn.Sequential(
             nn.Linear(self.latent_dim, self.latent_dim * 2),
             nn.ReLU(),
             nn.Linear(self.latent_dim * 2, 1),
             nn.Sigmoid(),
             nn.Flatten(start_dim=0)
        )
        self.global_user_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim* 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )
        # define layers and loss

        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.alpha = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.beta = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        # semantic-embeddings
        self.usrprf_embeds = torch.tensor(usrprf_embeds).float().cuda()
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.latent_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.latent_dim) // 2, self.latent_dim)
        )
        self.itmprf_embeds = torch.tensor(itemprf_embeds).float().cuda()
        self.mlp_i = nn.Sequential(
            nn.Linear(self.itmprf_embeds.shape[1], (self.itmprf_embeds.shape[1] + self.latent_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.itmprf_embeds.shape[1] + self.latent_dim) // 2, self.latent_dim)
        )
    def calc_eo1(self,sensitive, pred):
        indices = torch.argsort(sensitive)#根据esec排序
        #print('index:', indices)

        n = len(indices)
        q1_index = int(n / 4)
        q3_index = int(3 * n / 4)

        # Extract indices for three groups
        disadv_indices = indices[:q1_index]
        mid_indices = indices[q1_index:q3_index]
        adv_indices = indices[q3_index:]
        #print(disadv_indices, mid_indices, adv_indices)

        # Use indices to get corresponding pred values
        tpr_disadv = torch.mean(pred[disadv_indices])
        tpr_mid = torch.mean(pred[mid_indices])
        tpr_adv = torch.mean(pred[adv_indices])

        #tpr_disadv, tpr_mid, tpr_adv = torch.tensor(tpr_disadv), torch.tensor(tpr_mid), torch.tensor(tpr_adv)
        #tpr_disadv, tpr_mid, tpr_adv = tpr_disadv.requires_grad_(), tpr_mid.requires_grad_(), tpr_adv.requires_grad_()
        ##print(pred[disadv_indices], pred[mid_indices], pred[adv_indices])
        EO = torch.std(torch.stack([tpr_disadv, tpr_mid, tpr_adv]))
        return EO
    def calc_eo2(self,sensitive, pred):
        # Map original escs to new groups
        disadv_indices1 = torch.nonzero(sensitive == 0).squeeze()
        disadv_indices2= torch.nonzero(sensitive == 1).squeeze()
        mid_indices1 = torch.nonzero(sensitive == 2).squeeze()
        mid_indices2= torch.nonzero(sensitive == 3).squeeze()
        adv_indices1 = torch.nonzero(sensitive== 4).squeeze()
        adv_indices2 = torch.nonzero(sensitive == 5).squeeze()

        # Use indices to get corresponding pred values
        tpr_disadv1 = torch.mean(pred[disadv_indices1])
        tpr_disadv2 = torch.mean(pred[disadv_indices2])
        tpr_mid1= torch.mean(pred[mid_indices1])
        tpr_mid2 = torch.mean(pred[mid_indices2])
        tpr_adv1 = torch.mean(pred[adv_indices1])
        tpr_adv2 = torch.mean(pred[adv_indices2])

        # Calculate Equal Opportunity
        EO = torch.std(torch.stack([tpr_disadv1,tpr_disadv2, tpr_mid1,tpr_mid2, tpr_adv1,tpr_adv2]))
        return EO


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
        #col_indices = torch.tensor([2, 3, 0, 1, 1, -1])
        row_indices = torch.arange(self.n_users).repeat_interleave(
            history_item_id.shape[1], dim=0
        )
        #row_indices = torch.tensor([0, 0, 1, 1, 2, 2])
        rating_matrix = torch.zeros(1).repeat(self.n_users, self.n_items)

        rating_matrix.index_put_(
            (row_indices, col_indices), history_item_value.flatten()
        )

        return rating_matrix.to(self.device)

    def get_all_weights(self):
        h_encode = self.encoder(self.rating_matrix)
        return h_encode

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    def cal_infonce_loss(self,embeds1, embeds2, all_embeds2, temp=1.0):
        normed_embeds1 = embeds1 / (torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True)))
        normed_embeds2 = embeds2 / (torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True)))
        normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
        nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
        deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
        cl_loss = (nume_term + deno_term).sum()
        return cl_loss

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
            if self.restore_user_e is not None or self.restore_item_e is not None:
                  self.restore_user_e, self.restore_item_e = None, None

            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]
            user_all_embeddings, item_all_embeddings= self.forward()
            u_embeddings = user_all_embeddings[user]
            pos_embeddings = item_all_embeddings[pos_item]
            neg_embeddings = item_all_embeddings[neg_item]

            usrprf_embeds = self.mlp(self.usrprf_embeds)
            itmprf_embeds = self.mlp_i(self.itmprf_embeds)
            ancprf_embeds=usrprf_embeds[user]
            posprf_embeds=itmprf_embeds[pos_item]
            negprf_embeds=itmprf_embeds[neg_item]
            kd_loss =self.cal_infonce_loss(u_embeddings,ancprf_embeds, usrprf_embeds, self.kd_temperature)+ \
                     self.cal_infonce_loss(pos_embeddings, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                     self.cal_infonce_loss(neg_embeddings, negprf_embeds, negprf_embeds, self.kd_temperature)
            # ancprf_embeds=usrprf_embeds[user]
            # kd_loss =self.cal_infonce_loss(Uf_user_embeding,ancprf_embeds, usrprf_embeds, self.kd_temperature)
            kd_loss /= u_embeddings.shape[0]
            kd_loss *= self.kd_weight
            # calculate BPR Loss
            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

            usr_po_emb=self.global_user_pred(u_embeddings)
            usr_ci_emb = self.local_pred(u_embeddings)
            pos_ci_emb = self.local_pred(pos_embeddings)
            neg_ci_emb = self.local_pred(neg_embeddings)
            pos_ci_local = self.f(torch.mul(usr_ci_emb, pos_ci_emb).sum(1))
            neg_ci_local = self.f(torch.mul(usr_ci_emb, neg_ci_emb).sum(1))
            pos_ci_item_global = self.global_item_pred(pos_embeddings)
            neg_ci_item_global = self.global_item_pred(neg_embeddings)
            pos_scores = pos_scores * (pos_ci_local * pos_ci_item_global)
            neg_scores = neg_scores * (neg_ci_local* neg_ci_item_global)
            mf_loss = self.mf_loss(pos_scores, neg_scores)

            local_reg_loss = (self.reg_loss_fn(pos_ci_local, self.local_pop[user.cpu(), pos_item.cpu()].to(self.device)) +
                              self.reg_loss_fn(neg_ci_local, self.local_pop[user.cpu(), neg_item.cpu()].to(self.device))) / 2
            global_item_reg_loss=(self.reg_loss_fn(pos_ci_item_global, self.global_item_pop[pos_item.cpu()].to(self.device)) +
                                self.reg_loss_fn(neg_ci_item_global, self.global_item_pop[neg_item.cpu()].to(self.device))) / 2
            global_pop_reg_loss=(self.reg_loss_fn(usr_po_emb, self.global_user_pop[user.cpu()].to(self.device)))
            #linreg_loss = self.a*local_reg_loss + self.c*global_item_reg_loss
            linreg_loss = self.a*local_reg_loss+ self.c*global_item_reg_loss+global_pop_reg_loss
            # calculate BPR Loss
            u_ego_embeddings = self.user_embedding(user)
            pos_ego_embeddings = self.item_embedding(pos_item)
            neg_ego_embeddings = self.item_embedding(neg_item)

            reg_loss = self.reg_loss(
                u_ego_embeddings,
                pos_ego_embeddings,
                neg_ego_embeddings,
                require_pow=self.require_pow,
            )
            # pred_f,pred_d=self.full_sort_predict_2(interaction)
            # sensitive=interaction.interaction['age']
            # if self.sensitive_name == 'escs':
            #     theta_eo_loss = self.calc_eo1(sensitive, pred_f)
            #     Ud_eo_loss = self.calc_eo1(sensitive, pred_d)
            # else:
            #     theta_eo_loss = self.calc_eo2(sensitive, pred_f)
            #     Ud_eo_loss = self.calc_eo2(sensitive, pred_d)
            #loss = mf_loss + self.reg_weight * reg_loss
            # loss = mf_loss + self.reg_weight * reg_loss+self.eo_weight*theta_eo_loss+self.ud_eo_loss*Ud_eo_loss+self.reverse_weight*reverse_loss+linreg_loss * self.reg_coe+kd_loss
            loss = mf_loss + self.reg_weight * reg_loss+linreg_loss * self.reg_coe+kd_loss
            print('mfloss'+str(mf_loss),'reg_loss'+str(self.reg_weight * reg_loss))
            # print('Ud_eo_loss'+str(Ud_eo_loss)+'reverse_loss'+str(reverse_loss)+"linreg_loss"+str(linreg_loss)+'theta_eo_loss'+str(theta_eo_loss))
            print('kdloss'+str(kd_loss))
            print('loss'+str(loss))
            return loss

    @classmethod
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings= self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        user_ci_emb = self.local_pred(u_embeddings)
        item_ci_emb = self.local_pred(i_embeddings)

        pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
        #pred_global = self.global_pred(i_embeddings).expand(scores.shape)
        pred_item_global= self.global_item_pred(i_embeddings).expand(scores.shape)
        real_local = self.local_pop[user.cpu()].to(self.device)
        real_global = self.global_pop.expand(scores.shape).to(self.device)
        real_item_global= self.global_item_pop.expand(scores.shape).to(self.device)
        #scores = scores* (pred_local*pred_global*pred_item_global)-(self.gamma * real_local + self.beta2 * real_global+self.d*real_item_global)
        # scores = scores* (pred_local*pred_global)-(self.gamma * real_local + self.beta2 * real_global)
        scores = scores* (pred_local*pred_item_global)-(self.gamma * real_local +self.d*real_item_global)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:

            self.restore_user_e, self.restore_item_e= self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_item_e
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        user_ci_emb = self.local_pred(u_embeddings)
        item_ci_emb = self.local_pred(i_embeddings)
        pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
        # pred_global = self.global_pred(i_embeddings).expand(scores.shape)
        pred_item_global= self.global_item_pred(i_embeddings).expand(scores.shape)
        real_local = self.local_pop[user.cpu()].to(self.device)
        #real_global = self.global_pop.expand(scores.shape).to(self.device)
        real_item_global= self.global_item_pop.expand(scores.shape).to(self.device)
        #scores = scores* (pred_local*pred_global*pred_item_global)-(self.gamma * real_local *self.beta2* real_global*self.d*real_item_global)
        #scores = scores* (pred_local*pred_global*pred_item_global)-(self.gamma * real_local + self.beta2* real_global+self.d*real_item_global)
        #scores = scores* (pred_local*pred_global)-(self.gamma * real_local+self.beta2* real_global)
        scores = scores* (pred_local*pred_item_global)-(self.gamma * real_local+self.d*real_item_global)
        return scores
        #return scores.view(-1)
