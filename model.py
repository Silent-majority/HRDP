import torch
import torch.nn as nn
import numpy as np


class LightGCN(nn.Module):
    """Graph Convolution for graph"""

    def __init__(self, num_entity1, num_entity2, layers, graph):
        super(LightGCN, self).__init__()

        self.num_entity1, self.num_entity2 = num_entity1, num_entity2
        self.layers = layers
        self.graph = graph

    def compute(self, entity1_emb, entity2_emb):
        """LightGCN forward propagation"""
        all_emb = torch.cat([entity1_emb, entity2_emb])
        embeddings = [all_emb]

        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embeddings.append(all_emb)
        embeddings = torch.mean(torch.stack(embeddings, dim=1), dim=1)
        entity1, entity2 = torch.split(embeddings, [self.num_entity1, self.num_entity2])
        return entity1, entity2

    def forward(self, entity1_emb, entity2_emb):
        return self.compute(entity1_emb, entity2_emb)


class GraphConvolution(nn.Module):
    """Convolution for graph"""

    def __init__(
        self, num_user, num_group, num_item, layers, user_item_graph, group_item_graph
    ):
        super(GraphConvolution, self).__init__()

        self.num_user = num_user
        self.num_group = num_group
        self.num_item = num_item

        self.layers = layers

        self.user_item_graph = user_item_graph
        self.group_item_graph = group_item_graph

        self.gate0 = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.gate1 = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, user_emb, group_emb, item_emb):
        ui_emb = torch.cat([user_emb, item_emb])
        embeddings_ui = [ui_emb]
        for _ in range(self.layers):
            ui_emb = torch.sparse.mm(self.user_item_graph, ui_emb)
            embeddings_ui.append(ui_emb)

        # 获得用户与物品各层的emb
        user_emb_ui_layers = [emb[: self.num_user] for emb in embeddings_ui]
        item_emb_ui_layers = [emb[self.num_user :] for emb in embeddings_ui]

        user_emb_for_user = torch.mean(torch.stack(user_emb_ui_layers, dim=1), dim=1)
        item_emb_for_user = torch.mean(torch.stack(item_emb_ui_layers, dim=1), dim=1)

        gi_emb = torch.cat([group_emb, item_emb_for_user])

        embeddings_gi = [gi_emb]
        for _ in range(self.layers):
            gi_emb = torch.sparse.mm(self.group_item_graph, gi_emb)
            embeddings_gi.append(gi_emb)

        group_emb_gi_layers = [emb[: self.num_group] for emb in embeddings_gi]
        item_emb_gi_layers = [emb[self.num_group :] for emb in embeddings_gi]
        i_emb = torch.mean(torch.stack(item_emb_gi_layers, dim=1), dim=1)

        g_emb_layers = []
        for i in range(self.layers + 1):
            if i % 2 == 0:
                g_emb_layers.append(group_emb_gi_layers[i])
            else:
                emb_in = torch.cat([group_emb, item_emb_ui_layers[i - 1]])
                emb_out = torch.sparse.mm(self.group_item_graph, emb_in)
                g_emb_layers.append(emb_out[: self.num_group])

        g_emb = torch.mean(torch.stack(g_emb_layers, dim=1), dim=1)

        return user_emb_for_user, g_emb, i_emb


class OverlapGraphConvolution(nn.Module):
    """Graph Convolution for Group-level graph"""

    def __init__(self, layers, graph):
        super(OverlapGraphConvolution, self).__init__()
        self.layers = layers
        self.graph = graph

    def forward(self, embedding):
        group_emb = embedding
        final = [group_emb]

        for _ in range(self.layers):
            group_emb = torch.mm(self.graph, group_emb)
            final.append(group_emb)

        final_emb = torch.sum(torch.stack(final), dim=0)
        return final_emb


# 当前 Q：组，K：用户，V：用户
class AttentionAggregator(nn.Module):

    # 物品数量，每组最大用户数，嵌入维度
    def __init__(self, emb_dim):
        super(AttentionAggregator, self).__init__()

        # 注意力机制中Q与K的维度为self.dim
        self.dim = 64

        self.wq = nn.Parameter(torch.Tensor(emb_dim, self.dim))
        self.wk = nn.Parameter(torch.Tensor(emb_dim, self.dim))
        nn.init.xavier_uniform_(self.wq)
        nn.init.xavier_uniform_(self.wk)

    def forward(self, users_emb, group_emb, mask=None):
        # q.shape=[batch_size,emb_dim]
        q = torch.matmul(group_emb, self.wq)
        # k.shape=[batch_size,max_g_len,emb_dim]
        k = torch.matmul(users_emb, self.wk)
        v = users_emb

        attention = torch.einsum("BD,BDL->BL", q, k.transpose(-1, -2))
        attention = attention / np.power(self.dim, 0.5)  # 缩放

        # mask.shape=[batch_size,max_g_len]
        if mask is None:
            weight = torch.softmax(attention, dim=1)
        else:
            weight = torch.softmax(attention + mask, dim=1)

        # weight.shape=[batch_size,max_g_len]
        # v.shape=[batch_size,max_g_len,emb_dim]
        out = torch.einsum("BL,BLE->BE", weight, v)

        return out


class SocialRelationshipAgg(nn.Module):

    def __init__(self, layers, overlap_user, members_of_group, mask):
        super(SocialRelationshipAgg, self).__init__()

        self.overlap_user = overlap_user
        self.members_of_group = members_of_group
        self.mask = mask

        self.layers = layers

        # self.linear = nn.Sequential(
        #     nn.Linear(32, int(32 / 2)),
        #     nn.ReLU(),
        #     nn.Linear(int(32 / 2), 1),
        # )

    def forward(self, users_emb):

        user_embeddings = [users_emb]
        for _ in range(self.layers):
            users_emb = torch.mm(self.overlap_user, users_emb)
            user_embeddings.append(users_emb)

        users_emb = torch.sum(torch.stack(user_embeddings, dim=0), dim=0)

        user_emb_of_group = users_emb[self.members_of_group]

        # 注意力
        # out = self.linear(user_emb_of_group).squeeze()

        # mask = self.mask == 0
        # out.masked_fill_(mask, -np.inf)
        # weight = torch.softmax(out, dim=1)
        # group_emb = torch.matmul(weight.unsqueeze(1), user_emb_of_group).squeeze()

        group_emb = torch.einsum("BL,BLE->BE", self.mask, user_emb_of_group)

        return group_emb


class Gate(nn.Module):

    def __init__(self, emb_dim, num_groups):
        super(Gate, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid(),
        )
        self.num_groups = num_groups

    def forward(self, x):
        weight = self.linear(x)
        return (
            weight[0 : self.num_groups],
            weight[self.num_groups : 2 * self.num_groups],
        )


class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0.0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.linear(x)


class HRDPRec(nn.Module):

    def __init__(
        self,
        num_users,
        num_groups,
        num_items,
        config,
        user_item_graph,
        group_item_graph,
        overlap_user,
        overlap_item,
        all_group_users,
        all_group_mask,
    ):
        super(HRDPRec, self).__init__()

        # Hyperparameters
        emb_dim = config.emb_dim
        layers = config.layers
        social_layers = config.social_layers
        drop_ratio = config.drop_ratio

        self.all_group_users = all_group_users
        self.all_group_mask = all_group_mask

        self.num_groups = num_groups

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.group_embedding = nn.Embedding(num_groups, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)

        self.lightGCN_user_item = LightGCN(
            num_users, num_items, layers, user_item_graph
        )

        self.item_overlap_convolution = OverlapGraphConvolution(
            social_layers, overlap_item
        )

        self.graph_convolution = GraphConvolution(
            num_users,
            num_groups,
            num_items,
            layers,
            user_item_graph,
            group_item_graph,
        )

        self.user_social_agg = SocialRelationshipAgg(
            social_layers, overlap_user, all_group_users, all_group_mask
        )

        self.gate0 = Gate(emb_dim, num_groups)

        # Prediction Layer
        self.predict = PredictLayer(emb_dim, drop_ratio)

    def userforward(self, user_inputs, item_inputs):

        user_emb, item_emb = self.lightGCN_user_item(
            self.user_embedding.weight,
            self.item_embedding.weight,
        )

        u_emb = user_emb[user_inputs]
        i_emb = item_emb[item_inputs]

        return torch.sigmoid(self.predict(u_emb * i_emb))

    def forward(self, group_inputs, item_inputs):

        init_all_user_emb = self.user_embedding.weight
        init_all_group_emb = self.group_embedding.weight
        init_all_item_emb = self.item_embedding.weight

        all_user_emb, group_emb_itemlayer, item_emb_final = self.graph_convolution(
            init_all_user_emb, init_all_group_emb, init_all_item_emb
        )

        # # 获取各组的成员emb
        group_emb_userlayer = self.user_social_agg(all_user_emb)

        weight1, weight2 = self.gate0(
            torch.cat([group_emb_itemlayer, group_emb_userlayer])
        )

        group_emb_final = weight1 * group_emb_itemlayer + weight2 * group_emb_userlayer

        item_emb_final = self.item_overlap_convolution(item_emb_final)

        g_emb = group_emb_final[group_inputs]
        i_emb = item_emb_final[item_inputs]

        return torch.sigmoid(self.predict(g_emb * i_emb))

    def L2_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter**2)
        return loss / 2

    def GetUserEmbedding(self):
        init_all_user_emb = self.user_embedding.weight
        init_all_group_emb = self.group_embedding.weight
        init_all_item_emb = self.item_embedding.weight

        user_final_embeddings, group_emb_itemlayer, item_emb_final = (
            self.graph_convolution(
                init_all_user_emb, init_all_group_emb, init_all_item_emb
            )
        )

        users_emb = user_final_embeddings
        user_embeddings = [user_final_embeddings]
        for _ in range(3):
            users_emb = torch.mm(self.user_social_agg.overlap_user, users_emb)
            user_embeddings.append(users_emb)

        user_consensus_information = torch.sum(
            torch.stack(user_embeddings, dim=0), dim=0
        )

        return (
            user_final_embeddings.cpu().detach().numpy(),
            user_consensus_information.cpu().detach().numpy(),
        )
