from layer.layers import FeaturesEmbedding, AttentionalFactorizationMachine, TreesLayer, MultiLayerPerceptron
from typing import Dict, List
import torch
import torch.nn.functional as F


class AFM(torch.nn.Module):
    def __init__(self, field_dims: List,
                 embed_dim: int,
                 continue_feature_num: int,
                 hiden_dims: List,
                 attn_size: int,
                 dropout_rate: float,
                 use_tree: bool = False, **kwargs
                 ):
        super(AFM, self).__init__()

        self.num_fields = len(field_dims)
        self.cate_embed = FeaturesEmbedding(
            field_dims, embed_dim)
        self.conti_embed = torch.nn.Linear(
            continue_feature_num, embed_dim)
        self.mlp = MultiLayerPerceptron(
            input_dim=embed_dim, hiden_dims=hiden_dims, dropout=dropout_rate, output_layer=False)

        self.afm = AttentionalFactorizationMachine(
            embed_dim, attn_size, dropout_rate, output_layer=False)

        out_feat_dim = hiden_dims[-1] + embed_dim
        self.use_tree = use_tree
        if self.use_tree:
            output_logits_dim = kwargs['output_logits_dim']
            smooth_step_param = kwargs['smooth_step_param']
            sum_outputs = kwargs['sum_outputs']
            trees_num = kwargs['trees_num']
            depth = kwargs['depth']
            self.trees = TreesLayer(input_dim=self.num_fields + continue_feature_num, output_logits_dim=output_logits_dim,
                                    smooth_step_param=smooth_step_param, sum_outputs=sum_outputs,
                                    trees_num=trees_num, depth=depth)
            if sum_outputs:
                out_feat_dim += output_logits_dim
            else:
                out_feat_dim += output_logits_dim*trees_num
        self.proj = torch.nn.Linear(out_feat_dim, 1)
        self.criterion = torch.nn.BCELoss()

    def forward(self, categeory_feature, continue_feature, **kwargs):
        categeory_feature_emb = self.cate_embed(categeory_feature)
        continue_feature_emb = self.conti_embed(continue_feature)
        mlp = self.mlp(continue_feature_emb)
        afm = self.afm(categeory_feature_emb)
        proj = torch.cat([mlp, afm], dim=-1)
        if self.use_tree:
            x = torch.cat([categeory_feature, continue_feature], dim=1)
            tree = self.trees(x)
            proj = torch.cat([proj, tree], dim=-1)

        out = self.proj(proj).squeeze(-1)
        return {'logist': F.sigmoid(out)}

    def compute_loss(self, logist, y, **kwargs):
        loss = self.criterion(logist, y)
        return loss
