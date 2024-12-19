import torch
import torch.nn as nn
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.way*1)).long().view(1, 1, args.way), 
                     torch.Tensor(np.arange(args.way*1, args.way * (1 + 1))).long().view(1, 1, args.way))

    def forward(self, x, get_feature=False, qry_expansion=0, spt_expansion=0, augtype=''):
        if get_feature:
            return self.encoder(x)
        else:
            assert self.args.eval_way == self.args.way
            x = x.squeeze(0)
            instance_embs = self.encoder(x)           #len, emb
            spt_cutoff = self.args.way*(1 + spt_expansion)
            original_embs = instance_embs[:self.args.way].reshape(self.args.way, -1)
            if spt_expansion > 0:
                new_embs = instance_embs[self.args.way:spt_cutoff]
                new_embs = new_embs.reshape(spt_expansion, self.args.way, -1).\
                    mean(0)
                new_spt = 0.5 * (original_embs + new_embs)
            else:
                new_spt = original_embs

            original_embs = instance_embs[spt_cutoff:spt_cutoff+self.args.query*self.args.way].reshape(self.args.way, -1)
            if qry_expansion > 0:
                new_embs = instance_embs[spt_cutoff+1*self.args.way:]
                new_embs = new_embs.reshape(qry_expansion, self.args.way, -1).\
                    mean(0)
                new_qry = 0.5 * (original_embs + new_embs)
            else:
                new_qry = original_embs
            
            new_embs = torch.cat([new_spt, new_qry]).reshape(len(new_spt) + len(new_qry), -1)
            support_idx, query_idx = self.split_instances(x) 
            if self.training:
                logits, logits_reg = self._forward(new_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                # always 1-shot with 1 query
                if augtype != 'true-test':
                    assert self.args.shot == 1
                    assert self.args.eval_shot == 1
                    assert self.args.query == 1
                    assert self.args.eval_query == 1
                logits = self._forward(new_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')