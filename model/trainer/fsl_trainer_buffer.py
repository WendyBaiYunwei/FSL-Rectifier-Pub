import time
import os
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    prepare_model, prepare_optimizer,
)
from model.image_translator.utils import (
    get_recon, get_trans, get_sim,
    get_dichomy_loader, loader_from_list,
    get_augmentations
)
from model.utils import (
    Averager, count_acc,
    compute_confidence_interval
)
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args, config):
        super().__init__(args)
        self.config = config

        self.test_loader_fsl = get_dichomy_loader(
            episodes=config['max_iter'],
            root=config['data_folder_test'],
            file_list=config['data_list_test'],
            batch_size=config['batch_size'],
            new_size=config['new_size'],
            height=config['crop_image_height'],
            width=config['crop_image_width'],
            crop=True,
            num_workers=4,
            return_paths=True,
            n_cls=config['way_size'],
            dataset=config['dataset'])

        self.train_loader_image_translator = loader_from_list(
            root=config['data_folder_train'],
            file_list=config['data_list_train'],
            batch_size=config['pool_size'],
            new_size=config['new_size'],
            height=config['crop_image_height'],
            width=config['crop_image_width'],
            crop=True,
            num_workers=4,
            return_paths=True,
            dataset=config['dataset'])
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.eval_way, dtype=torch.int8).\
            repeat(args.eval_shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def train(self):
        args = self.args
        loaded_dict = torch.load(args.model_path, map_location='cuda:1')
        if 'state_dict' in loaded_dict:
            loaded_dict = loaded_dict['state_dict']
        else:
            loaded_dict = loaded_dict['params']
        new_params = {}
        keys = list(loaded_dict.keys())

        for key in self.model.state_dict().keys():
            if key in keys:
                new_params[key] = loaded_dict[key]
            else:
                new_params[key] = self.model.state_dict()[key]
        self.model.load_state_dict(new_params)
        self.model.train()
        
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        
        tl1 = Averager()
        tl2 = Averager()
        ta = Averager()

        for it, batch in enumerate(self.train_loader):
            self.train_step += 1

            if self.train_step > args.max_epoch * args.episodes_per_epoch:
                break
            if torch.cuda.is_available():
                data, gt_label = [_.cuda() for _ in batch]
            else:
                data, gt_label = batch[0], batch[1]
            
            data_tm = time.time()

            # get saved centers
            logits, reg_logits = self.para_model(data)
            if reg_logits is not None:
                loss = F.cross_entropy(logits, label)
                total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
            else:
                loss = F.cross_entropy(logits, label)
                total_loss = F.cross_entropy(logits, label)
                
            tl2.add(loss)
            forward_tm = time.time()
            self.ft.add(forward_tm - data_tm)
            acc = count_acc(logits, label)

            tl1.add(total_loss.item())
            ta.add(acc)

            self.optimizer.zero_grad()
            total_loss.backward()
            backward_tm = time.time()
            self.bt.add(backward_tm - forward_tm)

            self.optimizer.step()
            optimizer_tm = time.time()
            self.ot.add(optimizer_tm - backward_tm)    

        if not osp.exists(args.save_path):
            os.mkdir(args.save_path)

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).\
            repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        augtype = args.aug_type

        if args.dataset == 'animals' and 'translator' in augtype: 
            from model.image_translator.trainer import Translator_Trainer
            translator = Translator_Trainer(self.config)
            translator.cuda()
            translator.load_ckpt('animals_gen.pt')
            translator = translator.model
            translator.eval()
            self.translator = translator
        loaded_dict = torch.load(args.model_path)
        if 'state_dict' in loaded_dict:
            loaded_dict = loaded_dict['state_dict']
        else:
            loaded_dict = loaded_dict['params']
        new_params = {}
        keys = list(loaded_dict.keys())

        for key in self.model.state_dict().keys():
            if key in keys:
                new_params[key] = loaded_dict[key]
            else:
                new_params[key] = self.model.state_dict()[key]

        self.model.load_state_dict(new_params) 
        self.model.eval()
        baseline = np.zeros((args.num_eval_episodes, 2)) 

        expansion_res = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        old_shot = args.eval_shot
        old_qry = args.eval_query
        if augtype != 'true-test':
            assert args.eval_query == 1
        qry_expansion = args.qry_expansion
        spt_expansion = args.spt_expansion
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader_fsl)):
                if i >= args.num_eval_episodes:
                    break
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch[:-1]]
                else:
                    data = batch[0]
                paths = batch[-1]

                if augtype == 'true-test':
                    assert args.eval_query == args.qry_expansion + 1
                    assert args.eval_shot == args.spt_expansion + 1
                    logits = self.model(data, qry_expansion=qry_expansion, \
                        spt_expansion=spt_expansion, augtype=augtype)
                    label = label[:self.args.eval_way]
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                    expansion_res[i-1, 0] = loss.item()
                    expansion_res[i-1, 1] = acc
                    baseline[i-1, 0] = loss.item()
                    baseline[i-1, 1] = acc
                    continue
                
                original_spt = data[:self.args.eval_way, :, :, :]
                img_names = paths[:self.args.eval_way]
                if self.args.dataset == 'animals' and 'translator' in augtype:
                    recon_spt = self.get_class_expansion(original_spt,\
                                expansion=1, img_names=img_names, \
                                augtype='original')
                else:
                    recon_spt = original_spt

                '''Expand support'''
                # no support expansion
                if spt_expansion == 0 and augtype in ['sim-mix-up',\
                    'image_translator', 'mix-up', 'random-mix-up', \
                    'random-image_translator', 'reverse-image_translator']: 
                        combined_spt = recon_spt
                # have support expansion, use original support to expand support
                elif augtype in ['sim-mix-up', 'image_translator', 'mix-up', \
                    'random-mix-up', 'random-image_translator',\
                    'reverse-image_translator']:
                    expanded_spt = self.get_class_expansion(original_spt,\
                            spt_expansion, img_names=img_names, augtype=augtype)
                    combined_spt = torch.cat([recon_spt, expanded_spt], dim=0)
                # traditional augmentation   
                else:
                    expanded_spt = self.get_class_expansion(original_spt,\
                        spt_expansion, img_names = img_names, augtype=augtype)
                    combined_spt = torch.cat([original_spt, expanded_spt], dim=0)

                '''Expand query'''
                img_names = paths[self.args.eval_way:]
                original_qry = data[self.args.eval_way:, :, :, :]
                if self.args.dataset == 'animals' and 'translator' in augtype:
                    recon_qry = self.get_class_expansion(original_qry,\
                        expansion=1, img_names=img_names, \
                        augtype='original')
                else:
                    recon_qry = original_qry

                new_qries = torch.empty(old_qry, self.args.eval_way * qry_expansion, data.shape[1], data.shape[2],\
                    data.shape[3]).cuda()

                k = 0
                for class_chunk_i in range(self.args.eval_way, len(data), self.args.eval_way):
                    class_chunk = data[class_chunk_i:class_chunk_i+self.args.eval_way]
                    expansion = self.get_class_expansion(class_chunk, qry_expansion,\
                        img_names = img_names, augtype=augtype)
                    new_qries[k] = expansion
                    k += 1
                # original expansion
                new_qries = new_qries.flatten(end_dim=1)
                if qry_expansion > 0 and augtype in ['sim-mix-up', \
                    'image_translator', 'mix-up',\
                     'random-mix-up', 'random-image_translator', \
                     'reverse-image_translator']: 
                    expanded_data = torch.cat([combined_spt, recon_qry, \
                    new_qries], dim=0)
                elif qry_expansion == 0 and augtype in ['sim-mix-up', \
                'image_translator', 'mix-up',\
                     'random-mix-up', 'random-image_translator', \
                     'reverse-image_translator']: 
                    expanded_data = torch.cat([combined_spt, recon_qry], dim=0)
                elif qry_expansion == 0:
                    expanded_data = torch.cat([combined_spt, original_qry], dim=0)
                else:
                    expanded_data = torch.cat([combined_spt, original_qry, \
                    new_qries], dim=0)

                data = torch.cat([original_spt, original_qry], dim=0)
                logits = self.model(data) # get baseline results
                loss = F.cross_entropy(logits, label)
                baseline[i-1, 0] = loss.item()
                baseline[i-1, 1] = count_acc(logits, label)

                
                logits = self.model(expanded_data, spt_expansion=spt_expansion,\
                     qry_expansion=qry_expansion)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                expansion_res[i-1, 0] = loss.item()
                expansion_res[i-1, 1] = acc

        assert(i == baseline.shape[0])
        vl, _ = compute_confidence_interval(baseline[:,0])
        va, vap = compute_confidence_interval(baseline[:,1])
        
        result_str = ''
        result_str += 'Baseline Test acc={:.4f} + {:.4f}\n'.format(va, vap)
        
        vl, _ = compute_confidence_interval(expansion_res[:,0])
        va, vap = compute_confidence_interval(expansion_res[:,1])
        
        result_str += f'{augtype} Test acc={va} + {vap}\n'

        with open(f'./outputs/{args.model_class}-{args.backbone_class}-{args.dataset}-{args.use_euclidean}-' +\
        f'{augtype}-{args.spt_expansion}-{args.qry_expansion}-record-{args.note}.txt', 'w') as file:
            file.write(result_str)
        return vl, va, vap
    
    # input (filename) abcde, return a1a2a3b1b2b3 (additional images for augmentation)
    # 'oracle', 'mix_up', 'affine', 'color', 'image_translator'...
    def get_class_expansion(self, data, expansion, augtype='image_translator', img_names=''):
        expanded = torch.empty(self.args.eval_way, expansion, data.shape[1], data.shape[2], data.shape[3]).cuda()
        for class_i in range(self.args.eval_way):
            img = img_names[class_i]
            img_data = data[class_i, :, :, :]
            if augtype == 'image_translator':
                class_expansions = get_trans(img_names[class_i], expansion_size=expansion)
            elif augtype == 'sim-mix-up':
                class_expansions = get_sim(img, expansion_size=expansion, dataset=self.args.dataset)
            elif augtype == 'original':
                class_expansions = get_recon(img)
            else: # traditional augmentation
                class_expansions = get_augmentations(img_data, expansion, augtype, get_img=False)
            if class_expansions == None:
                return None
            expanded[class_i] = class_expansions
        expanded = expanded.swapaxes(0, 1).flatten(end_dim=1)
        return expanded
        
    def final_record(self):
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))           
