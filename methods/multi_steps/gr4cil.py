import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from argparse import ArgumentParser

from backbone.my_clip.model import GR4CIL, ResidualHead
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy
import math
from transformers import CLIPProcessor
import json
from utils.data_manager import DataManager
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, roc_auc_score
from utils.toolkit import accuracy, cal_bwf, mean_class_recall
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

EPSILON = 1e-8


def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--lambd", type=int, default=None, help='hyperparameter of image to image similarity')
    parser.add_argument("--test_bs", type=int, default=128, help='test batch size')
    parser.add_argument("--is_OOD_test", type=bool, default=False, help="Whether to do OOD testing")
    parser.add_argument("--T", type=float, default=None, help='Temperature')


class gr4cil(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._lambda = config.lambd
        self._is_openset_test = config.is_OOD_test
        self._T = config.T
        self._test_bs = config.test_bs
        self._init_cls = config.init_cls
        self._increment = config.increment
        self._til_record = []
        self._cil_record = []
        self._tid_record = []
        self.task_metric_curve = []
        self.cnn_metric_curve = []
        self.positive_means = []
        self.negative_means = []
        if self._is_openset_test:
            self._AUC_record = []
            self._FPR95_record = []

        if config.backbone == "clip_vit_b_16_224":
            self._pretrained_weights_path = "pretrain_weights/clip_vit_base_patch16"
        ##
        self._clip_process = CLIPProcessor.from_pretrained(self._pretrained_weights_path)

        if config.dataset == 'cifar100_i2t' or config.dataset == 'cifar100_i2t_few_shot':
            desp_json = 'datasets/cifar100_prompts_base.json'
        elif config.dataset == 'imagenetr_i2t':
            desp_json = 'datasets/I2T_Imagenet_r.json'
        elif config.dataset == 'imagenet1000_i2t' or config.dataset == 'imagenet100_i2t_new':
            desp_json = 'datasets/I2T_Imagenet_1000.json'


        id_class_desp = []
        with open(desp_json) as f:
            id_texts = json.load(f)

        id_class_desp = []

        for i in range(len(id_texts[list(id_texts.keys())[0]])):
            id_class_desp.append([id_texts[label][i] for label in list(id_texts.keys())])

        self._id_text_embeddings = {}
        # tokenizer
        for i in range(len(id_class_desp)):
            self._id_text_embeddings.update(
                {i: self._clip_process.tokenizer(id_class_desp[i], return_tensors='pt', padding=True)})
            
        self.text_tokens = self._id_text_embeddings[0]

        self._logger.info('Applying GR4CIL (a class incremental method, test with {})'.format(self._incre_type))

        self._network = GR4CIL(self._logger, self._pretrained_weights_path)

    def prepare_model(self):
        self._cur_task += 1
        # update_visual_encoder
        if self._cur_task > 0:
            self._network.update_visual_encoder()
        self._network = self._network.cuda()

    def prepare_task_data(self, data_manager_ID):
        self._cur_classes = data_manager_ID.get_task_size(self._cur_task)
        print("self._known_classes", self._known_classes)
        print("self._cur_classes", self._cur_classes)
        self._total_classes = self._known_classes + self._cur_classes

        self._train_dataset_ID = data_manager_ID.get_dataset(
            indices=np.arange(self._known_classes, self._total_classes),
            source='train', mode='train')
        self._train_dataset_ID_prototype = data_manager_ID.get_dataset(
            indices=np.arange(self._known_classes, self._total_classes),
            source='train', mode='test')

        self._test_dataset = data_manager_ID.get_dataset(indices=np.arange(0, self._total_classes), source='test',
                                                         mode='test')
        self._test_dataset_fc = data_manager_ID.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                                                            source='test', mode='test')
        self._openset_test_dataset = data_manager_ID.get_openset_dataset(
            known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._cur_task_test_samples_num = len(self._test_dataset)

        self._logger.info('Train dataset of ID size: {}'.format(len(self._train_dataset_ID)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))
        self._logger.info('Test dataset of current task size: {}'.format(len(self._test_dataset_fc)))

        self._train_loader_ID = DataLoader(self._train_dataset_ID, batch_size=self._batch_size, shuffle=True,
                                           num_workers=self._num_workers)
        self._train_loader_prototype = DataLoader(self._train_dataset_ID_prototype, batch_size=self._batch_size,
                                                  shuffle=False, num_workers=self._num_workers)

        self._test_loader = DataLoader(self._test_dataset, batch_size=self._test_bs, shuffle=False,
                                       num_workers=self._num_workers)
        self._test_fc_loader = DataLoader(self._test_dataset_fc, batch_size=self._test_bs, shuffle=False,
                                          num_workers=self._num_workers)

        self._iters_per_epoch_lora = math.ceil(len(self._train_dataset_ID) * 1.0 / self._batch_size)

        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False,
                                               num_workers=self._num_workers)

        self._order = torch.tensor(data_manager_ID._class_order)

    def build_text_basis(self, task_text_features, r=256):
        T = task_text_features.float()
        # SVD: T = U S Vh, Vh: [min(Ck,D), D]
        _, _, Vh = torch.linalg.svd(T, full_matrices=False)
        r = min(r, Vh.shape[0])
        B_T = Vh[:r].T.contiguous()  # [D, r]
        return B_T


    def incremental_train(self):
        # train lora
        self._logger.info("Training current task-special visual lora and share lora with data of current task!")
        self._network = self._network.cuda()
        self._network.train_lora_mode()

        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        optimizer = self._get_optimizer(self._network.parameters(), self._config, False)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=self._epochs * self._iters_per_epoch_lora)
        
        self._network = self._train_model(self._network, self._train_loader_ID, self.text_tokens, optimizer, scheduler,
                                              test_loader=self._test_fc_loader,
                                              task_id=self._cur_task,
                                              epochs=self._epochs, note='_')

        task_text_features = self._cal_text_feature(self._network, self.text_tokens, task_id=self._cur_task)

        # cal prototype
        self._cal_image_prototype(self._network, self._train_loader_ID, task_id=self._cur_task, task_text_features=task_text_features, train_res=True)

        self._network.test_mode()

        # save lora and prototype
        self._save_checkpoint('seed{}_task{}_checkpoint.pkl'.format(self._seed, self._cur_task),
                              self._network.cpu())

    def _train_model(self, model, train_loader, text_tokens, optimizer, scheduler, test_loader=None, task_id=None, epochs=100, note=''):
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        if note != '':
            note += '_'
        self._scaler = GradScaler()
        gap_scale = 1

        for epoch in range(epochs):
            model, train_losses, positive_mean, negative_mean = self._epoch_train(model, train_loader, optimizer,
                                                                                  scheduler,
                                                                                  text_tokens=text_tokens,
                                                                                  task_begin=task_begin,
                                                                                  task_end=task_end, task_id=task_id,
                                                                                  gap_scale=gap_scale)
            info = (
                    'Task {}, Epoch {}/{} => '.format(task_id, epoch + 1, epochs)
                    + ('{} {:.3f}, ' * (len(train_losses) // 2)).format(*train_losses)
                    + 'pos {:.3f}, neg {:.3f}'.format(positive_mean.item(), negative_mean.item())
            )

            self._logger.info(info)

        self.positive_means.append(positive_mean)
        self.negative_means.append(negative_mean)

        cur_classes = self._order[task_begin : task_end]
        with torch.no_grad():
            id_text_features = model.get_texts_feature(text_tokens["input_ids"][cur_classes].cuda(), text_tokens["attention_mask"][cur_classes].cuda())
            id_text_features /= id_text_features.norm(p=2, dim=-1, keepdim=True)
        test_acc = self._epoch_test(model, test_loader,  text_features=id_text_features, task_begin=task_begin, task_end=task_end, task_id=task_id)

        info = info + 'test_acc {:.3f}, '.format(test_acc)
        self._logger.info(info)

        self._til_record.append(test_acc)
        self._logger.info(
            "TIL: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t" * len(self._til_record)).format(
                *self._til_record) + ']')
        return model

    def _epoch_train(self, model, train_loader, optimizer, scheduler, text_tokens, task_begin=None,
                     task_end=None, task_id=None, gap_scale=None):
        losses = 0
        total = 0
        correct = 0
        clip_losses = 0.
        text_distance_losses = 0.
        text_unchange_losses = 0.

        model.train()
        positive_outputs = []
        negative_outputs = []

        cur_pre_classes = self._order[:task_end]

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            targets = targets - task_begin
            with autocast():
                image_embeds, text_features = model(inputs, text_tokens["input_ids"][cur_pre_classes].cuda(), text_tokens["attention_mask"][cur_pre_classes].cuda())  # forward(self, image_inputs, input_ids, attention_mask)

            image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  
            cur_classes_text_features = text_features[task_begin:task_end,:]

            gap_logits_per_image = torch.matmul(image_features.float(), text_features.float().t())
            one_hot_targets = torch.nn.functional.one_hot(targets.long(), gap_logits_per_image.shape[1]).float()
            positive_outputs.append((gap_logits_per_image * one_hot_targets).sum(dim=1).mean())
            mask = 1 - one_hot_targets
            negative_outputs.append(((gap_logits_per_image * mask).sum(dim=1) / mask.sum(dim=1)).mean())

            logit_scale = model.logit_scale.exp()
            logits_per_image = torch.matmul(image_features.float(), cur_classes_text_features.float().t()) * logit_scale
            clip_loss = self._clip_loss(logits_per_image, targets)

            if task_id == 0:
                text_distance_loss = self._text_distance_loss(cur_classes_text_features, threshold=0.7)
                loss = clip_loss + 1.0 * text_distance_loss
            else:
                old_classes_text_features = text_features[:task_begin,:]
                text_distance_loss = self._text_distance_loss(cur_classes_text_features, old_text=old_classes_text_features, threshold=0.7)
                text_unchange_loss = self._text_cos_loss(text_features[:task_begin,:], self.pre_text_features)

                loss = clip_loss + 1.0 * text_distance_loss + 1.0 * text_unchange_loss

            optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.step(optimizer)
            self._scaler.update()


            clip_losses += clip_loss.item()
            text_distance_losses += text_distance_loss.item()
            if task_id > 0:
                text_unchange_losses += text_unchange_loss.item()

            _, predicted = (logits_per_image.max(1))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            losses += loss.item()

            if scheduler != None:
                scheduler.step()

        train_loss_acc = ['Loss', losses/len(train_loader),  'clip_loss', clip_losses/len(train_loader),  
                          'text_distance_loss', text_distance_losses/len(train_loader),   'text_unchange_loss',
                          text_unchange_losses/len(train_loader), 'train_acc', correct / (total+EPSILON)*100]

        positive_mean = sum(positive_outputs) / len(positive_outputs)
        negative_mean = sum(negative_outputs) / len(negative_outputs)

        return model, train_loss_acc, positive_mean.detach(), negative_mean.detach()
    

    def _epoch_test(self, model, test_loader, text_features=None, task_begin=None, task_end=None, task_id=None):

        correct = 0.
        total = 0

        model.eval()
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                targets = targets - task_begin
                with autocast():
                    image_embeds = model.get_images_feature(inputs)

                image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits_per_image = torch.matmul(image_features.float(), text_features.t()) * logit_scale

                _, predicted = (logits_per_image.max(1))
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = correct / (total + EPSILON) * 100

        return test_acc
    
    def _cal_text_feature(self, model, text_tokens, task_id=None):
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        cur_pre_classes = self._order[:task_end]
        with torch.no_grad():
            self.pre_text_features = model.get_texts_feature(text_tokens["input_ids"][cur_pre_classes].cuda(), text_tokens["attention_mask"][cur_pre_classes].cuda())
            self.pre_text_features /= self.pre_text_features.norm(p=2, dim=-1, keepdim=True)

        self.pre_text_embeddings = {}
        for i in range(task_id+1):
            task_begin = sum(self._increment_steps[:i])
            task_end = task_begin + self._increment_steps[i]
            self.pre_text_embeddings.update({i : self.pre_text_features[task_begin:task_end,:]})

        return self.pre_text_features[task_begin:task_end,:]

    def _cal_image_prototype(self, model, train_loader, task_id=None, task_text_features=None, train_res=False):
        # peak_mem = torch.cuda.max_memory_allocated('cuda') / 1024**2
                # logger.info(peak_mem)
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        cur_num_classes = self._increment_steps[task_id]
        self.image_prototype = torch.zeros(cur_num_classes, 512).float().cuda()
        self.prototypes_sim = torch.zeros(cur_num_classes, 512).float().cuda()
        # model.eval()
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                targets = targets - task_begin
                with autocast():
                    image_embeds = model.get_images_feature(inputs)
                image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                targets = targets.long()
                self.image_prototype.scatter_add_(0, targets.unsqueeze(1).expand(-1, image_features.size(1)),
                                                  image_features.float())
            self.image_prototype /= self.image_prototype.norm(p=2, dim=-1, keepdim=True)

        if train_res:
            Ck = task_end - task_begin
            D = task_text_features.shape[1]
            model.res_head = ResidualHead(D, Ck).cuda()

            with torch.no_grad():
                B_T = self.build_text_basis(task_text_features)
                proto = self.image_prototype.detach().float()
                proto_perp = self.project_prototypes_orthogonal(proto, B_T)          # [Ck, D]
                proto_perp = proto_perp / (proto_perp.norm(p=2, dim=-1, keepdim=True) + 1e-8)        # [10, 512]
                model.res_head.U.copy_(proto_perp.t().contiguous())
            model.res_head.B_T = B_T

            # stage-2: train residual head only
            model.train_projector_mode()

            res_opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)

            epochs = 3
            for e in range(epochs):
                losses = 0
                for _, inputs, targets in train_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    targets = targets - task_begin

                    with torch.no_grad(), autocast():
                        image_embeds = model.get_images_feature(inputs)
                    x = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

                    # fixed text logits
                    logit_scale = model.logit_scale.exp()

                    z_res = model.res_head(x.float())

                    logits = (z_res) * logit_scale

                    loss = F.cross_entropy(logits, targets.long())

                    res_opt.zero_grad()
                    loss.backward()
                    res_opt.step()

                    losses += loss.item()

                train_loss_acc = ['Loss', losses / len(train_loader)]

                info = (
                            'Task {}, Epoch {}/{} => '.format(task_id, e + 1, epochs)
                            + ('{} {:.3f}, ' * (len(train_loss_acc) // 2)).format(*train_loss_acc))
                
                self._logger.info(info)

    def eval_task(self):
        # Prepare checkpoints for each stage
        # seed_checkpoint_paths: Save the checkpoint paths obtained after continual learning under each seed

        checkpoint_paths = [i for i in os.listdir(self._logdir) if i.endswith('.pkl')]
        chks_path = self._logdir
        if self._config.test_dir:
            checkpoint_paths = [i for i in os.listdir(self._config.test_dir) if i.endswith('.pkl')]
            chks_path = self._config.test_dir
        checkpoint_paths.sort()
        seed_checkpoint_paths = {}
        for path in checkpoint_paths:
            splited_text = path.split('_')
            checkpoint_seed = int(splited_text[0].replace('seed', ''))

            if re.match('task[0-9]+$', splited_text[1]):  # for multi_steps checkpoints
                checkpoint_task_id = int(splited_text[1].replace('task', ''))
            else:  # for single_step checkpoints
                checkpoint_task_id = 0

            # gather checkpoints with the same random seed into a group
            if checkpoint_seed in seed_checkpoint_paths.keys():
                seed_checkpoint_paths[checkpoint_seed][checkpoint_task_id] = path
            else:
                seed_checkpoint_paths[checkpoint_seed] = {checkpoint_task_id: path}

        chk_paths = seed_checkpoint_paths[self._seed]

        pre_tasks_classes = torch.tensor(
            [sum(self._increment_steps[:i]) for i in range(len(self._increment_steps))]).cuda()
        if self._is_openset_test and self._cur_task < self._nb_tasks - 1:
            self._test_loader = self._openset_test_loader

        for cur_task in range(len(chk_paths)):
            chk_name = chk_paths[cur_task]
            class_num = self._increment_steps[cur_task]
            tmp_checkpoint = torch.load(os.path.join(chks_path, chk_name))
            self._network.load_state_dict(tmp_checkpoint, strict=False)
            self.image_prototype = tmp_checkpoint['image_prototype']
            self.prototypes_sim = tmp_checkpoint['prototypes_sim']
            self._network.res_head.B_T = tmp_checkpoint['B_T']

            self.image_prototype = self.image_prototype.cuda()
            self.prototypes_sim = self.prototypes_sim.cuda()
            self._network = self._network.cuda()
            task_begin = sum(self._increment_steps[:cur_task])
            task_end = task_begin + self._increment_steps[cur_task]
            cur_task_text_features = self.pre_text_features[task_begin:task_end,:]

            self._network.eval()
            idx = 0

            cal = self.positive_means[cur_task] / (torch.tensor(self.positive_means).mean())
            with torch.no_grad():
                for _, inputs, targets in self._test_loader:
                    idx = idx + 1
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with autocast():
                        image_embeds = self._network.get_images_feature(inputs)
                    image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

                    x = image_features.float()


                    z_text = (x @ cur_task_text_features.t().float())


                    B_T = self._network.res_head.B_T.float()
                    U = self._network.res_head.U.float()
                    U_perp = U - B_T @ (B_T.t() @ U)
                    U_perp = F.normalize(U_perp, dim=0)

                    z_res = (x @ U_perp)

                    z_pro = (image_features.float() @ self.image_prototype.T.float())

                    logits_per_image = z_text + 0.2 * z_res + 0.2 * z_pro

                    ood_scores_per_image, predicted = torch.max(logits_per_image, dim=1, keepdim=True)
                    if idx == 1:
                        cur_task_ood_scores = ood_scores_per_image
                        cur_task_preds = predicted
                        cur_targets = targets
                        cur_task_logits = logits_per_image
                    else:
                        cur_task_ood_scores = torch.cat((cur_task_ood_scores, ood_scores_per_image), dim=0)
                        cur_task_preds = torch.cat((cur_task_preds, predicted), dim=0)
                        cur_targets = torch.cat((cur_targets, targets), dim=0)
                        cur_task_logits = torch.cat((cur_task_logits, logits_per_image), dim=0)

            if cur_task == 0:
                all_tasks_ood_scores = cur_task_ood_scores
                all_tasks_preds = cur_task_preds
                all_task_logits = cur_task_logits
            else:
                all_tasks_ood_scores = torch.cat((all_tasks_ood_scores, cur_task_ood_scores), dim=1)
                all_tasks_preds = torch.cat((all_tasks_preds, cur_task_preds), dim=1)
                all_task_logits = torch.cat((all_task_logits, cur_task_logits), dim=1)

        task_id_per_image = torch.argmax(all_tasks_ood_scores, dim=1)
        task_pred_per_image = all_tasks_preds[range(len(all_tasks_ood_scores)), task_id_per_image]
        all_tasks_preds = pre_tasks_classes[task_id_per_image] + task_pred_per_image
        cur_all_preds = all_tasks_preds[:self._cur_task_test_samples_num]
        cur_total = cur_targets[:self._cur_task_test_samples_num]
        if self._eval_metric == "acc":
            total = cur_total.size(0)
            correct = cur_all_preds.eq(cur_total).sum().item()
            t_id_correct = task_id_per_image[:self._cur_task_test_samples_num].eq(
                cur_total // self._increment).sum().item()
            t_id_acc = t_id_correct / total * 100
            acc = correct / total * 100
            self._logger.info(
                "After training the {}th task, the accuracy of the test set: {}".format(len(chk_paths) - 1, acc))
            self._cil_record.append(acc)
            self._tid_record.append(t_id_acc)
            self._logger.info("Task ID: {} curve of all task is [\t".format(self._eval_metric) + (
                        "{:2.2f}\t" * len(self._tid_record)).format(*self._tid_record) + ']')
        elif self._eval_metric == "mcr":
            cm = confusion_matrix(cur_total.cpu(), cur_all_preds.cpu())
            right_of_class = np.diag(cm)
            num_of_class = cm.sum(axis=1)
            task_size = cm.shape[0]
            mcr = np.around((right_of_class * 100 / (num_of_class + 1e-8)).sum() / task_size, decimals=2)
            self._logger.info(
                "After training the {}th task, the mean class recall of the test set: {}".format(len(chk_paths) - 1,
                                                                                                 mcr))
            self._cil_record.append(mcr)
        else:
            assert self._eval_metric != "mcr" and self._eval_metric != "acc", "Please enter the correct eval metric (mcr or acc)!"

        if self._is_openset_test and self._cur_task < self._nb_tasks - 1:
            labels_list = [1] * self._cur_task_test_samples_num
            labels_list.extend([0] * (len(self._openset_test_dataset) - self._cur_task_test_samples_num))
            scores = all_task_logits
            if self._T == 0 or self._T == None:
                scores_softmax = scores
            else:
                scores_softmax = torch.softmax(scores.float() / self._T, dim=1)
            max_scores = torch.max(scores_softmax, dim=1)[0]
            scores_list = max_scores.tolist()
            rocauc = roc_auc_score(labels_list, scores_list)
            fpr, tpr, thresholds = roc_curve(labels_list, scores_list)
            fpr95_idx = np.where(tpr >= 0.95)[0]
            fpr95 = fpr[fpr95_idx[0]]
            self._AUC_record.append(rocauc * 100)
            self._FPR95_record.append(fpr95 * 100)

            self._logger.info("AUC curve of all stages is [\t" + ("{:2.2f}\t" * len(self._AUC_record)).format(
                *self._AUC_record) + ']')
            self._logger.info("FPR95 curve of all stages is [\t" + ("{:2.2f}\t" * len(self._FPR95_record)).format(
                *self._FPR95_record) + ']')

        self._logger.info(
            "CIL: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t" * len(self._cil_record)).format(
                *self._cil_record) + ']')

    def _clip_loss(self, logits, labels):
        labels = labels.long()
        return F.cross_entropy(logits, labels)
    
     
    def _text_cos_loss(self, pred, targets):
        return (1 - F.cosine_similarity(pred, targets)).mean()

    def _text_distance_loss(self, new_text, old_text=None, threshold=0.7):
        if old_text is not None: 
            # old_text = old_text / old_text.norm(p=2, dim=-1, keepdim=True)   
            text_embeddings = torch.cat((new_text, old_text), dim=0)
            text_sim = torch.matmul(new_text, text_embeddings.t()).fill_diagonal_(0)
            text_sim = text_sim - threshold
            text_sim = F.relu(text_sim)
        else:
            text_sim = torch.matmul(new_text, new_text.t()).fill_diagonal_(0)
            text_sim = text_sim - threshold
            text_sim = F.relu(text_sim)
        
        return text_sim.sum()/(text_sim.size(0)*text_sim.size(1)-text_sim.size(0))

    def after_task(self):
        self._known_classes = self._total_classes

    def _save_checkpoint(self, filename, model=None):
        save_path = os.path.join(self._logdir, filename)

        # save lora and final fc
        my_state_dict = model.state_dict()
        model_state_dict = {
            k: v for k, v in my_state_dict.items()
            if ('lora_' in k) or k.startswith('res_head')
        }

        # save model config
        model_state_dict.update({'config': self._config.get_parameters_dict()})

        # save current task
        model_state_dict.update({'task_id': self._cur_task})

        # save prototype
        model_state_dict.update({'image_prototype': self.image_prototype})
        model_state_dict.update({'prototypes_sim': self.prototypes_sim})

        model_state_dict.update({'B_T': model.res_head.B_T})

        # save text features
        model_state_dict.update({'pre_text_features':self.pre_text_features})

        torch.save(model_state_dict, save_path)
        self._logger.info('checkpoint saved at: {}'.format(save_path))

    def store_samples(self):
        pass

    def project_prototypes_orthogonal(self, proto, B_T):
        """
        proto: [Ck, D]
        B_T:  [D, r]
        return proto_perp: [Ck, D]
        """
        P = proto.float()
        P_parallel = (P @ B_T) @ B_T.T    # [Ck, D]
        P_perp = P - P_parallel
        return P_perp