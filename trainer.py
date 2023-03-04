"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import utils as utils
import numpy as np
import random
import torch

__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G,
	             train_loader, test_loader, settings, logger, tensorboard_logger=None,
	             opt_type="SGD", optimizer_state=None, run_count=0):
		"""
		init trainer
		"""
		
		self.settings = settings
		
		self.model = utils.data_parallel(
			model, self.settings.nGPU, self.settings.GPU)
		self.model_teacher = utils.data_parallel(
			model_teacher, self.settings.nGPU, self.settings.GPU)

		self.generator = utils.data_parallel(
			generator, self.settings.nGPU, self.settings.GPU)

		self.train_loader = train_loader
		self.test_loader = test_loader
		self.tensorboard_logger = tensorboard_logger
		self.criterion = nn.CrossEntropyLoss().cuda()
		self.bce_logits = nn.BCEWithLogitsLoss().cuda()
		self.MSE_loss = nn.MSELoss().cuda()
		self.lr_master_S = lr_master_S
		self.lr_master_G = lr_master_G
		self.opt_type = opt_type
		if opt_type == "SGD":
			self.optimizer_S = torch.optim.SGD(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				nesterov=True,
			)
		elif opt_type == "RMSProp":
			self.optimizer_S = torch.optim.RMSprop(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1.0,
				weight_decay=self.settings.weightDecay,
				momentum=self.settings.momentum,
				alpha=self.settings.momentum
			)
		elif opt_type == "Adam":
			self.optimizer_S = torch.optim.Adam(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1e-5,
				weight_decay=self.settings.weightDecay
			)
		else:
			assert False, "invalid type: %d" % opt_type
		if optimizer_state is not None:
			self.optimizer_S.load_state_dict(optimizer_state)\

		self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
											betas=(self.settings.b1, self.settings.b2))

		self.logger = logger
		self.run_count = run_count
		self.scalar_info = {}
		self.mean_list = []
		self.var_list = []
		self.teacher_running_mean = []
		self.teacher_running_var = []
		self.save_BN_mean = []
		self.save_BN_var = []

		self.BN_layer_num = 0
		for m in self.model_teacher.modules():
			if isinstance(m, nn.BatchNorm2d):
				self.BN_layer_num += 1
		# set the start layer of BNS statistics calculation 
		self.start_layer = int((self.BN_layer_num + 1) / 2) - 2
		print(f"-- Starting layer: {self.start_layer}\n")

		self.fix_G = False

	def compute_BNS_loss(self):
		# BN statistic loss
		BNS_loss = torch.zeros(1).cuda()
		
		for num in range(len(self.mean_list)):
			# statistics in different layers
			BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) \
					  + self.MSE_loss(self.var_list[num], self.teacher_running_var[num])
		BNS_loss = BNS_loss / len(self.mean_list)
		return BNS_loss
    
	def augmentation(self,images):
		"""
		random crop the generated images
		"""
		p = random.uniform(0,1)
		if p < 0.5:
			return images
		Cropper = transforms.RandomResizedCrop(size=images.shape[-1],scale=(self.settings.crop_ratio, 1.0))
		Flipper = transforms.RandomHorizontalFlip()
		cropped_images = []
		for image in images:
			cropped_images.append(Flipper(Cropper(image))) 
		cropped_images = torch.stack(cropped_images)
		return cropped_images

	def init_soft_stat(self):
		"""
		init a soft_mean and a soft_var dictionary 
		to save teacher model batch normalization layer statistics
		NOTE: 
		this operation should be induced after GAN training up
		"""
		self.soft_mean = {}
		self.soft_var = {}
		self.BN_layer_num = 0
		for m in self.model_teacher.modules():
			if isinstance(m, nn.BatchNorm2d):
				self.BN_layer_num += 1
		# set the start layer of BNS statistics calculation 
		self.start_layer = int((self.BN_layer_num + 1) / 2) - 2
		#self.start_layer = 15
		print(f"soft labels start layer: {self.start_layer}")

		for c in range(self.settings.nClasses):
			labels = Variable((torch.ones(self.settings.soft_stat_batchsize) * c).long()).cuda()
			labels = labels.contiguous()
			z = Variable(torch.randn(self.settings.soft_stat_batchsize, self.settings.latent_dim)).cuda()
			z = z.contiguous()
			images = self.generator(z, labels) # out of memory
			self.mean_list.clear()
			self.var_list.clear()
			output_teacher_batch = self.model_teacher(images) # [batchsize, number of classes]
			
			self.soft_mean[c] = []
			self.soft_var[c] = []
			for layer_index in range(self.BN_layer_num):
				self.soft_mean[c].append(self.teacher_running_mean[layer_index])
				self.soft_var[c].append(self.teacher_running_var[layer_index])
			self.teacher_running_var.clear()
			self.teacher_running_mean.clear()
			
	def update_soft_stat(self,):
		"""
		Apply EMA on soft BNS update
		"""
		for c in range(self.settings.nClasses):
			labels = Variable((torch.ones(self.settings.soft_stat_batchsize) * c).long()).cuda()
			labels = labels.contiguous()
			z = Variable(torch.randn(self.settings.soft_stat_batchsize, self.settings.latent_dim)).cuda()
			z = z.contiguous()
			images = self.generator(z, labels)
			images = self.augmentation(images)
			self.mean_list.clear()
			self.var_list.clear()

			output_teacher_batch = self.model_teacher(images) # [batchsize, number of classes]
			_, index = torch.max(output_teacher_batch, dim=1) 
			assert index.size() == labels.size()
			c = int(labels)
			idx = int(index)
			if (idx == c):
				for layer_index in range(self.BN_layer_num):
					self.soft_mean[c][layer_index] = (1.0 - self.settings.soft_decay) * self.soft_mean[c][layer_index] + \
													self.settings.soft_decay * (self.teacher_running_mean[layer_index])
					self.soft_var[c][layer_index] = (1.0 - self.settings.soft_decay) * self.soft_var[c][layer_index] + \
													self.settings.soft_decay * (self.teacher_running_var[layer_index])

	def compute_class_BNS_loss(self, labels, output_teacher_batch=None):
		"""
		compute class-wise BNS loss
		"""
		D_BNS_loss = torch.zeros(1).cuda()
		C_BNS_loss = torch.zeros(1).cuda()
		_, index = torch.max(output_teacher_batch, dim=1) 
		assert index.size() == labels.size()

		for i in range(len(labels)):
			"""
			the output teacher batch should be used to distinguish
			"""
			c = int(labels[i])
			idx = int(index[i])
			for layer in range(self.start_layer, len(self.mean_list)):
				# compute the distance between centroids and the running BNS
				C_BNS_loss += self.MSE_loss(self.mean_list[layer], self.soft_mean[c][layer].cuda()) \
							+ self.MSE_loss(self.var_list[layer], self.soft_var[c][layer].cuda())
				C_BNS_loss = C_BNS_loss / (len(self.mean_list) - self.start_layer)
				'''
				if (idx == c):
					for layer in range(self.start_layer, len(self.mean_list)):
						# compute the distance between centroids and the running BNS
						C_BNS_loss += self.MSE_loss(self.mean_list[layer], self.soft_mean[c][layer].cuda()) \
									+ self.MSE_loss(self.var_list[layer], self.soft_var[c][layer].cuda())
						C_BNS_loss = C_BNS_loss / (len(self.mean_list) - self.start_layer)
				'''
				D_BNS_loss += self.MSE_loss(self.mean_list[layer], self.mean_list[layer] + torch.FloatTensor(self.mean_list[layer].shape).uniform_(-0.03, 0.03).cuda()) \
							+ self.MSE_loss(self.var_list[layer], self.var_list[layer] + torch.FloatTensor(self.var_list[layer].shape).uniform_(-0.05, 0.05).cuda())
		return C_BNS_loss + 0.1 * D_BNS_loss

	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_S = self.lr_master_S.get_lr(epoch)
		lr_G = self.lr_master_G.get_lr(epoch)
		# update learning rate of model optimizer
		for param_group in self.optimizer_S.param_groups:
			param_group['lr'] = lr_S

		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr_G
	
	def loss_fn_kd(self, output, labels, teacher_outputs):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha

		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""

		criterion_d = nn.CrossEntropyLoss().cuda()
		kdloss = nn.KLDivLoss(reduction='batchmean').cuda()

		alpha = self.settings.alpha
		T = self.settings.temperature
		a = F.log_softmax(output / T, dim=1)
		b = F.softmax(teacher_outputs / T, dim=1)
		c = (alpha * T * T)
		d = criterion_d(output, labels)

		KD_loss = kdloss(a,b)*c + d
		return KD_loss
	
	def forward(self, images, teacher_outputs, labels=None):
		"""
		forward propagation
		"""
		# forward and backward and optimize
		output = self.model(images)
		if labels is not None:
			loss = self.loss_fn_kd(output, labels, teacher_outputs)
			return output, loss
		else:
			return output, None
	
	def backward_G(self, loss_G):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		loss_G.backward()
		self.optimizer_G.step()

	def backward_S(self, loss_S):
		"""
		backward propagation
		"""
		self.optimizer_S.zero_grad()
		loss_S.backward()
		self.optimizer_S.step()

	def backward(self, loss):
		# unused in this method
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		self.optimizer_S.zero_grad()
		loss.backward()
		self.optimizer_G.step()
		self.optimizer_S.step()

	def hook_fn_forward(self, module, input, output):
		input = input[0]
		mean = input.mean([0, 2, 3])
		# use biased var in train
		var = input.var([0, 2, 3], unbiased=False)

		self.mean_list.append(mean)
		self.var_list.append(var)
		self.teacher_running_mean.append(module.running_mean) # number_of_layers x channels
		self.teacher_running_var.append(module.running_var)   # number_of_layers x channels

	def hook_fn_forward_saveBN(self,module, input, output):
		self.save_BN_mean.append(module.running_mean.cpu())
		self.save_BN_var.append(module.running_var.cpu())
	
	def train(self, epoch):
		"""
		training
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		fp_acc = utils.AverageMeter()

		iters = 200
		self.epoch = epoch
		self.update_lr(self.epoch)

		self.model.eval()
		self.model_teacher.eval()
		self.generator.train()
		
		start_time = time.time()
		end_time = start_time
		
		if self.epoch==0:
			for m in self.model_teacher.modules():
				if isinstance(m, nn.BatchNorm2d):
					m.register_forward_hook(self.hook_fn_forward)
		
		if (self.epoch==self.settings.soft_stat_epoch): 
			print('\ninitialization of soft BNS\n')
			self.init_soft_stat()

		if (self.epoch > self.settings.soft_stat_epoch):
			'''
			update soft BNS 
			'''
			self.update_soft_stat()
		
		for i in range(iters):
			start_time = time.time()
			data_time = start_time - end_time

			# Get labels ranging from 0 to n_classes for n rows
			labels = Variable(torch.randint(0, self.settings.nClasses, (self.settings.batchSize,))).cuda()
			labels = labels.contiguous()# [batchsize]
			# Gaussian noise (latent_dim [200, 100])
			z = Variable(torch.randn(self.settings.batchSize, self.settings.latent_dim)).cuda()
			z = z.contiguous()
			# CGAN for images
			images = self.generator(z, labels)
			#f self.epoch > self.settings.soft_stat_epoch:
			images = self.augmentation(images)

			self.mean_list.clear()
			self.var_list.clear()
			output_teacher_batch = self.model_teacher(images) # [batchsize, number of classes]

			# One hot loss
			loss_one_hot = self.criterion(output_teacher_batch, labels)
			# BNS_loss
			BNS_loss = self.compute_BNS_loss()

			# loss of Generator
			if (self.epoch > self.settings.soft_stat_epoch):
				'''
				update the soft stat and compute class-BNS loss
				'''
				C_BNS_loss = self.compute_class_BNS_loss(labels, output_teacher_batch)
				loss_G = loss_one_hot + 0.1 * BNS_loss + self.settings.C_BNSLoss_weight * C_BNS_loss
			else:
				loss_G = loss_one_hot + 0.1 * BNS_loss
			self.backward_G(loss_G)

			# loss of KL Divergence 
			output, loss_S = self.forward(images.detach(), output_teacher_batch.detach(), labels)
			if self.epoch >= self.settings.warmup_epochs:
				self.backward_S(loss_S)

			single_error, single_loss, single5_error = utils.compute_singlecrop(
				outputs=output, labels=labels,
				loss=loss_S, top5_flag=True, mean_flag=True)
			
			top1_error.update(single_error, images.size(0))
			top1_loss.update(single_loss, images.size(0))
			top5_error.update(single5_error, images.size(0))
			
			end_time = time.time()
			
			gt = labels.data.cpu().numpy()
			d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)

			fp_acc.update(d_acc)

		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f] [S loss: %f] "
			% (self.epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(), BNS_loss.item(),
			 loss_S.item())
		)
		if (self.epoch > self.settings.soft_stat_epoch):
			print(f"	C BNS Loss: {C_BNS_loss.item()}")

		self.scalar_info['accuracy every epoch'] = 100 * d_acc
		self.scalar_info['G loss every epoch'] = loss_G
		self.scalar_info['One-hot loss every epoch'] = loss_one_hot
		self.scalar_info['S loss every epoch'] = loss_S

		self.scalar_info['training_top1error'] = top1_error.avg
		self.scalar_info['training_top5error'] = top5_error.avg
		self.scalar_info['training_loss'] = top1_loss.avg
		
		if self.tensorboard_logger is not None:
			for tag, value in list(self.scalar_info.items()):
				self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
			self.scalar_info = {}

		return top1_error.avg, top1_loss.avg, top5_error.avg
	
	def test(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.cuda()
				images = images.cuda()
				output = self.model(images)

				loss = torch.ones(1)
				self.mean_list.clear()
				self.var_list.clear()

				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
		
		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
		)
		
		self.scalar_info['testing_top1error'] = top1_error.avg
		self.scalar_info['testing_top5error'] = top5_error.avg
		self.scalar_info['testing_loss'] = top1_loss.avg
		if self.tensorboard_logger is not None:
			for tag, value in self.scalar_info.items():
				self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
			self.scalar_info = {}
		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg

	def test_teacher(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				data_time = start_time - end_time

				labels = labels.cuda()
				if self.settings.tenCrop:
					image_size = images.size()
					images = images.view(
						image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
					images_tuple = images.split(image_size[0])
					output = None
					for img in images_tuple:
						if self.settings.nGPU == 1:
							img = img.cuda()
						img_var = Variable(img, volatile=True)
						temp_output, _ = self.forward(img_var)
						if output is None:
							output = temp_output.data
						else:
							output = torch.cat((output, temp_output.data))
					single_error, single_loss, single5_error = utils.compute_tencrop(
						outputs=output, labels=labels)
				else:
					if self.settings.nGPU == 1:
						images = images.cuda()

					output = self.model_teacher(images)

					loss = torch.ones(1)
					self.mean_list.clear()
					self.var_list.clear()

					single_error, single_loss, single5_error = utils.compute_singlecrop(
						outputs=output, loss=loss,
						labels=labels, top5_flag=True, mean_flag=True)
				#
				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))

				end_time = time.time()
				iter_time = end_time - start_time

		print(
				"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
		)

		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg
