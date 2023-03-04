import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

# option file should be modified according to your expriment
from options import Option

from dataloader import DataLoader
from trainer import Trainer

import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from gan import Generator, Generator_imagenet

class ExperimentDesign:
	def __init__(self, generator=None, options=None, conf_path=None):
		self.settings = options or Option(conf_path)
		self.generator = generator
		self.train_loader = None
		self.test_loader = None
		self.model = None
		self.model_teacher = None
		
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0
		self.end_epoch = self.settings.nEpochs
		self.test_input = None

		self.unfreeze_Flag = True
		
		os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
		os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices
		
		self.settings.set_save_path()
		self.logger = self.set_logger()
		self.settings.paramscheck(self.logger)

		self.prepare()
	
	def set_logger(self):
		logger = logging.getLogger('baseline')
		file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
		console_formatter = logging.Formatter('%(message)s')
		# file log
		file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
		file_handler.setFormatter(file_formatter)
		
		# console log
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(console_formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		
		logger.setLevel(logging.INFO)
		return logger

	def prepare(self):
		self._set_gpu()
		self._set_dataloader()
		self._set_model()
		self._replace()
		self.logger.info(self.model)
		self._set_trainer()

		self._writer = SummaryWriter(os.path.join(self.settings.save_path, "writerlog"))
	
	def _set_gpu(self):
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
		cudnn.benchmark = True

	def _set_dataloader(self):
		# create data loader
		data_loader = DataLoader(dataset=self.settings.dataset,
		                         batch_size=self.settings.batchSize,
		                         data_path=self.settings.dataPath,
		                         n_threads=self.settings.nThreads,
		                         ten_crop=self.settings.tenCrop,
		                         logger=self.logger)
		
		self.train_loader, self.test_loader = data_loader.getloader()

	def _set_model(self):
		if self.settings.dataset in ["cifar10"]:
			self.test_input = Variable(torch.randn(1, 3, 32, 32).cuda())
			self.model = ptcv_get_model('resnet20_cifar10', pretrained=False)
			self.model_teacher = ptcv_get_model('resnet20_cifar10', pretrained=False)
			self.model.load_state_dict(torch.load("Ori_models/resnet20_cifar10.pth"))
			self.model_teacher.load_state_dict(torch.load("Ori_models/resnet20_cifar10.pth"))
			self.model_teacher.eval()
		elif self.settings.dataset in ["cifar100"]:
			self.test_input = Variable(torch.randn(1, 3, 32, 32).cuda())
			self.model = ptcv_get_model('resnet20_cifar100', pretrained=False)
			self.model_teacher = ptcv_get_model('resnet20_cifar100', pretrained=False)
			self.model.load_state_dict(torch.load("Ori_models/resnet20_cifar100.pth"))
			self.model_teacher.load_state_dict(torch.load("Ori_models/resnet20_cifar100.pth"))
			self.model_teacher.eval()

		elif self.settings.dataset in ["imagenet"]:
			self.test_input = Variable(torch.randn(1, 3, 224, 224).cuda())
			self.logger.info(f"Using original pretrained model: {self.settings.model}")
			self.model = ptcv_get_model(self.settings.model, pretrained=False)
			self.model_teacher = ptcv_get_model(self.settings.model, pretrained=False)
			self.model.load_state_dict(torch.load(f"Ori_models/{self.settings.model}.pth"))
			self.model_teacher.load_state_dict(torch.load(f"Ori_models/{self.settings.model}.pth"))
			self.model_teacher.eval()

		else:
			assert False, "unsupport data set: " + self.settings.dataset

	def _set_trainer(self):
		# set lr master
		lr_master_S = utils.LRPolicy(self.settings.lr_S,
		                           self.settings.nEpochs,
		                           self.settings.lrPolicy_S)
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_G)

		params_dict_S = {
			'step': self.settings.step_S,
			'decay_rate': self.settings.decayRate_S
		}

		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}
		
		lr_master_S.set_params(params_dict=params_dict_S)
		lr_master_G.set_params(params_dict=params_dict_G)

		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			generator = self.generator,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			lr_master_S=lr_master_S,
			lr_master_G=lr_master_G,
			settings=self.settings,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			run_count=self.start_epoch)

	def quantize_model(self,model):
		"""
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
		
		weight_bit = self.settings.qw
		act_bit = self.settings.qa
		
		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		
		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
		
		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model(m))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model(mod))
			return q_model
	
	def _replace(self):
		self.model = self.quantize_model(self.model)
	
	def freeze_model(self,model):
		"""
		freeze the activation range
		"""
		if type(model) == QuantAct:
			model.fix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.freeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.freeze_model(mod)
			return model
	
	def unfreeze_model(self,model):
		"""
		unfreeze the activation range
		"""
		if type(model) == QuantAct:
			model.unfix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.unfreeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.unfreeze_model(mod)
			return model

	def run(self):
		best_top1 = 100
		best_top5 = 100
		start_time = time.time()

		quantized_model_path = self.settings.save_path + "{}_{}_quantized.pth".format(self.settings.dataset, self.settings.model)
		generator_path = self.settings.save_path + "{}_{}_generator.pth".format(self.settings.dataset, self.settings.model)
		self.logger.info("quantized model path: {}\ngenerator path: {}".format(quantized_model_path, generator_path))

		#print("Testing teacher model...")
		#test_error, test_loss, test5_error = self.trainer.test_teacher(0)
		#self.logger.info("#==>Orignal Result is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(100-test_error, 100-test5_error))

		if (self.settings.load_quantized):
			self.model.load_state_dict(torch.load(quantized_model_path))
			self.generator.load_state_dict(torch.load(generator_path))
			self.start_epoch = self.settings.nEpochs
			self.end_epoch = self.settings.nEpochs * 2
			self.settings.soft_stat_epoch = self.settings.nEpochs

		try:
			for epoch in range(self.start_epoch, self.end_epoch):
				self.epoch = epoch

				if epoch < 4:
					print ("\nself.unfreeze_model(self.model)\n")
					self.unfreeze_model(self.model)

				train_error, train_loss, train5_error = self.trainer.train(epoch=epoch)

				self.freeze_model(self.model)

				if self.settings.dataset in ["cifar100", "cifar10"]:
					test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
				elif self.settings.dataset in ["imagenet"]:
					if epoch > self.settings.warmup_epochs - 2:
						test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
					else:
						test_error = 100
						test5_error = 100
				else:
					assert False, "invalid data set"

				self._writer.add_scalar("Test Top1 Error", test_error, epoch)
				self._writer.add_scalar("Test Top5 Error", test5_error, epoch)

				if best_top1 >= test_error:
					best_top1 = test_error
					best_top5 = test5_error
					# update the loaded model
					torch.save(self.model.state_dict(), quantized_model_path)
					torch.save(self.generator.state_dict(), generator_path)
				
				self.logger.info("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(best_top1, best_top5))
				self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1, 100 - best_top5))

		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		self._writer.close()

		return best_top1, best_top5


def main():
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
	                    help='input the path of config file')
	parser.add_argument('--id', type=int, metavar='experiment_id',
	                    help='Experiment ID')
	args = parser.parse_args()
	option = Option(args.conf_path)
	option.manualSeed = args.id + 1
	option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)

	if option.dataset in ["cifar100", "cifar10"]:
		generator = Generator(option)
	elif option.dataset in ["imagenet"]:
		generator = Generator_imagenet(option)
	else:
		assert False, "invalid data set"
	experiment = ExperimentDesign(generator, option)
	experiment.run()


if __name__ == '__main__':
	main()
