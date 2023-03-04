import torch.nn as nn
import torch
from conditional_batchnorm import CategoricalConditionalBatchNorm2d

from options import Option

# original c-gan
class Generator(nn.Module):
	def __init__(self, options=None, conf_path=None):
		super(Generator, self).__init__()
		self.settings = options or Option(conf_path)
		self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0 = nn.Sequential(
			nn.BatchNorm2d(128),
		)

		self.conv_blocks1 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.conv_blocks2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(self.settings.channels, affine=False)
		)

	def forward(self, z, labels):
		gen_input = torch.mul(self.label_emb(labels), z)
		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0(out)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2(img)
		return img

class Generator_imagenet(nn.Module):
	def __init__(self, options=None, conf_path=None):
		self.settings = options or Option(conf_path)

		super(Generator_imagenet, self).__init__()

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img

class Qimera_Generator(nn.Module):
	def __init__(self, options=None, conf_path=None, teacher_weight=None, freeze=True):
		super(Qimera_Generator, self).__init__()
		self.settings = options or Option(conf_path)

		self.init_size = self.settings.img_size // 4
		self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.intermediate_dim)
		self.embed_normalizer = nn.BatchNorm1d(self.label_emb.weight.T.shape,affine=False,track_running_stats=False)
		
		#self.fc_reducer = nn.Linear(10, 64)
		self.l1 = nn.Sequential(nn.Linear(64, 8192))

		'''
		self.fc_reducer = nn.Linear(in_features=self.label_emb.weight.shape[-1], out_features=self.settings.intermediate_dim) # 10 * 64
		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.intermediate_dim, 128 * self.init_size ** 2))

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))
		'''

		self.conv_blocks0 = nn.Sequential(
			nn.BatchNorm2d(128),
		)

		self.conv_blocks1 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.conv_blocks2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(self.settings.channels, affine=False)
		)

	def forward(self, z, labels, linear=None, z2=None):
		gen_input = self.embed_normalizer(torch.add(self.label_emb(labels),self.settings.noise_scale*z).T).T 
		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0(out)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2(img)
		return img

if __name__ == '__main__':
	conf_path = "cifar100_resnet20.hocon"
	option = Option(conf_path)
    
	z = torch.zeros((200, 100)).contiguous().cuda()
	labels = torch.autograd.Variable(torch.randint(0, 100, (200,))).contiguous().cuda()
	g = Generator(options=option).cuda()
    #x = torch.zeros((1, 3, 32, 32))
    #d = Discriminator()

	g(z, labels)