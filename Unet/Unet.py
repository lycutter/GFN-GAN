import torch
import torch.nn as nn
import functools
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.net = UnetGenerator()

    def forward(self, x):
        return self.net(x)

class UnetGenerator(nn.Module):
	def __init__(self, input_nc=3, output_nc=3, num_downs=7, ngf=64,
				 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], use_parallel = True, learn_residual = False):
		super(UnetGenerator, self).__init__()
		self.gpu_ids = gpu_ids
		self.use_parallel = use_parallel
		self.learn_residual = learn_residual
		# currently support only input_nc == output_nc
		assert(input_nc == output_nc)

		# construct unet structure
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
		for i in range(num_downs - 5):
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

		self.model = unet_block

	def forward(self, input):
		if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
			output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			output = self.model(input)
		if self.learn_residual:
			output = input + output
			output = torch.clamp(output,min = -1,max = 1)
		return output


class UnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc,
				 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost

		# self.conv_out_filters = ['3-64', '64-128', '128-256', '256-512', '512-512-1', '512-512-2', '512-512-3']
		# self.upconv_out_filters = ['512-512', '1024-512-1', '1024-512-2', '1024-256', '512-128', '256-64', '128-3']
		# self.conv_layer_count = 0
		# self.deconv_layer_count = 0

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
							 stride=2, padding=1, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)

		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]

			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):

		# if self.conv_layer_count < len(self.conv_out_filters):
		# 	print(self.conv_out_filters[self.conv_layer_count])
		# 	self.conv_layer_count += 1
		# if self.conv_layer_count >= len(self.conv_out_filters):
		# 	print(self.upconv_out_filters[self.deconv_layer_count])
		# 	self.deconv_layer_count += 1


		if self.outermost:
			return self.model(x)
		else:
			out = self.model(x)
			out = torch.cat([out, x], 1)
			return out