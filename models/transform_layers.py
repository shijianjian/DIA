import math
import numbers
import numpy as np
import cv2

import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, Beta
from torch.autograd import Function

if torch.__version__ >= '1.4.0':
    kwargs = {'align_corners': False}
else:
    kwargs = {}


def rgb2hsv(rgb):
    """Convert a 4-d RGB tensor to the HSV counterpart.

    Here, we compute hue using atan2() based on the definition in [1],
    instead of using the common lookup table approach as in [2, 3].
    Those values agree when the angle is a multiple of 30°,
    otherwise they may differ at most ~1.2°.

    References
    [1] https://en.wikipedia.org/wiki/Hue
    [2] https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    [3] https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L212
    """

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin

    hue = torch.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)
    hue = (hue % (2 * math.pi)) / (2 * math.pi)
    saturate = delta / Cmax
    value = Cmax
    hsv = torch.stack([hue, saturate, value], dim=1)
    hsv[~torch.isfinite(hsv)] = 0.
    return hsv


def hsv2rgb(hsv):
    """Convert a 4-d HSV tensor to the RGB counterpart.

    >>> %timeit hsv2rgb(hsv)
    2.37 ms ± 13.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> %timeit rgb2hsv_fast(rgb)
    298 µs ± 542 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> torch.allclose(hsv2rgb(hsv), hsv2rgb_fast(hsv), atol=1e-6)
    True

    References
    [1] https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    """
    h, s, v = hsv[:, [0]], hsv[:, [1]], hsv[:, [2]]
    c = v * s

    n = hsv.new_tensor([5, 3, 1]).view(3, 1, 1)
    k = (n + h * 6) % 6
    t = torch.min(k, 4 - k)
    t = torch.clamp(t, 0, 1)

    return v - c * t


class RandomResizedCropLayer(nn.Module):
    def __init__(self, size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        '''
            Inception Crop
            size (tuple): size of fowarding image (C, W, H)
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        '''
        super(RandomResizedCropLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.size = size
        self.register_buffer('_eye', _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, inputs, whbias=None):
        _device = inputs.device
        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        if whbias is None:
            whbias = self._sample_latent(inputs)

        _theta[:, 0, 0] = whbias[:, 0]
        _theta[:, 1, 1] = whbias[:, 1]
        _theta[:, 0, 2] = whbias[:, 2]
        _theta[:, 1, 2] = whbias[:, 3]

        grid = F.affine_grid(_theta, inputs.size(), **kwargs).to(_device)
        output = F.grid_sample(inputs, grid, padding_mode='reflection', **kwargs)

        if self.size is not None:
            output = F.adaptive_avg_pool2d(output, self.size[:2])

        return output

    def _clamp(self, whbias):

        w = whbias[:, 0]
        h = whbias[:, 1]
        w_bias = whbias[:, 2]
        h_bias = whbias[:, 3]

        # Clamp with scale
        w = torch.clamp(w, *self.scale)
        h = torch.clamp(h, *self.scale)

        # Clamp with ratio
        w = self.ratio[0] * h + torch.relu(w - self.ratio[0] * h)
        w = self.ratio[1] * h - torch.relu(self.ratio[1] * h - w)

        # Clamp with bias range: w_bias \in (w - 1, 1 - w), h_bias \in (h - 1, 1 - h)
        w_bias = w - 1 + torch.relu(w_bias - w + 1)
        w_bias = 1 - w - torch.relu(1 - w - w_bias)

        h_bias = h - 1 + torch.relu(h_bias - h + 1)
        h_bias = 1 - h - torch.relu(1 - h - h_bias)

        whbias = torch.stack([w, h, w_bias, h_bias], dim=0).t()

        return whbias

    def _sample_latent(self, inputs):

        _device = inputs.device
        N, _, width, height = inputs.shape

        # N * 10 trial
        area = width * height
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))

        # If doesn't satisfy ratio condition, then do central crop
        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        cond_len = w.shape[0]
        if cond_len >= N:
            w = w[:N]
            h = h[:N]
        else:
            w = np.concatenate([w, np.ones(N - cond_len) * width])
            h = np.concatenate([h, np.ones(N - cond_len) * height])

        w_bias = np.random.randint(w - width, width - w + 1) / width
        h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height

        whbias = np.column_stack([w, h, w_bias, h_bias])
        whbias = torch.tensor(whbias, device=_device)

        return whbias


class HorizontalFlipRandomCrop(nn.Module):
    def __init__(self, max_range):
        super(HorizontalFlipRandomCrop, self).__init__()
        self.max_range = max_range
        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, input, sign=None, bias=None, rotation=None):
        _device = input.device
        N = input.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        if sign is None:
            sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        if bias is None:
            bias = torch.empty((N, 2), device=_device).uniform_(-self.max_range, self.max_range)
        _theta[:, 0, 0] = sign
        _theta[:, :, 2] = bias

        if rotation is not None:
            _theta[:, 0:2, 0:2] = rotation

        grid = F.affine_grid(_theta, input.size(), **kwargs).to(_device)
        output = F.grid_sample(input, grid, padding_mode='reflection', **kwargs)

        return output

    def _sample_latent(self, N, device=None):
        sign = torch.bernoulli(torch.ones(N, device=device) * 0.5) * 2 - 1
        bias = torch.empty((N, 2), device=device).uniform_(-self.max_range, self.max_range)
        return sign, bias


class Rotation(nn.Module):
    def __init__(self, max_range = 4):
        super(Rotation, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None, **kwargs):
        _device = input.device

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = torch.rot90(input, aug_index, (2, 3))

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1-_mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = torch.rot90(input, aug_index, (2, 3))

        return output


class Diffusion(nn.Module):
    def __init__(
        self, model=None, model_path=None, max_range = 6,
        step_scale=200, pre_process=nn.Identity(), post_process=nn.Identity()
    ):
        super(Diffusion, self).__init__()
        if model is None and model_path is not None:
            from denoising_diffusion_pytorch import GaussianDiffusion, Unet
            _model = Unet(
                dim = 64,
                dim_mults = (1, 2, 4, 8)
            )
            self.model = GaussianDiffusion(
                _model,
                image_size = 32,
                timesteps = 1000,
                sampling_timesteps = 250,
                loss_type = 'l1'
            )

            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict["model"])
        elif model is not None:
            self.model = model
        else:
            raise ValueError

        self._step_sampler = Uniform(0, 1)
        self.prob = 0.5
        self.max_range = max_range
        self.step_scale = step_scale
        self.pre_process = pre_process
        self.post_process = post_process

    def forward(
        self, input, aug_index=None, diffusion=True, diffusion_offset=0., diffusion_upper_offset=1.
    ):
        _device = input.device

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(self.max_range)
            # step_no = aug_index * (self.step_scale + diffusion_offset)

            if diffusion:
                sample_no = self._step_sampler.sample() * (diffusion_upper_offset - diffusion_offset) + diffusion_offset
                step_no = (sample_no * self.step_scale).int().item()
                input = self.pre_process(input)
                input = self.model.p_sample(input, step_no)[1]  # output: (pred_img, x_start)
                input = self.post_process(input)

            output = torch.rot90(input, aug_index, (2, 3))

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output
        else:
            # step_no = aug_index * (self.step_scale + diffusion_offset)
            # step_no = (self._step_sampler.sample() * aug_index * self.step_scale).int().item()
            if diffusion:
                step_no = ((self._step_sampler.sample() + diffusion_offset) * self.step_scale).int().item()
                input = self.pre_process(input)
                input = self.model.p_sample(input, step_no)[1]  # output: (pred_img, x_start)
                input = self.post_process(input)

            output = torch.rot90(input, aug_index, (2, 3))

        return output


class DiffusionCustomTrans(nn.Module):
    def __init__(
        self, model=None, model_path=None, max_range = 4,
        step_scale=100, pre_process=nn.Identity(), post_process=nn.Identity(),
        transform: str = "rotation",
    ):
        super(DiffusionCustomTrans, self).__init__()
        if model is None and model_path is not None:
            from denoising_diffusion_pytorch import GaussianDiffusion, Unet
            _model = Unet(
                dim = 64,
                dim_mults = (1, 2, 4, 8)
            )
            self.model = GaussianDiffusion(
                _model,
                image_size = 32,
                timesteps = 1000,
                sampling_timesteps = 250,
                loss_type = 'l1'
            )

            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict["model"])
        elif model is not None:
            self.model = model
        else:
            raise ValueError

        self._step_sampler = Uniform(0, 1)
        # self._step_sampler = Beta(2.5, 1)
        self.prob = 0.5
        self.max_range = max_range
        self.step_scale = step_scale
        self.pre_process = pre_process
        self.post_process = post_process
        self.tranform = self.get_transform(transform)

    def get_transform(self, transform):
        if transform == "rotation":
            return Rotation(self.max_range)
        elif transform == "cutperm":
            return CutPerm(self.max_range)
        raise NotImplementedError

    def forward(
        self, input, aug_index=None, diffusion=True, diffusion_offset=0., diffusion_upper_offset=1., random=True
    ):
        if diffusion:
            if random:
                sample_no = self._step_sampler.sample() * (diffusion_upper_offset - diffusion_offset) + diffusion_offset
                step_no = (sample_no * self.step_scale).int().item()
            else:
                step_no = diffusion_offset * self.step_scale
            input = self.pre_process(input)
            input = self.model.p_sample(input, step_no)[1]  # output: (pred_img, x_start)
            input = self.post_process(input)

        return self.tranform(input, aug_index=aug_index)


class CutPerm(nn.Module):
    def __init__(self, max_range = 4):
        super(CutPerm, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = self._cutperm(input, aug_index)

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = self._cutperm(input, aug_index)

        return output

    def _cutperm(self, inputs, aug_index):

        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)

        return inputs


class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(HorizontalFlipLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, inputs):
        _device = inputs.device

        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        _theta[:, 0, 0] = r_sign
        grid = F.affine_grid(_theta, inputs.size(), **kwargs).to(_device)
        inputs = F.grid_sample(inputs, grid, padding_mode='reflection', **kwargs)

        return inputs


class RandomColorGrayLayer(nn.Module):
    def __init__(self, p):
        super(RandomColorGrayLayer, self).__init__()
        self.prob = p

        _weight = torch.tensor([[0.299, 0.587, 0.114]])
        self.register_buffer('_weight', _weight.view(1, 3, 1, 1))

    def forward(self, inputs, aug_index=None):

        if aug_index == 0:
            return inputs

        l = F.conv2d(inputs, self._weight)
        gray = torch.cat([l, l, l], dim=1)

        if aug_index is None:
            _prob = inputs.new_full((inputs.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)

            gray = inputs * (1 - _mask) + gray * _mask

        return gray


class ColorJitterLayer(nn.Module):
    def __init__(self, p, brightness, contrast, saturation, hue):
        super(ColorJitterLayer, self).__init__()
        self.prob = p
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        if self.contrast:
            factor = x.new_empty(x.size(0), 1, 1, 1).uniform_(*self.contrast)
            means = torch.mean(x, dim=[2, 3], keepdim=True)
            x = (x - means) * factor + means
        return torch.clamp(x, 0, 1)

    def adjust_hsv(self, x):
        f_h = x.new_zeros(x.size(0), 1, 1)
        f_s = x.new_ones(x.size(0), 1, 1)
        f_v = x.new_ones(x.size(0), 1, 1)

        if self.hue:
            f_h.uniform_(*self.hue)
        if self.saturation:
            f_s = f_s.uniform_(*self.saturation)
        if self.brightness:
            f_v = f_v.uniform_(*self.brightness)

        return RandomHSVFunction.apply(x, f_h, f_s, f_v)

    def transform(self, inputs):
        # Shuffle transform
        if np.random.rand() > 0.5:
            transforms = [self.adjust_contrast, self.adjust_hsv]
        else:
            transforms = [self.adjust_hsv, self.adjust_contrast]

        for t in transforms:
            inputs = t(inputs)

        return inputs

    def forward(self, inputs):
        _prob = inputs.new_full((inputs.size(0),), self.prob)
        _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
        return inputs * (1 - _mask) + self.transform(inputs) * _mask


class RandomHSVFunction(Function):
    @staticmethod
    def forward(ctx, x, f_h, f_s, f_v):
        # ctx is a context object that can be used to stash information
        # for backward computation
        x = rgb2hsv(x)
        h = x[:, 0, :, :]
        h += (f_h * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        x[:, 1, :, :] = x[:, 1, :, :] * f_s
        x[:, 2, :, :] = x[:, 2, :, :] * f_v
        x = torch.clamp(x, 0, 1)
        x = hsv2rgb(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


class NormalizeLayer(nn.Module):
    """
    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, inputs):
        return (inputs - 0.5) / 0.5


class Blur(nn.Module):
    def __init__(self, size=3, gaussian=True, constant_size=32, resize_to_constant=False, method="cutperm", max_range = 4):
        super(Blur, self).__init__()
        self.size = size
        self.prob = 0.5
        self.constant_size = constant_size
        self.resize_to_constant = resize_to_constant
        self.gaussian = gaussian
        self.max_range = max_range
        if gaussian:
            self.kernel = self.get_gaussian_kernel(size)
        elif gaussian is None:
            pass
        else:
            self.kernel = K.filters.MedianBlur([size, size])
        
        if method == "cutperm":
            self.method = self._cutperm
        elif method == "rotate":
            self.method = self._rotate
        else:
            raise NotImplementedError

    def get_gaussian_kernel(self, size=3, sigma=0):
        # if sigma is non-positive, sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
        kernel = cv2.getGaussianKernel(size, sigma).dot(cv2.getGaussianKernel(size, sigma).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def get_sobel_kernel(self):
        kernel = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype='float32')
        kernel = kernel.reshape((1, 1, 3, 3))
        kernel = torch.nn.Parameter(data=torch.from_numpy(kernel), requires_grad=False)
        return kernel

    def _rotate(self, inputs: torch.Tensor, aug_index: int) -> torch.Tensor:
        return torch.rot90(inputs, aug_index, (2, 3))

    def _cutperm(self, inputs: torch.Tensor, aug_index: int) -> torch.Tensor:

        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)

        return inputs

    def forward(self, input: torch.Tensor, aug_index=None, **kwargs):
        _device = input.device
        _, _, H, W = input.size()

        if self.resize_to_constant:
            input = F.interpolate(input, (self.constant_size, self.constant_size))

        if aug_index is None:
            aug_index = np.random.randint(self.max_range)
            # step_no = aug_index * (self.step_scale + diffusion_offset)

            if self.gaussian:
                input = torch.cat([F.conv2d(
                    input[:, c:c + 1], self.kernel, padding=self.size // 2, stride=1
                ) for c in range(input.shape[1])], dim=1)
            elif self.gaussian is None:
                pass
            else:
                input = self.kernel(input)

            output = self.method(input, aug_index)

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output
        else:
            # step_no = aug_index * (self.step_scale + diffusion_offset)
            # step_no = (self._step_sampler.sample() * aug_index * self.step_scale).int().item()
            if self.gaussian:
                input = torch.cat([F.conv2d(
                    input[:, c:c + 1], self.kernel, padding=self.size // 2, stride=1
                ) for c in range(input.shape[1])], dim=1)
            elif self.gaussian is None:
                pass
            else:
                input = self.kernel(input)

            output = self.method(input, aug_index)

        if self.resize_to_constant:
            output = F.interpolate(output, (H, W))
        return output


class RandomizedQuantizationAugModule(nn.Module):
    def __init__(self, region_num, collapse_to_val = 'inside_random', spacing='random', transforms_like=False, p_random_apply_rand_quant = 1):
        """
        region_num: int;
        """
        super().__init__()
        self.region_num = region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.p_random_apply_rand_quant = p_random_apply_rand_quant

    def get_params(self, x):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.size() # one batch img
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0] # min, max over batch size, spatial dimension
        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int()
        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        EPSILON = 1
        if self.p_random_apply_rand_quant != 1:
            x_orig = x
        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c
            x = x.view(C, H, W)
        else:
            C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x) # -> (C), (C), (C)

        # region percentiles for each channel
        if self.spacing == "random":
            region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)
        elif self.spacing == "uniform":
            region_percentiles = torch.tile(torch.arange(1/(total_region_percentile_number_per_channel[0] + 1), 1, step=1/(total_region_percentile_number_per_channel[0]+1), device=x.device), [C])
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])
        # ordered region ends
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1, 1)
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]
        # ordered middle points
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2

        # associate region id
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.view(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)

        if self.collapse_to_val == 'middle':
            # middle points as the proxy for all values in corresponding regions
            proxy_vals = torch.gather(ordered_region_mid.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'inside_random':
            # random points inside each region as the proxy for all values in corresponding regions
            proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(), device=x.device)
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, self.region_num])
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (ordered_region_right_ends - ordered_region_left_ends)
            proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'all_zeros':
            proxy_vals = torch.zeros_like(x, device=x.device)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
            x = x.view(B, c, H, W)

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                x = torch.where(torch.rand([B,1,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                x = torch.where(torch.rand([C,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)

        return x
