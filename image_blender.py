import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from deep_blending_utils import compute_gt_gradient_alt, laplacian_filter_tensor, MeanShift, Vgg16, gram_matrix


class ImageBlender:

    def __init__(self, device):

        self.device = device

        self.mse = torch.nn.MSELoss()
        self.mean_shift = MeanShift(device)
        self.vgg = Vgg16().to(device)

    def blend_images(self, source_img: np.ndarray, target_image: np.ndarray, mask: np.ndarray,
                     num_blending_steps_first_pass: int = 1000, num_blending_steps_second_pass: int = 0,
                     grad_weight: float = 1e4, style_weight: float = 1e4, content_weight: float = 1,
                     tv_weight: float = 1e-6):

        source_image_tensor = \
            torch.tensor(source_img, dtype=torch.float32, device=self.device).unsqueeze(0).permute([0, 3, 1, 2])

        target_image_tensor = \
            torch.tensor(target_image, dtype=torch.float32, device=self.device).unsqueeze(0).permute([0, 3, 1, 2])

        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0).permute([0, 3, 1, 2])

        first_pass_result = self._run_first_pass(source_image_tensor, target_image_tensor, mask_tensor,
                                                 num_blending_steps_first_pass, grad_weight, style_weight,
                                                 content_weight, tv_weight)

        if num_blending_steps_second_pass > 0:
            second_pass_result = self._run_second_pass(first_pass_result, target_image_tensor, mask_tensor,
                                                       num_blending_steps_second_pass, style_weight, content_weight)
            return first_pass_result, second_pass_result

        return first_pass_result

    def _run_first_pass(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, mask,
                        num_steps: int, grad_weight: float, style_weight: float, content_weight: float,
                        tv_weight: float):

        input_img = torch.randn(target_image_tensor.shape).to(self.device)
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        gt_gradient = compute_gt_gradient_alt(source_image_tensor, target_image_tensor, mask, self.device)

        target_features_style = self.vgg(self.mean_shift(target_image_tensor))
        target_gram_style = [gram_matrix(y) for y in target_features_style]

        for _ in range(num_steps):
            def closure():
                blend_img = input_img * mask + target_image_tensor * (mask - 1) * (-1)

                grad_loss = self._compute_gradient_loss(blend_img, gt_gradient) * grad_weight

                style_loss = self._compute_style_loss(input_img, target_gram_style) * style_weight

                # Compute Content Loss
                source_object_features = self.vgg(self.mean_shift(source_image_tensor * mask))
                blend_object_features = self.vgg(self.mean_shift(blend_img * mask))
                content_loss = content_weight * self.mse(blend_object_features.relu2_2, source_object_features.relu2_2)

                # Compute TV Reg Loss
                tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                          torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
                tv_loss *= tv_weight

                # Compute Total Loss and Update Image
                loss = grad_loss + style_loss + content_loss + tv_loss
                optimizer.zero_grad()
                loss.backward()

                return loss

            optimizer.step(closure)

        input_img.data.clamp_(0, 255)
        blend_img_final = input_img * mask + target_image_tensor * (mask - 1) * (-1)

        return blend_img_final

    def _run_second_pass(self, first_pass_result: torch.Tensor, target_image_tensor: torch.Tensor, mask,
                         num_steps: int, style_weight: float, content_weight: float):

        x0 = torch.tensor(first_pass_result)
        optimizer = optim.LBFGS([x0.requires_grad_()])

        target_features_style = self.vgg(self.mean_shift(target_image_tensor))
        target_gram_style = [gram_matrix(y) for y in target_features_style]

        for _ in range(num_steps):
            def closure():
                style_loss = self._compute_style_loss(x0, target_gram_style) * style_weight

                # # Compute Content Loss
                # source_object_features = self.vgg(self.mean_shift(x0))
                # blend_object_features = self.vgg(self.mean_shift(blend_img * mask))
                # content_loss = content_weight * self.mse(blend_object_features.relu2_2, source_object_features.relu2_2)

                content_loss = 0

                # Compute Total Loss and Update Image
                loss = style_loss + content_loss
                optimizer.zero_grad()
                loss.backward()

                return loss

            optimizer.step(closure)

        x0.data.clamp_(0, 255)
        blend_img_final = x0 * mask + target_image_tensor * (mask - 1) * (-1)

        return blend_img_final

    def _compute_gradient_loss(self, img, gt_gradients):
        # Compute Laplacian Gradient of Blended Image
        pred_gradient = laplacian_filter_tensor(img, self.device)

        # Compute Gradient Loss
        grad_loss = 0
        for pred_grad, gt_grad in zip(pred_gradient, gt_gradients):
            grad_loss += self.mse(pred_grad, gt_grad)
        grad_loss = (grad_loss / len(pred_gradient))

        return grad_loss

    def _compute_style_loss(self, img, gt_style_gram):
        # Compute Style Loss
        style_features = self.vgg(self.mean_shift(img))
        style_features_gram = [gram_matrix(y) for y in style_features]

        style_loss = 0
        for layer_features_gram, target_layer_features_gram in zip(style_features_gram, gt_style_gram):
            style_loss += self.mse(layer_features_gram, target_layer_features_gram)
        style_loss = (style_loss / len(style_features_gram))

        return style_loss
