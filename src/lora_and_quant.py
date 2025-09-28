import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.functional import quantize_nf4, dequantize_nf4

import logging

log = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    A standard Low-Rank Adaptation (LoRA) layer.

    This layer implements the LoRA technique by creating two smaller matrices (A and B)
    whose product is added to the original weight matrix's output. This allows for
    efficient fine-tuning of large models.

    Args:
        in_features (int): Number of input features to the original linear layer.
        out_features (int): Number of output features from the original linear layer.
        rank (int, optional): The rank of the LoRA decomposition. Defaults to 4.
        alpha (int, optional): The scaling factor for the LoRA output. Defaults to 1.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        Forward pass for the LoRA layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The LoRA-adjusted output tensor.
        """
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class FusedLoRALayer(nn.Module):
    """
    A LoRA layer designed for fused linear layers.

    Some models fuse multiple linear layers (e.g., query, key, and value in self-attention)
    into a single larger layer for efficiency. This class applies LoRA to such fused layers
    by partitioning the LoRA matrices accordingly.

    Args:
        in_features (int): Number of input features to the original fused linear layer.
        fused_dim_list (list[int]): A list of the output dimensions of the original,
                                    un-fused layers. The sum of this list should equal
                                    the `out_features` of the fused layer.
        rank (int, optional): The rank of the LoRA decomposition. Defaults to 4.
        alpha (int, optional): The scaling factor for the LoRA output. Defaults to 1.
    """
    def __init__(self, in_features, fused_dim_list, rank=4, alpha=1):
        super().__init__()
        self.fused_dim_list = fused_dim_list
        self.lora_As = nn.Parameter(
            torch.zeros(rank * len(fused_dim_list), in_features)
        )
        self.lora_Bs = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, rank)) for dim in fused_dim_list]
        )
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_As, a=math.sqrt(5))

    def forward(self, x):
        """
        Forward pass for the FusedLoRALayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The LoRA-adjusted output tensor for the fused layer.
        """
        As = F.linear(x, self.lora_As).chunk(len(self.fused_dim_list), dim=-1)
        Bs = []
        for i, lora_b in enumerate(self.lora_Bs):
            Bs.append(F.linear(As[i], lora_b))

        return torch.cat(Bs, dim=-1) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    A wrapper module that replaces a standard `nn.Linear` layer with a version
    that includes LoRA. The original linear layer's weights are frozen.

    Args:
        linear (nn.Linear): The original linear layer to be wrapped.
        fused_dim_list (list[int], optional): If the linear layer is a fused layer,
                                              this list specifies the dimensions of the
                                              original layers. Defaults to None.
        rank (int, optional): The rank for the LoRA decomposition. Defaults to 4.
        alpha (int, optional): The scaling factor for LoRA. Defaults to 1.
    """
    def __init__(self, linear, fused_dim_list=None, rank=4, alpha=1):
        super().__init__()

        self.linear = linear
        self.linear.requires_grad_(False)
        if fused_dim_list:
            self.fused_dim_list = fused_dim_list
            self.is_fused_linear = True
            assert (
                sum(fused_dim_list) == linear.out_features
            ), f"sum of fused_dim_list {sum(fused_dim_list)} is not equal to linear.out_features {linear.out_features}"
            self.lora = FusedLoRALayer(linear.in_features, fused_dim_list, rank, alpha)
        else:
            self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.lora.to(device=linear.weight.device)
        self.lora_enabled = True  # toggle flag

    def enable_lora(self, mode=True):
        """Enable or disable the LoRA adjustment."""
        self.lora_enabled = mode

    def disable_lora(self):
        """Disable the LoRA adjustment."""
        self.lora_enabled = False
    
    def forward(self, x):
        """
        Forward pass. If LoRA is enabled, its output is added to the original linear layer's output.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.lora_enabled:
            return self.linear(x) + self.lora(x)
        else:
            return self.linear(x)

class Quantized8BitLinearWithLoRA(nn.Module):
    """
    A wrapper module that replaces a `nn.Linear` layer with a version that has
    an 8-bit quantized weight and an attached LoRA layer.

    Args:
        linear (nn.Linear): The original linear layer.
        fused_dim_list (list[int], optional): For fused layers, the dimensions of original layers. Defaults to None.
        rank (int, optional): The rank for LoRA. Defaults to 4.
        alpha (int, optional): The scaling factor for LoRA. Defaults to 1.
        quant (torch.dtype, optional): The 8-bit floating-point format to use for quantization.
                                       Defaults to `torch.float8_e4m3fn`.
    """
    def __init__(
        self, linear, fused_dim_list=None, rank=4, alpha=1, quant=torch.float8_e4m3fn
    ):
        super().__init__()
        assert quant in {
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        }, "Unknown quantization"

        self.linear_weight = linear.weight.to(dtype=quant)
        self.linear_weight.detach()

        if linear.bias is not None:
            self.linear_bias = linear.bias.to(dtype=quant)
            self.linear_bias.detach()
        else:
            self.linear_bias = None

        if fused_dim_list:
            self.fused_dim_list = fused_dim_list
            self.is_fused_linear = True
            assert (
                sum(fused_dim_list) == linear.out_features
            ), f"sum of fused_dim_list {sum(fused_dim_list)} is not equal to linear.out_features {linear.out_features}"
            self.lora = FusedLoRALayer(linear.in_features, fused_dim_list, rank, alpha)
        else:
            self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

        self.lora.to(device=linear.weight.device)
        self.lora_enabled = True  # toggle flag

    def enable_lora(self, mode=True):
        """Enable or disable the LoRA adjustment."""
        self.lora_enabled = mode

    def disable_lora(self):
        """Disable the LoRA adjustment."""
        self.lora_enabled = False
    
    def forward(self, x):
        """
        Forward pass. If LoRA is enabled, its output is added to the quantized linear layer's output.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.lora_enabled:
            return F.linear(x, self.linear_weight, self.linear_bias) + self.lora(x)
        else:
            return F.linear(x, self.linear_weight, self.linear_bias)


class Quantized4BitLinearWithLoRA(nn.Module):
    """
    A wrapper module that replaces a `nn.Linear` layer with a version that has
    an NF4-quantized 4-bit weight and an attached LoRA layer.

    Args:
        linear (nn.Linear): The original linear layer.
        fused_dim_list (list[int], optional): For fused layers, the dimensions of original layers. Defaults to None.
        rank (int, optional): The rank for LoRA. Defaults to 4.
        alpha (int, optional): The scaling factor for LoRA. Defaults to 1.
    """
    def __init__(self, linear, fused_dim_list=None, rank=4, alpha=1):
        super().__init__()

        self.linear_weight = quantize_nf4(linear.weight)
        self.linear_weight[0].requires_grad_(False)

        if linear.bias is not None:
            self.linear_bias = quantize_nf4(linear.bias)
            self.linear_bias[0].requires_grad_(False)
        else:
            self.linear_bias = None

        if fused_dim_list:
            self.fused_dim_list = fused_dim_list
            self.is_fused_linear = True
            assert (
                sum(fused_dim_list) == linear.out_features
            ), f"sum of fused_dim_list {sum(fused_dim_list)} is not equal to linear.out_features {linear.out_features}"
            self.lora = FusedLoRALayer(linear.in_features, fused_dim_list, rank, alpha)
        else:
            self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

        self.lora.to(device=linear.weight.device)
        self.lora_enabled = True  # toggle flag

    def enable_lora(self, mode=True):
        """Enable or disable the LoRA adjustment."""
        self.lora_enabled = mode

    def disable_lora(self):
        """Disable the LoRA adjustment."""
        self.lora_enabled = False

    def forward(self, x):
        """
        Forward pass. Dequantizes the weight and bias on the fly. If LoRA is enabled,
        its output is added.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.lora_enabled:
            if self.linear_bias is not None:
                return F.linear(
                    x,
                    dequantize_nf4(*self.linear_weight),
                    dequantize_nf4(*self.linear_bias),
                ) + self.lora(x)
            else:
                return F.linear(x, dequantize_nf4(*self.linear_weight), None) + self.lora(x)
        else:
            if self.linear_bias is not None:
                return F.linear(
                    x,
                    dequantize_nf4(*self.linear_weight),
                    dequantize_nf4(*self.linear_bias),
                )
            else:
                return F.linear(x, dequantize_nf4(*self.linear_weight), None)


class Quantized8bitLinear(nn.Module):
    """
    A simple `nn.Linear` layer replacement with an 8-bit quantized weight and bias.

    Args:
        linear (nn.Linear): The original linear layer to be quantized.
        quant (torch.dtype, optional): The 8-bit float format. Defaults to `torch.float8_e4m3fn`.
        device (str, optional): The device to store the quantized weights on. Defaults to "cpu".
    """
    def __init__(self, linear, quant=torch.float8_e4m3fn, device="cpu"):
        super().__init__()
        assert quant in {
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        }, "Unknown quantization"

        self.linear_weight = linear.weight.to(dtype=quant, device=device)
        if linear.bias is not None:
            self.linear_bias = linear.bias.to(dtype=quant, device=device)
        else:
            self.linear_bias = None

    def forward(self, x):
        """
        Performs the linear operation with the quantized weights.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return F.linear(x, self.linear_weight, self.linear_bias)


class Quantized4bitLinear(nn.Module):
    """
    A simple `nn.Linear` layer replacement with an NF4-quantized 4-bit weight and bias.

    Args:
        linear (nn.Linear): The original linear layer to be quantized.
    """
    def __init__(self, linear):
        super().__init__()

        self.linear_weight = quantize_nf4(linear.weight)
        self.linear_weight[0].requires_grad_(False)

        if linear.bias is not None:
            self.linear_bias = quantize_nf4(linear.bias)
            self.linear_bias[0].requires_grad_(False)
        else:
            self.linear_bias = None

    def forward(self, x):
        """
        Performs the linear operation by dequantizing the weights on the fly.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.linear_bias is not None:
            return F.linear(
                x,
                dequantize_nf4(*self.linear_weight),
                dequantize_nf4(*self.linear_bias),
            )
        else:
            return F.linear(x, dequantize_nf4(*self.linear_weight), None)


def swap_linear_simple(
    model, replacement_module, include_keywords=None, **module_kwargs
):
    """
    Recursively traverses a model and replaces `nn.Linear` layers with a specified
    replacement module, but only for layers whose names contain given keywords.

    Args:
        model (nn.Module): The model to modify.
        replacement_module (nn.Module): The class of the module to substitute for `nn.Linear`.
        include_keywords (list[str], optional): A list of keywords. A linear layer is only
                                                replaced if its name contains one of these keywords.
                                                If None or empty, no layers are replaced.
        **module_kwargs: Additional keyword arguments to be passed to the `replacement_module`'s constructor.
    """
    # explicitly included layers
    if include_keywords is None:
        include_keywords = []

    def should_replace(name):
        """Replace only if the name matches one of the include keywords, or replace all if empty."""
        return not include_keywords or any(
            keyword in name for keyword in include_keywords
        )

    def recursive_swap(module, parent_name=""):
        for name, child in module.named_children():
            current_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, nn.Linear) and should_replace(current_name):
                setattr(module, name, replacement_module(child, **module_kwargs))
                print(f"Replacing {current_name}")
                log.info(f"Replacing {current_name}")
            else:
                recursive_swap(child, current_name)

    recursive_swap(model)


def swap_linear_recursive(
    model,
    replacement_module,
    exclude_keywords=None,
    fused_linear_patterns=None,
    **module_kwargs,
):
    """
    Recursively traverses a model and replaces `nn.Linear` layers with a specified
    replacement module, supporting exclusions and special handling for fused layers.

    Args:
        model (nn.Module): The model to modify.
        replacement_module (nn.Module): The class of the module to substitute for `nn.Linear`.
        exclude_keywords (list[str], optional): A list of keywords. If a layer's name contains
                                                any of these keywords, it will not be replaced.
                                                Defaults to [].
        fused_linear_patterns (list[tuple[str, list[int]]], optional):
            A list of tuples for handling fused layers. Each tuple should contain:
            - A regex pattern to match the name of the fused layer.
            - A list of integers representing the original output dimensions.
            Defaults to [].
        **module_kwargs: Additional keyword arguments passed to the `replacement_module`'s constructor.
    """
    # omitted layers
    if exclude_keywords is None:
        exclude_keywords = []

    # store regex pattern of fused layers
    if fused_linear_patterns is None:
        fused_linear_patterns = []

    def is_fused_linear(name):
        # check if the linear is fused linear like qkv linear fused together
        return any(re.match(pattern, name) for pattern, _ in fused_linear_patterns)

    def get_fused_dim_list(name):
        # how to partition the fused linear because some layer can be partitioned unevenly
        # ie: [256, 192, 256]
        for pattern, dim_list in fused_linear_patterns:
            if re.match(pattern, name):
                return dim_list
        return None

    def recursive_swap(module, parent_name=""):
        for name, child in module.named_children():
            current_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, nn.Linear) and not any(
                keyword in current_name for keyword in exclude_keywords
            ):
                if fused_linear_patterns != None and is_fused_linear(current_name):
                    fused_dim_list = get_fused_dim_list(current_name)
                    setattr(
                        module,
                        name,
                        replacement_module(
                            child, fused_dim_list=fused_dim_list, **module_kwargs
                        ),
                    )
                    print(
                        f"Replacing fused linear layer {current_name} with dimensions {fused_dim_list}"
                    )
                    log.info(
                        f"Replacing fused linear layer {current_name} with dimensions {fused_dim_list}"
                    )
                else:
                    setattr(module, name, replacement_module(child, **module_kwargs))
                    print(f"Replacing {current_name}")
                    log.info(f"Replacing {current_name}")
            else:
                recursive_swap(child, current_name)

    recursive_swap(model)


def swap_linear(model, replacement_module, exclude_keywords=None, **module_kwargs):
    """
    Iteratively traverses a model and replaces `nn.Linear` layers with a specified
    replacement module, supporting exclusions. This is a non-recursive implementation.

    Args:
        model (nn.Module): The model to modify.
        replacement_module (nn.Module): The class of the module to substitute for `nn.Linear`.
        exclude_keywords (list[str], optional): A list of keywords. If a layer's name contains
                                                any of these keywords, it will not be replaced.
                                                Defaults to [].
        **module_kwargs: Additional keyword arguments passed to the `replacement_module`'s constructor.

    Returns:
        nn.Module: The modified model.
    """
    if exclude_keywords is None:
        exclude_keywords = []

    # non recursive solution
    stack = [(model, "")]
    while stack:
        module, parent_name = stack.pop()
        for name, child in list(module.named_children()):
            current_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, nn.Linear) and not any(
                keyword in current_name for keyword in exclude_keywords
            ):
                # replace nn.Linear with another module
                setattr(module, name, replacement_module(child, **module_kwargs))
                log.info(f"Replacing {current_name}")
            else:
                stack.append((child, current_name))

    return model


def find_lora_params(model):
    """
    Finds all LoRA-specific parameters in a model.

    Args:
        model (nn.Module): The model to search.

    Returns:
        list[tuple[str, nn.Parameter]]: A list of tuples, where each tuple contains the
                                        parameter's name and the parameter itself.
    """
    lora_params = []
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            lora_params.append((n, p))
    return lora_params


def change_lora_scale(model, lora_instance, scale):
    """
    Traverses a model and changes the scaling factor of all LoRA layers.

    Args:
        model (nn.Module): The model containing LoRA layers.
        lora_instance (type): The class type of the LoRA-wrapped layer
                              (e.g., `LinearWithLoRA`).
        scale (float): The new scaling factor to apply.
    """
    def traverse_model(model, lora_instance, scale, path=""):
        for name, module in model.named_children():
            current_path = f"{path}.{name}" if path else name
            if isinstance(module, lora_instance):
                module.lora.scaling = scale
                log.info(f"changing {current_path} lora scale to {scale}")
            else:
                traverse_model(
                    model=module,
                    lora_instance=lora_instance,
                    scale=scale,
                    path=current_path,
                )

    traverse_model(model, lora_instance, scale, "")


def merge_lora_weights(model, replacement_module=LinearWithLoRA):
    """
    Merges the LoRA weights back into their corresponding original linear layers
    and replaces the LoRA-wrapped module with the original, now-updated, linear layer.

    Args:
        model (nn.Module): The model with LoRA layers to be merged.
        replacement_module (type, optional): The class type of the LoRA-wrapped layer.
                                             Defaults to `LinearWithLoRA`.

    Returns:
        nn.Module: The model with LoRA weights merged.
    """
    for name, module in model.named_children():
        if isinstance(module, replacement_module):
            original_linear = module.linear
            lora_layer = module.lora

            if isinstance(lora_layer, LoRALayer):
                # Merge regular LoRA weights
                merged_weight = (
                    original_linear.weight
                    + (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
                )

                # Update the original linear layer's weight
                original_linear.weight.data.copy_(merged_weight)

            elif isinstance(lora_layer, FusedLoRALayer):
                # Merge Fused LoRA weights
                lora_As = lora_layer.lora_As.chunk(
                    len(lora_layer.fused_dim_list), dim=0
                )
                merged_weight = original_linear.weight.clone()

                start_idx = 0
                for i, (lora_A, lora_B) in enumerate(zip(lora_As, lora_layer.lora_Bs)):
                    end_idx = start_idx + lora_layer.fused_dim_list[i]
                    merged_weight[start_idx:end_idx] += (
                        lora_B @ lora_A
                    ) * lora_layer.scaling
                    start_idx = end_idx

                # Update the original linear layer's weight
                original_linear.weight.data.copy_(merged_weight)

            # Replace the LoRA module with the original linear layer
            setattr(model, name, original_linear)
        else:
            # Recursively apply to child modules
            merge_lora_weights(module, replacement_module)

    return model


def set_lora_enabled(module, enabled=True, lora_module=LinearWithLoRA):
    """
    Traverses a model and enables or disables all LoRA layers.

    Args:
        module (nn.Module): The model to modify.
        enabled (bool, optional): Whether to enable (True) or disable (False) LoRA.
                                  Defaults to True.
        lora_module (type, optional): The class type of the LoRA-wrapped layer.
                                      Defaults to `LinearWithLoRA`.
    """
    for child in module.modules():
        if isinstance(child, lora_module):
            child.enable_lora(enabled)