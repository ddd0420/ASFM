import torch
from thop import profile, clever_format


def count_params_and_flops(
    model,
    input_shape=(1, 1, 256, 256),
    device="cpu",
    verbose=True
):
    """
    统计模型参数量和 FLOPs（支持 DataParallel / DDP）

    参数:
        model (torch.nn.Module): PyTorch 模型
        input_shape (tuple): 输入张量形状 (N, C, H, W)
        device (str): 'cpu' or 'cuda'
        verbose (bool): 是否打印结果

    返回:
        params (float): 参数量（M）
        flops (float): 计算量（G)
    """

    # 兼容 DataParallel / DDP
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if hasattr(model, "module"):
        model = model.module

    model = model.to(device)
    model.eval()

    # 构造 dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # 统计 FLOPs 和 Params
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 格式化（自动转为 M / G）
    flops, params = clever_format([flops, params], format="%.2f")

    if verbose:
        print(f"📌 Model Params : {params}")
        print(f"⚡ Model FLOPs  : {flops}")

    return params, flops
