import argparse

import torch

from aimb import Module
from aimb.rpc import RPCExecutionDevice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_model', type=str, default='model.pth')
    params = parser.parse_args().pt_model
    
    model = torch.nn.Sequential(
        torch.nn.Linear(3819, 36), torch.nn.Tanh(), torch.nn.Linear(36, 9))
    model.load_state_dict(params)

    torch.onnx.export(model,
                      torch.randn(1, 3819, requires_grad=True), 
                      "model.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    module = Module(model_path="model.onnx", executor='tensorrt')
    rpc_ip_address = "172.31.20.199"
    rpc_device = RPCExecutionDevice(rpc_ip_address, '50051')
    module.create_executor(device=rpc_device)
    module.evaluate(number=2048)
