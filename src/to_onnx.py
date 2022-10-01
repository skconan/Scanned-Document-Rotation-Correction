import argparse
import onnx
import onnxruntime as ort
import torch.onnx
import numpy as np
from models.rotation_net import RotationNet


def main(gpu_id, torch_weight_path, onnx_weight_path):
    model = RotationNet(out_channels=64)

    print("Load weight from %s" % torch_weight_path)

    if gpu_id >= 0:
        device = torch.device('cuda', gpu_id)
        checkpoint = torch.load(torch_weight_path, map_location=device)
        model.cuda()
        model.load_state_dict(checkpoint)
        model.eval()
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(torch_weight_path)
        model.to(device)
        model.load_state_dict(checkpoint)
        model.eval()

    batch_size = 1
    gpu_id = -1

    x = torch.randn(batch_size, 1, 128, 128, device=device)
    torch_out = model(x)
    print(torch_out)

    torch.onnx.export(model,
                      x,
                      onnx_weight_path,
                      verbose=True,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])

    onnx_model = onnx.load(onnx_weight_path)
    onnx.checker.check_model(onnx_model)

    print(onnx.helper.printable_graph(onnx_model.graph))

    # Inference Testing
    ort_session = ort.InferenceSession(onnx_weight_path)
    outputs = ort_session.run(
        None,
        {"input": np.random.randn(1, 1, 128, 128).astype(np.float32)},
    )
    print(outputs)
    print("Exported model success")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-gpu_id', '--gpu_id', type=int, required=False, help='GPU ID. (-1 for CPU)',
                    default=0)
    ap.add_argument('-torch_w', '--torch_weight_path', type=str, required=False, help='Torch weight path',
                    default='./models/model_rotation_net.pt')
    ap.add_argument('-onnx_w', '--onnx_weight_path', type=str, required=False, help='Output ONNX weight path',
                    default='./models/model_rotation_net.onnx')
    args = vars(ap.parse_args())

    gpu_id = args['gpu_id']
    torch_weight_path = args['torch_weight_path']
    onnx_weight_path = args['onnx_weight_path']

    main(gpu_id, torch_weight_path, onnx_weight_path)
