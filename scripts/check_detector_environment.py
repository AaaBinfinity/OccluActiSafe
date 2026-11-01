"""N17 CUDA前向与torchvision NMS真实设备检查，失败原样落盘。"""
from pathlib import Path
import os
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from occluactisafe.experiments.detector_data import load_config
from occluactisafe.experiments.run_store import write_json, now


def main():
    cfg = load_config()
    out = Path(cfg['output_root'])/'03_验收证据'/('环境检查_'+uuid.uuid4().hex+'.json')
    state = {'started_at': now(), 'status': 'running'}
    os.environ['YOLO_CONFIG_DIR'] = str(Path(cfg['output_root'])/'04_开发与复核'/'Ultralytics设置')
    try:
        import torch
        import torchvision
        import ultralytics
        from torchvision.ops import nms
        state.update(torch=torch.__version__, torchvision=torchvision.__version__, ultralytics=ultralytics.__version__)
        if not torch.cuda.is_available():
            raise RuntimeError('N17需要CUDA')
        state['device'] = torch.cuda.get_device_name(0)
        boxes = torch.tensor([[0.,0.,10.,10.]], device='cuda')
        result = nms(boxes, torch.tensor([.9], device='cuda'), .5)
        torch.cuda.synchronize()
        if result.tolist() != [0]:
            raise RuntimeError('GPU NMS结果错误')
        state.update(status='passed', finished_at=now(), nms_result=result.tolist())
    except BaseException as exc:
        state.update(status='failed', finished_at=now(), error=repr(exc))
        raise
    finally:
        write_json(out, state)
        print(out, flush=True)


if __name__ == '__main__':
    main()
