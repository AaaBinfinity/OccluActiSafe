"""N14两种AEB×已采集角度/在线示意独立视频；IDE默认最近完整运行。"""
import argparse
from datetime import datetime
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import uuid

import imageio_ffmpeg
from PIL import Image, __version__ as pillow_version


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path('H:/OccluActiSafeDataset/artifacts')
BRANCHES = {'local_aeb': '仅车端AEB', 'cooperative_aeb': '车路协同AEB'}
VIEWS = {'车端': '车内前向视角', '路侧': '路侧视角', '跟车': '跟车视角',
         '俯瞰': '俯瞰实景视角', '前侧': '前侧观察视角', '示意': '实时俯瞰示意图'}
SIZE = (1280, 720)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_check(path, expected_frames, fps):
    """实际逐帧解码，只保留当前帧，不把视频整体载入内存。"""
    stream = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        metadata = next(stream)
        count = 0
        for frame in stream:
            if len(frame) != SIZE[0] * SIZE[1] * 3:
                raise ValueError(f'解码帧字节数不符：{path}')
            count += 1
    finally:
        stream.close()
    passed = (count == expected_frames and tuple(metadata['size']) == SIZE
              and math.isclose(metadata['fps'], fps, abs_tol=1e-6))
    return {'passed': passed, 'decoded_frames': count, 'expected_frames': expected_frames,
            'size': metadata['size'], 'fps': metadata['fps'],
            'encoded_duration_s': metadata['duration']}


def export(run_dir=None, output_dir=None):
    if run_dir is None:
        candidates=[p for p in ARTIFACTS.glob('n14_*') if (p/'配对验收报告.json').is_file()
                    and all((p/b/'公共前缀.json').is_file() for b in ('reference',*BRANCHES))]
        if not candidates:raise FileNotFoundError('没有已分析的完整N14三分支运行')
        run_dir=max(candidates,key=lambda p:(p/'源码版本.json').stat().st_mtime)
    run_dir = Path(run_dir).resolve()
    if read(run_dir/'配对验收报告.json')['failed']:raise ValueError('最近运行验收失败，拒绝回退旧结果冒充本轮成功')
    original_index = run_dir / '证据索引.json'
    frozen = read(original_index)['files']
    identifier = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = Path(output_dir).resolve() if output_dir else run_dir / '独立视角视频' / identifier
    if output.is_relative_to(ROOT) or not output.is_relative_to(run_dir):
        raise ValueError('视频输出必须在仓库外的原运行目录内')
    output.mkdir(parents=True, exist_ok=False)
    report = {'export_id': identifier, 'source_run': str(run_dir),
              'started_at': datetime.now().astimezone().isoformat(), 'status': 'running',
              'export_code_commit': subprocess.check_output(
                  ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'export_source_sha256': sha(Path(__file__)),
              'original_evidence_index_sha256': sha(original_index),
              'source_simulation_version': read(run_dir / '源码版本.json'),
              'environment': {'os': platform.platform(), 'python': sys.version,
                              'pillow': pillow_version, 'imageio_ffmpeg': imageio_ffmpeg.__version__},
              'camera_note': '车内前向为既有车端前向相机(1.5,0,1.6)m、yaw=-12度，不含仪表台内饰',
              'processing': '实际RGB及在线生成示意各自独立编码；不拼接、不裁剪、不缩放、不插帧、不补齐碰撞后未来',
              'videos': []}
    manifest = output / '独立视频来源与复核.json'

    def persist():
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(output, flush=True)
    persist()
    try:
        for branch, branch_name in BRANCHES.items():
            folder = run_dir / branch
            index_path = folder / '相机帧索引.json'
            relative_index = index_path.relative_to(run_dir).as_posix()
            if sha(index_path) != frozen[relative_index]:
                raise ValueError(f'原始相机帧索引哈希不符：{index_path}')
            index = read(index_path)
            config_path = folder / '运行配置.json'
            if sha(config_path) != frozen[config_path.relative_to(run_dir).as_posix()]:
                raise ValueError('原始运行配置哈希不符')
            interval = read(config_path)['image_interval_ticks']
            source_indexes={camera:index_path for camera in ('车端','路侧','跟车',*read(config_path).get('extra_cameras',{}))}
            schematic=folder/'示意帧索引.json'
            if schematic.is_file():
                if sha(schematic)!=frozen[schematic.relative_to(run_dir).as_posix()]:raise ValueError('在线示意索引哈希不符')
                index.extend(read(schematic));source_indexes['示意']=schematic
            available={r['source'] for r in index}
            expected=set(source_indexes)
            if read(config_path).get('overview',{}).get('enabled'):expected.add('示意')
            if not expected<=available or not available<=set(VIEWS):raise ValueError('实际角度缺失或未声明')
            period = .05 * interval
            fps = 1 / period
            for camera, view_name in VIEWS.items():
                if camera not in available:continue
                index_path=source_indexes[camera]
                records = sorted((r for r in index if r['source'] == camera), key=lambda r: r['index'])
                if not records or records[0]['index'] != 0:
                    raise ValueError(f'{branch}/{camera}缺少起始相机帧')
                for left, right in zip(records, records[1:]):
                    if (right['index'] - left['index'] != interval
                            or (camera!='示意' and right['frame'] - left['frame'] != interval)
                            or not math.isclose(right['time_s'] - left['time_s'], period, abs_tol=1e-6)):
                        raise ValueError(f'{branch}/{camera}存在重复、缺失或不等间隔帧')
                video = output / f'{branch_name}_{view_name}.mp4'
                result = {'branch': branch, 'camera': camera, 'video': video.name,
                          'source_index_sha256': sha(index_path), 'fps': fps, 'size': list(SIZE),
                          'frames': len(records), 'first_source_time_s': records[0]['time_s'],
                          'last_source_time_s': records[-1]['time_s'], 'source_frames': []}
                report['videos'].append(result)
                command = [imageio_ffmpeg.get_ffmpeg_exe(), '-n', '-v', 'error', '-f', 'rawvideo',
                           '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', str(fps), '-i', 'pipe:0',
                           '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', str(video)]
                result['encoder_command'] = command
                with (output / f'{video.stem}_编码日志.txt').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
                    try:
                        for record in records:
                            source = (folder / record['image']).resolve()
                            if not source.is_relative_to(folder):
                                raise ValueError('相机帧路径越出分支目录')
                            raw = source.read_bytes()
                            digest = hashlib.sha256(raw).hexdigest()
                            if digest != frozen[source.relative_to(run_dir).as_posix()]:
                                raise ValueError(f'原始图片哈希不符：{source}')
                            with Image.open(BytesIO(raw)) as image:
                                if image.size != SIZE:
                                    raise ValueError('相机原始尺寸不符，拒绝缩放')
                                process.stdin.write(image.convert('RGB').tobytes())
                            result['source_frames'].append(dict(record, sha256=digest))
                    finally:
                        process.stdin.close()
                        process.wait(timeout=60)
                    if process.returncode:
                        raise RuntimeError(f'独立相机视频编码失败：{video}')
                result['sha256'] = sha(video)
                result['decode_check'] = decode_check(video, len(records), fps)
                persist()
                if not result['decode_check']['passed']:
                    raise RuntimeError(f'独立相机视频逐帧解码复核失败：{video}')
                print(f'{video.name}: {len(records)}帧，{fps:g}fps，解码通过', flush=True)
        report['status'] = 'completed'
        report['checks'] = {'passed': len(report['videos']), 'failed': 0, 'skipped': 0, 'not_run': 0}
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        report['finished_at'] = datetime.now().astimezone().isoformat()
        persist()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)
