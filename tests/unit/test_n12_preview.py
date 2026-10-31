"""RGB预览归档的机制检查；真实窗口和真实CARLA另行验收。"""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from occluactisafe.common.schemas import TickEvent
from occluactisafe.simulation.observation_adapter import stationary_control
from occluactisafe.simulation.preview import FrameArchive


def event(frame):return TickEvent('1.0','preview','episode',1,frame,'carla',frame*.05,())
def camera(frame,valid=True):
    return {'frame':frame,'valid':valid,'width':320 if valid else None,'height':180 if valid else None,
            'bgra':base64.b64encode(bytes([10,20,30,255])*(320*180)).decode() if valid else None,
            'reason':None if valid else 'timeout'}


class PreviewTests(unittest.TestCase):
    def test_bgra_channels_png_hash_and_control_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=FrameArchive(directory);tick=event(1);control=stationary_control(tick,0);before=control.to_json()
            picture,row=archive.update(tick,camera(1),control);archive.close()
            self.assertEqual(picture.getpixel((0,0)),(30,20,10));self.assertEqual(control.to_json(),before)
            metadata=json.loads((Path(directory)/'预览图片来源.json').read_text(encoding='utf-8'))
            for entry in metadata['images']:
                path=Path(directory)/entry['image'];self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])
                with Image.open(path) as img:self.assertEqual(img.size,(320,180))
            self.assertEqual(metadata['source_sha256'],hashlib.sha256((Path(directory)/'预览帧记录.jsonl').read_bytes()).hexdigest())

    def test_invalid_frame_clears_previous_image_and_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=FrameArchive(directory)
            for frame,valid in ((1,True),(2,False)):
                picture,row=archive.update(event(frame),camera(frame,valid),stationary_control(event(frame),frame))
            self.assertNotEqual(picture.getpixel((0,0)),(30,20,10));self.assertFalse(row['valid']);self.assertIsNone(row['raw_bgra_sha256'])
            self.assertTrue(any(item['kind']=='invalid_placeholder' for item in archive.images));archive.close()

    def test_periodic_manual_and_last_frames_are_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=FrameArchive(directory,interval=2)
            for frame in range(1,4):archive.update(event(frame),camera(frame),stationary_control(event(frame),frame))
            archive.save();archive.close()
            self.assertEqual([item['frame'] for item in archive.images],[1,2,3,3])
            self.assertEqual(len({item['image'] for item in archive.images}),4)

    def test_wrong_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=FrameArchive(directory)
            with self.assertRaises(ValueError):archive.update(event(1),camera(2),stationary_control(event(1),0))
            archive.close();self.assertEqual(archive.count,0)


if __name__=='__main__':unittest.main()
