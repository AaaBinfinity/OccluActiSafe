import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from occluactisafe.experiments.detector_data import load_config, load_index
from occluactisafe.experiments.detector_portable import bounded, row_paths, stage_snapshot, verify_input
from occluactisafe.experiments.run_store import sha256, write_json
from occluactisafe.experiments.source_version import source_info, VERSION_FILE


class PortableTests(unittest.TestCase):
    def fixture(self, root):
        row = dict(key='spd_vehicle_1', role='train', domain='SPD', image='H:\\old\\frame.jpg',
                   source_image='H:\\original\\frame.jpg', gt=[], image_sha256='', label_sha256='')
        image = root/'data/images/train/spd_vehicle_1.jpg';image.parent.mkdir(parents=True);image.write_bytes(b'mechanism-fixture-only')
        label = root/'data/labels/train/spd_vehicle_1.txt';label.parent.mkdir(parents=True);label.write_text('\n')
        row.update(image_sha256=sha256(image), label_sha256=sha256(label))
        index=root/'data/逐帧数据索引.jsonl';index.write_text(json.dumps(row)+'\n',encoding='utf-8')
        digest=sha256(index)
        write_json(root/'data/数据清单.json',{'status':'completed','index_sha256':digest,'config_sha256':'fixture-config'})
        assets=[{'path':p.relative_to(root).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size}
                for p in (image,label,index,root/'data/数据清单.json')]
        write_json(root/'输入清单.json',{'schema_version':'n17-portable-input-1','status':'completed',
                   'source_index_sha256':digest,'source_config_sha256':'fixture-config','classes':load_config()['classes'],
                   'counts':{'SPD:train':1},'assets':assets})
        (root/'weights').mkdir()
        return row,digest

    def test_windows_source_path_is_not_used_as_server_image_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row,digest=self.fixture(root)
            with patch('occluactisafe.experiments.detector_portable.INDEX_SHA',digest),patch('occluactisafe.experiments.detector_portable.CONFIG_SHA','fixture-config'),patch('occluactisafe.experiments.detector_portable.WEIGHTS',{}):
                rows,report=verify_input(root)
                self.assertEqual(report['images'],1)
                before=(root/'data/逐帧数据索引.jsonl').read_bytes()
                snap=stage_snapshot(root,rows,root/'snapshot')
            cfg=load_config();cfg['dataset_snapshot']=snap
            bound,audit=load_index(cfg)
            self.assertTrue(os.path.samefile(bound[0]['image'],root/'data/images/train/spd_vehicle_1.jpg'))
            self.assertEqual(bound[0]['original_image'],row['image'])
            self.assertEqual((root/'data/逐帧数据索引.jsonl').read_bytes(),before)
            self.assertNotIn('test',__import__('yaml').safe_load((root/'snapshot/dataset.yaml').read_text()))

    def test_paths_reject_escape_absolute_drive_and_noncanonical_names(self):
        with tempfile.TemporaryDirectory() as temp:
            for relative in ('../secret','/etc/file','D:/secret','a\\b','a/../b','a//b','.'):
                with self.subTest(relative=relative),self.assertRaises(ValueError):bounded(Path(temp),relative)

    def test_test_role_and_unsafe_key_are_rejected_before_media_open(self):
        for row in ({'role':'test','key':'x','image':'H:/x.jpg'},
                    {'role':'train','key':'../../x','image':'H:/x.jpg'}):
            with self.assertRaises(ValueError):row_paths(row)

    def test_changed_label_and_extra_test_media_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,digest=self.fixture(root)
            with patch('occluactisafe.experiments.detector_portable.INDEX_SHA',digest),patch('occluactisafe.experiments.detector_portable.CONFIG_SHA','fixture-config'),patch('occluactisafe.experiments.detector_portable.WEIGHTS',{}):
                extra=root/'data/test.jpg';extra.write_bytes(b'forbidden')
                with self.assertRaisesRegex(ValueError,'未登记'):verify_input(root)
                extra.unlink()
                (root/'data/labels/train/spd_vehicle_1.txt').write_text('0 0 0 0 0\n')
                with self.assertRaises(ValueError):verify_input(root)

    def test_package_source_checks_tampering_and_extra_python(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'src/a.py';p.parent.mkdir();p.write_text('print(1)\n')
            write_json(root/VERSION_FILE,{'source_commit':'a'*40,'files':{'src/a.py':sha256(p)}})
            info=source_info(root);self.assertEqual(info['source_kind'],'verified_source_package')
            p.write_text('print(2)\n')
            with self.assertRaisesRegex(ValueError,'已修改'):source_info(root)

    def test_snapshot_manifest_and_class_order_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row,digest=self.fixture(root)
            with patch('occluactisafe.experiments.detector_portable.INDEX_SHA',digest):
                snap=stage_snapshot(root,[row],root/'snapshot')
            cfg=load_config();cfg['dataset_snapshot']=snap
            cfg['classes']=list(reversed(cfg['classes']))
            with self.assertRaises(ValueError):load_index(cfg)


if __name__=='__main__':unittest.main()
