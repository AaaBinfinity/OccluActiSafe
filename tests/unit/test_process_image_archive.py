"""共享图片归档：原位批量登记、失败续传及真实重命名不丢源。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec=importlib.util.spec_from_file_location('archive_images',Path(__file__).resolve().parents[2]/'scripts/archive_process_images.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ArchiveTests(unittest.TestCase):
    def test_inplace_registration_is_atomic_and_retry_keeps_all_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);workspace=root/'repo';metadata=workspace/'docs/过程图';artifacts=root/'external/artifacts';images=artifacts/'n14_fixture';images.mkdir(parents=True)
            for name in ('中文图一.svg','中文图二.svg'):(images/name).write_text('<svg/>',encoding='utf-8')
            with patch.object(module,'PROJECT_ROOT',workspace),patch.object(module,'write',side_effect=PermissionError('fixture')):
                with self.assertRaises(PermissionError):module.archive(artifacts,metadata)
            self.assertEqual(len(list(images.glob('*.svg'))),2)
            with patch.object(module,'PROJECT_ROOT',workspace):
                self.assertEqual(module.archive(artifacts,metadata),{'verified':2,'moved':0})
                self.assertEqual(module.archive(artifacts,metadata,verify_only=True)['verified'],2)
            catalog=json.loads((metadata/'图片来源与哈希.json').read_text(encoding='utf-8'))
            self.assertEqual(len(catalog['images']),2)

    def test_real_rename_does_not_delete_source_before_catalog_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);workspace=root/'repo';metadata=workspace/'docs/过程图';artifacts=root/'external/artifacts';images=artifacts/'n11_fixture';images.mkdir(parents=True)
            source=images/'trajectory_preview.svg';source.write_text('<svg/>',encoding='utf-8')
            with patch.object(module,'PROJECT_ROOT',workspace),patch.object(module,'write',side_effect=PermissionError('fixture')):
                with self.assertRaises(PermissionError):module.archive(artifacts,metadata)
            self.assertTrue(source.is_file())
            with patch.object(module,'PROJECT_ROOT',workspace):self.assertEqual(module.archive(artifacts,metadata)['verified'],1)
            self.assertFalse(source.exists());self.assertTrue((images/'轨迹预览.svg').is_file())


if __name__=='__main__':unittest.main()
