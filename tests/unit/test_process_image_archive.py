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
    def test_frozen_rgb_cache_is_excluded_but_generated_graph_is_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);node=root/'节点资料/N17_检测器比较/02_运行产物';raw=node/'冻结数据';raw.mkdir(parents=True)
            index=raw/'逐帧数据索引.jsonl';index.write_text('{}\n',encoding='utf-8')
            (raw/'source.jpg').write_bytes(b'original-rgb')
            module.write(raw/'数据清单.json',dict(status='completed',index_sha256=module.sha(index)))
            generated=node/'实际训练曲线.svg';generated.write_bytes(b'<svg/>')
            self.assertEqual(list(module.artifact_images(root)),[generated])
            index.write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):list(module.artifact_images(root))

    def test_legacy_catalog_moves_to_disk_with_identical_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();workspace=root/'repo';metadata=workspace/'docs/过程图';metadata.mkdir(parents=True)
            artifacts=root/'external/artifacts';images=artifacts/'n14_fixture';images.mkdir(parents=True)
            image=images/'旧中文图.svg';image.write_bytes(b'<svg/>')
            entry=dict(original_artifact_path='n14_fixture/旧中文图.svg',artifact_path='n14_fixture/旧中文图.svg',
                legacy_repository_path='docs/过程图/n14_fixture/旧中文图.svg',sha256=module.sha(image),bytes=image.stat().st_size)
            original=(json.dumps(dict(schema_version='process-images-2',images=[entry]),ensure_ascii=False)+'\r\n').encode('utf-8')
            path=metadata/'图片来源与哈希.json';path.write_bytes(original)
            with patch.object(module,'PROJECT_ROOT',workspace):self.assertEqual(module.archive(artifacts,metadata)['verified'],1)
            pointer=json.loads(path.read_text('utf-8'));reference=pointer['migrated_repository_catalog']
            self.assertEqual((artifacts/reference['artifact_path']).read_bytes(),original)
            self.assertEqual(module.load_catalog(artifacts,path)['images'],[entry])

    def test_inplace_registration_is_atomic_and_retry_keeps_all_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();workspace=(root/'repo').resolve();metadata=workspace/'docs/过程图';artifacts=root/'external/artifacts';images=artifacts/'n14_fixture';images.mkdir(parents=True)
            for name in ('中文图一.svg','中文图二.svg'):(images/name).write_text('<svg/>',encoding='utf-8')
            with patch.object(module,'PROJECT_ROOT',workspace),patch.object(module,'write',side_effect=PermissionError('fixture')):
                with self.assertRaises(PermissionError):module.archive(artifacts,metadata)
            self.assertEqual(len(list(images.glob('*.svg'))),2)
            with patch.object(module,'PROJECT_ROOT',workspace):
                self.assertEqual(module.archive(artifacts,metadata),{'verified':2,'moved':0})
                self.assertEqual(module.archive(artifacts,metadata,verify_only=True)['verified'],2)
            catalog=module.load_catalog(artifacts,metadata/'图片来源与哈希.json')
            self.assertEqual(len(catalog['images']),2)
            summary=json.loads((metadata/'图片来源与哈希.json').read_text('utf-8'))
            self.assertEqual(summary['image_count'],2);self.assertNotIn('images',summary)
            self.assertLess((metadata/'图片来源与哈希.json').stat().st_size,4096)

    def test_summary_publication_failure_preserves_old_catalog_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();workspace=root/'repo';metadata=workspace/'docs/过程图';artifacts=root/'external/artifacts';images=artifacts/'n14_fixture';images.mkdir(parents=True)
            (images/'中文图一.svg').write_text('<svg/>',encoding='utf-8')
            with patch.object(module,'PROJECT_ROOT',workspace):module.archive(artifacts,metadata)
            pointer=metadata/'图片来源与哈希.json';before=pointer.read_bytes();first=json.loads(before)
            (images/'中文图二.svg').write_text('<svg>new</svg>',encoding='utf-8')
            with patch.object(module,'PROJECT_ROOT',workspace),patch.object(module,'write',side_effect=PermissionError('摘要发布失败')):
                with self.assertRaises(PermissionError):module.archive(artifacts,metadata)
            self.assertEqual(pointer.read_bytes(),before)
            self.assertEqual(module.sha(artifacts/first['full_catalog']['artifact_path']),first['full_catalog']['sha256'])
            with patch.object(module,'PROJECT_ROOT',workspace):
                self.assertEqual(module.archive(artifacts,metadata,verify_only=True)['verified'],1)
                self.assertEqual(module.archive(artifacts,metadata)['verified'],2)
            final=json.loads(pointer.read_text('utf-8'));full=artifacts/final['full_catalog']['artifact_path']
            full.write_bytes(b'corrupt')
            with patch.object(module,'PROJECT_ROOT',workspace):
                with self.assertRaises(ValueError):module.archive(artifacts,metadata,verify_only=True)

    def test_real_rename_does_not_delete_source_before_catalog_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();workspace=(root/'repo').resolve();metadata=workspace/'docs/过程图';artifacts=root/'external/artifacts';images=artifacts/'n11_fixture';images.mkdir(parents=True)
            source=images/'trajectory_preview.svg';source.write_text('<svg/>',encoding='utf-8')
            with patch.object(module,'PROJECT_ROOT',workspace),patch.object(module,'write',side_effect=PermissionError('fixture')):
                with self.assertRaises(PermissionError):module.archive(artifacts,metadata)
            self.assertTrue(source.is_file())
            with patch.object(module,'PROJECT_ROOT',workspace):self.assertEqual(module.archive(artifacts,metadata)['verified'],1)
            self.assertFalse(source.exists());self.assertTrue((images/'轨迹预览.svg').is_file())


if __name__=='__main__':unittest.main()
