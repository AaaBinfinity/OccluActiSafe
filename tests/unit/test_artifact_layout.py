"""硬盘归类的范围、冲突、实际Windows链接及旧图片清单兼容性。"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


layout = load('artifact_layout', 'organize_artifacts.py')
archive = load('layout_images', 'archive_process_images.py')


def fixture(root):
    source = root/'n14_fixture'; source.mkdir(parents=True)
    (source/'证据.json').write_text('{"状态":"完成"}', encoding='utf-8')
    (source/'中文图.svg').write_text('<svg/>', encoding='utf-8')
    return {'entries': [{'node': 'N14', 'purpose': '机制夹具', 'directory': 'last_n14_fixture',
                         'original_artifact_path': 'n14_fixture', 'evidence_file': '证据.json',
                         'evidence_sha256': layout.sha(source/'证据.json')}]}


class LayoutTests(unittest.TestCase):
    def test_preview_classifies_without_moving_or_creating(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()/'artifacts'; catalog = fixture(root)
            for name in ('n14_failed', 'n14_20261031', 'n04', 'uv_cache'): (root/name).mkdir()
            before = sorted(p.name for p in root.iterdir())
            moves = {x['original']: x['physical'] for x in layout.build_plan(root, catalog)['moves']}
            self.assertIn('02_运行产物', moves['n14_fixture'])
            self.assertIn('04_开发与复核', moves['n14_failed'])
            self.assertIn('03_验收证据', moves['n14_20261031'])
            self.assertIn('03_工具缓存', moves['uv_cache'])
            self.assertEqual(before, sorted(p.name for p in root.iterdir()))

    def test_traversal_escape_and_unknown_content_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()/'artifacts'; catalog = fixture(root)
            for relative in ('../外部', str(root/'绝对路径'), '.'):
                with self.assertRaises(ValueError): layout.bounded(root, relative)
            outside = root.parent/'外部'; outside.mkdir()
            (root/'逃逸链接').symlink_to(outside, target_is_directory=True)
            try:
                with self.assertRaises(ValueError): layout.bounded(root, '逃逸链接/目标')
            finally: (root/'逃逸链接').unlink()
            (root/'未知目录').mkdir()
            with self.assertRaisesRegex(ValueError, '未识别'): layout.build_plan(root, catalog)

    def test_collision_keeps_both_original_and_existing_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()/'artifacts'; catalog = fixture(root)
            expected = root/layout.node_folder('N14')/'02_运行产物/n14_fixture'
            expected.mkdir(parents=True); (expected/'用户文件.txt').write_text('保留', encoding='utf-8')
            with self.assertRaises(FileExistsError): layout.build_plan(root, catalog)
            self.assertTrue((root/'n14_fixture/证据.json').is_file())
            self.assertEqual((expected/'用户文件.txt').read_text('utf-8'), '保留')

    def test_real_directory_and_file_links_preserve_identity_and_visible_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()/'artifacts'; catalog = fixture(root)
            (root/'图片核验.json').write_text('{"值":1}', encoding='utf-8')
            original = root/'n14_fixture'; identity = original.stat().st_ino
            report = root/'公共管理/02_目录管理/机制核验.json'
            updated = layout.apply_plan(layout.build_plan(root, catalog), catalog, report)
            new = root/updated['entries'][0]['physical_artifact_path']
            self.assertTrue(os.path.samefile(original, new))
            self.assertEqual(new.stat().st_ino, identity)
            self.assertTrue(original.lstat().st_file_attributes & 2)
            self.assertFalse(new.lstat().st_file_attributes & 2)
            old_file = root/'图片核验.json'
            self.assertTrue(old_file.is_symlink())
            self.assertTrue(old_file.lstat().st_file_attributes & 2)
            self.assertFalse(old_file.resolve().lstat().st_file_attributes & 2)
            self.assertEqual(old_file.read_text('utf-8'), '{"值":1}')
            settings = json.loads((root/'.vscode/settings.json').read_text('utf-8'))
            self.assertTrue(settings['files.exclude']['n14_fixture'])
            self.assertTrue(settings['files.exclude']['last_n14_fixture'])
            self.assertNotIn('节点资料', settings['files.exclude'])
            settings['editor.fontSize'] = 17
            (root/'.vscode/settings.json').write_text(json.dumps(settings), encoding='utf-8')
            layout.configure_explorer(root)
            self.assertEqual(json.loads((root/'.vscode/settings.json').read_text('utf-8'))['editor.fontSize'], 17)
            self.assertEqual(layout.build_plan(root, updated)['moves'], [])
            with self.assertRaises(FileExistsError): layout.make_link(root/'n14_fixture', root/'公共管理')
            # 联接只移除入口本身，TemporaryDirectory不沿它递归。
            for p in (original, root/'last_n14_fixture', root/updated['entries'][0]['directory']): os.rmdir(p)

    def test_archive_matches_old_logical_paths_and_skips_aliases_and_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve(); root = parent/'artifacts'; catalog = fixture(root)
            workspace = parent/'repo'; metadata = workspace/'docs/过程图'
            with patch.object(archive, 'PROJECT_ROOT', workspace):
                self.assertEqual(archive.archive(root, metadata)['verified'], 1)
                old = archive.load_catalog(root,metadata/'图片来源与哈希.json')['images'][0]
                report = root/'公共管理/02_目录管理/机制核验.json'
                updated = layout.apply_plan(layout.build_plan(root, catalog), catalog, report)
                new = root/updated['entries'][0]['physical_artifact_path']
                (new/'新增中文图.svg').write_text('<svg>new</svg>', encoding='utf-8')
                cache = root/'公共管理/03_工具缓存'; cache.mkdir(parents=True)
                (cache/'不会登记.svg').write_text('<svg>cache</svg>', encoding='utf-8')
                self.assertEqual(archive.archive(root, metadata), {'verified': 2, 'moved': 0})
                images = archive.load_catalog(root,metadata/'图片来源与哈希.json')['images']
                self.assertIn(old, images)
                self.assertEqual(len(list(archive.artifact_images(root))), 2)
            for p in (root/'n14_fixture', root/'last_n14_fixture', root/updated['entries'][0]['directory']): os.rmdir(p)


if __name__ == '__main__': unittest.main()
