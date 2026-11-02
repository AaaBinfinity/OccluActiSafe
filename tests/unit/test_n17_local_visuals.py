import csv
import json
import os
from pathlib import Path
import unittest
import uuid

from occluactisafe.experiments.detector_local_visuals import native_image_name,normalize_native_images,comprehensive_plots,label_diagnostics
from occluactisafe.experiments.run_store import read_json,sha256


def retained_directory():
    root=Path(os.environ.get('OAS_TEST_ARTIFACT_ROOT',r'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较\04_开发与复核\本机入口机制图'))
    if not root.anchor or not Path(root.anchor).exists():raise FileNotFoundError('请配置已接入硬盘OAS_TEST_ARTIFACT_ROOT，测试图必须保留在仓库外')
    path=root/('unit_'+uuid.uuid4().hex);path.mkdir(parents=True)
    (path/'机制样例说明.md').write_text('此目录为构造机制检查，不是真实训练、验证或性能结果。\n',encoding='utf-8')
    return path


class LocalVisualTests(unittest.TestCase):
    def test_reference_image_categories_have_chinese_names(self):
        for name in ('confusion_matrix','confusion_matrix_normalized','BoxF1_curve','P_curve','R_curve','PR_curve',
                     'labels','labels_correlogram','results','train_batch0','val_batch2_labels','val_batch2_pred'):
            self.assertTrue(any('\u4e00'<=c<='\u9fff' for c in native_image_name(name)))
        with self.assertRaises(ValueError):native_image_name('unknown_unreviewed_plot')

    def test_native_jpeg_retained_and_png_has_same_pixels(self):
        from PIL import Image
        root=retained_directory();p=root/'train_batch0.jpg';Image.new('RGB',(31,23),(90,100,110)).save(p)
        with Image.open(p) as im:before=im.convert('RGB').tobytes()
        result=normalize_native_images(root)
        self.assertEqual(len(result),1);self.assertFalse(p.exists())
        jpg=root/result[0]['file'];png=root/result[0]['png_file'];self.assertTrue(jpg.exists())
        with Image.open(png) as im:self.assertEqual(im.tobytes(),before)
        self.assertEqual(sha256(png),result[0]['png_sha256'])

    def test_streamed_step_pages_keep_every_raw_point_and_hash(self):
        root=retained_directory();p=root/'逐步优化指标.csv'
        fields=['step','loss','gradient_norm','step_seconds','gpu_allocated_mib','gpu_reserved_mib','host_rss_mib','learning_rate','loss_components']
        with p.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i in range(1,8):w.writerow(dict(zip(fields,[i,1/i,.5,.01,20,30,40,.001,json.dumps({'box':.1*i,'cls':.2})])))
        result=comprehensive_plots(root,points_per_page=3)
        self.assertEqual(result['points'],7);self.assertEqual(len(result['figures']),24)
        self.assertFalse(result['smoothing']);self.assertTrue(result['all_step_values_plotted'])
        self.assertEqual(result['step_csv_sha256'],sha256(p))
        for item in result['figures']:self.assertEqual(sha256(root/item['file']),item['sha256'])

    def test_label_zero_classes_and_geometry_raw_are_preserved(self):
        import numpy as np
        root=retained_directory();boxes=np.array([[.5,.5,.2,.4],[.25,.5,.1,.2]])
        label_diagnostics(root,boxes,[0,0],{0:'vehicle',1:'pedestrian'})
        raw=read_json(root/'训练标签分类计数.json');self.assertEqual(raw['counts'],[2,0])
        self.assertEqual(raw['zero_gt_classes'],['pedestrian'])
        with np.load(root/'训练标签几何原值.npz') as data:np.testing.assert_equal(data['xywh_normalized'],boxes)
        self.assertEqual(len(read_json(root/'标签图表来源.json')['figures']),4)


if __name__=='__main__':unittest.main()
