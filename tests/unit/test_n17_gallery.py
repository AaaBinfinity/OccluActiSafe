import math
import unittest
from unittest.mock import patch

from occluactisafe.experiments.detector_local_gallery import gallery_predictions,detection_gallery
from occluactisafe.experiments.run_store import read_json,write_json,sha256
from occluactisafe.perception.types import Detection
from test_n17_local_visuals import retained_directory


def prediction(box=(1,1,4,4),score=.8,**extra):
    return {'image_id':'frame','category_id':1,'bbox':list(box),'score':score,**extra}


class GalleryTests(unittest.TestCase):
    def test_zero_width_height_and_point_remain_in_ledger_and_count_as_fp(self):
        predictions=[prediction((1,1,0,3)),prediction((1,1,3,0)),prediction((1,1,0,0)),prediction()]
        detections,ledger,fp=gallery_predictions(predictions,{1:'vehicle'},{'frame'})
        self.assertEqual((len(detections['frame']),len(ledger),fp['frame']),(1,3,3))
        self.assertEqual([v['bbox_xywh'] for v in ledger],[v['bbox'] for v in predictions[:3]])
        self.assertEqual(detections['frame'][0].xyxy,(1,1,5,5))
        with self.assertRaises(ValueError):Detection((1,1,5,1),.8,'vehicle')

    def test_score_boundary_and_low_score_zero_area_are_not_silently_discarded(self):
        _,ledger,fp=gallery_predictions([prediction((0,0,2,0),.25),prediction((0,0,2,0),.24999)],{1:'vehicle'},{'frame'})
        self.assertEqual(len(ledger),2);self.assertEqual(fp['frame'],1)
        self.assertEqual([v['counted_as_fp'] for v in ledger],[True,False])

    def test_corrupt_geometry_and_scores_still_fail(self):
        for box in [(1,2,-1,3),(1,2,3,-1),(1,2,math.nan,3),(1,2,math.inf,3),(1,2,3),(1,2,True,3),(1e308,2,1e308,3)]:
            with self.subTest(box=box),self.assertRaises(ValueError):gallery_predictions([prediction(box)],{1:'vehicle'},{'frame'})
        for score in [math.nan,math.inf,-.1,1.1,True]:
            with self.subTest(score=score),self.assertRaises(ValueError):gallery_predictions([prediction(score=score)],{1:'vehicle'},{'frame'})

    def test_unknown_frame_or_category_still_fails_below_threshold(self):
        for item in [prediction(score=.1,image_id='test_frame'),prediction(score=.1,category_id=9)]:
            with self.assertRaises(ValueError):gallery_predictions([item],{1:'vehicle'},{'frame'})

    def test_floating_point_collapsed_box_is_not_passed_as_valid_detection(self):
        _,ledger,fp=gallery_predictions([prediction((1,1,1e-300,2))],{1:'vehicle'},{'frame'})
        self.assertEqual((len(ledger),fp['frame']),(1,1))

    def fixture(self):
        from PIL import Image
        root=retained_directory();rgb=root/'机制原图.png';Image.new('RGB',(1280,100),(35,40,45)).save(rgb)
        rows=[{'role':'model_val','key':key,'frame_id':key,'domain':'机制','camera_side':'vehicle-side',
               'width':1280,'height':100,'image':str(rgb),'image_sha256':sha256(rgb),
               'gt':[{'xyxy':[1,1,5,5],'class_id':'vehicle'}] if key=='frame' else []} for key in ['frame','empty']]
        cfg={'classes':['vehicle'],'_config_sha256':'mechanism','local_profile':{'model_id':'机制','seed':17,'development_only':True,'source_fingerprint_sha256':'training-source'}}
        (root/'weights').mkdir();(root/'weights/best.pt').write_bytes(b'mechanism metadata only, no model')
        (root/'逐轮验证原值').mkdir();write_json(root/'逐轮验证原值/最佳权重验证.json',{'category_id_map':[1]})
        return root,rows,cfg

    def test_gallery_preserves_frame_denominators_counts_zero_as_fp_and_renders_examples(self):
        root,rows,cfg=self.fixture();preds=[prediction(),prediction((0,0,3,0)),prediction((0,0,0,3),image_id='empty'),prediction((0,0,2,0),score=.1)]
        write_json(root/'predictions.json',preds);before=sha256(root/'predictions.json')
        with patch('occluactisafe.experiments.detector_data.load_index',return_value=(rows,{})):
            result=detection_gallery(cfg,root,maximum=2)
            with self.assertRaises(FileExistsError):detection_gallery(cfg,root,maximum=2)
        self.assertEqual((result['validation_frames'],result['gt'],result['tp'],result['fp'],result['fn']),(2,1,1,2,0))
        self.assertEqual((result['zero_area_total'],result['zero_area_at_threshold'],result['renderable_at_threshold']),(3,2,1))
        self.assertEqual(len(result['zero_area_images']),2)
        self.assertEqual(sha256(root/'predictions.json'),before)
        ledger=read_json(root/'误检漏检与检测样例/零面积预测原始记录.json')['records'];self.assertEqual(len(ledger),3)
        for item in result['images']+result['zero_area_images']:
            self.assertEqual(sha256(root/'误检漏检与检测样例'/item['file']),item['sha256'])

    def test_normal_predictions_keep_original_scores_and_empty_frames(self):
        root,rows,cfg=self.fixture();write_json(root/'predictions.json',[prediction()])
        with patch('occluactisafe.experiments.detector_data.load_index',return_value=(rows,{})):
            r=detection_gallery(cfg,root)
        self.assertEqual((r['validation_frames'],r['tp'],r['fp'],r['fn'],r['zero_area_total']),(2,1,0,0,0))

    def test_changed_source_image_is_rejected_without_fabricated_overlay(self):
        root,rows,cfg=self.fixture();write_json(root/'predictions.json',[prediction()]);rows[0]['image_sha256']='wrong'
        with patch('occluactisafe.experiments.detector_data.load_index',return_value=(rows,{})),self.assertRaisesRegex(ValueError,'SHA'):
            detection_gallery(cfg,root)


if __name__=='__main__':unittest.main()
