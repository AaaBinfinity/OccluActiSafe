import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from occluactisafe.experiments import detector_local as local
from occluactisafe.experiments.detector_terminal import choose_options,execute_options
from occluactisafe.experiments.run_store import read_json


class TerminalTests(unittest.TestCase):
    def options(self,**overrides):
        return local.local_options(local.FAST_OPTIONS,overrides)

    def choose(self,answers,opts=None):
        output=[];stream=iter(answers)
        result=choose_options(opts or self.options(),input_fn=lambda _:next(stream),output=output.append)
        return result,'\n'.join(output)

    def test_select_model_seed_rate_batch_and_training_without_running_engine(self):
        result,text=self.choose(['YOLOv8n','29','0.0003','8','train','y'])
        self.assertEqual((result['MODEL'],result['SEED'],result['LR0'],result['BATCH'],result['ACTION']),
                         ('YOLOv8n',29,.0003,8,'train'))
        self.assertIn('最多300轮',text)
        self.assertIn('0.0003',text)
        self.assertIn('早停耐心：30',text)

    def test_numbered_model_and_action_accept_safe_defaults(self):
        result,_=self.choose(['2','','','','','y'])
        self.assertEqual((result['MODEL'],result['ACTION']),('YOLOv8n','check'))

    def test_size_change_recomputes_batch_and_explicit_override_wins(self):
        result,_=self.choose(['YOLO11m','43','3e-4','','plan','yes'])
        self.assertEqual((result['MODEL'],result['BATCH'],result['LR0']),('YOLO11m',2,.0003))
        opts=local.local_options(local.FAST_OPTIONS,{'MODEL':'YOLO26x','BATCH':3})
        self.assertEqual(opts['BATCH'],3)
        legacy=local.local_options(overrides={'MODEL':'YOLO11m','BATCH':2})
        self.assertEqual(legacy['BATCH'],2)

    def test_invalid_fields_reprompt_without_starting_partial_task(self):
        answers=['0','unknown','YOLOv8n','-1','4294967296','17','nan','inf','0','0.0003',
                 '0','8','99','plan','y']
        result,text=self.choose(answers)
        self.assertEqual(result['ACTION'],'plan')
        self.assertGreaterEqual(text.count('输入无效'),8)

    def test_empty_confirmation_and_quit_cancel(self):
        result,text=self.choose(['','','','','train',''])
        self.assertIsNone(result);self.assertIn('未启动预检或训练',text)
        self.assertIsNone(self.choose(['q'])[0])

    def test_eof_and_keyboard_interrupt_cancel_without_execution(self):
        for error in (EOFError,KeyboardInterrupt):
            with self.subTest(error=error):
                def reader(_):raise error
                self.assertIsNone(choose_options(self.options(),input_fn=reader,output=lambda _:None))

    def test_old_file_training_default_does_not_automatically_train_in_menu(self):
        result,_=self.choose(['','','','','','y'],self.options(ACTION='train'))
        self.assertEqual(result['ACTION'],'check')

    def test_same_captured_bytes_are_used_if_defaults_file_changes_mid_menu(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'defaults.py';original=local.FAST_OPTIONS.read_bytes()
            p.write_bytes(original);opts=local.local_options(p)
            answers=iter(['YOLOv8n','','','','plan','y']);first=True
            def reader(_):
                nonlocal first
                if first:
                    first=False;p.write_bytes(original+b'\nLR0 = 0.1\n')
                return next(answers)
            result=choose_options(opts,input_fn=reader,output=lambda _:None)
            self.assertEqual(result['LR0'],.001)
            self.assertEqual(result['options_source_bytes'],original)
            self.assertEqual(result['options_sha256'],hashlib.sha256(original).hexdigest())

    def test_lr_reaches_training_and_recipe_hash_not_only_user_interface(self):
        one=self.options(LR0=.001);two=self.options(LR0=.0003)
        env={'font_path':'fixture','source':{'source_fingerprint_sha256':'fixture'},
             'input_verification':{'source_index_sha256':'fixture'}}
        snapshot={'index_sha256':'fixture'}
        a=local.build_local_config(one,env,snapshot,Path('unused'))
        b=local.build_local_config(two,env,snapshot,Path('unused'))
        self.assertEqual((a['training']['lr0'],b['training']['lr0']),(.001,.0003))
        self.assertNotEqual(a['local_profile']['recipe_sha256'],b['local_profile']['recipe_sha256'])
        self.assertEqual(a['gates'],b['gates']);self.assertEqual(a['evaluation'],b['evaluation'])
        for value in (0,-1,True,float('nan'),float('inf'),2):
            with self.assertRaises(ValueError):self.options(LR0=value)

    def test_resume_prompts_for_real_task_directory_before_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'本机训练任务.json').write_text('{}',encoding='utf-8')
            result,_=self.choose(['','','','','resume',str(p),'y'])
            self.assertEqual(result['RESUME_RUN'],p.resolve())

    def test_cli_dry_run_never_calls_training_or_preflight(self):
        args=['train_n17_local.py','--action','train','--model','YOLOv8m','--seed','43',
              '--lr0','0.0003','--dry-run']
        with patch.object(sys,'argv',args),patch.object(local,'run_local',side_effect=AssertionError('must not execute')):
            result=local.main()
        self.assertEqual((result['MODEL'],result['SEED'],result['LR0'],result['BATCH']),('YOLOv8m',43,.0003,2))

    def test_internal_preflight_entry_never_opens_interactive_menu(self):
        with patch.object(sys,'argv',['train_n17_local.py','--preflight-task','fixture.json']),patch.object(local,'preflight_worker') as worker:
            local.main();worker.assert_called_once_with(Path('fixture.json'))

    def test_confirmed_selection_is_saved_without_mutating_default_file(self):
        with tempfile.TemporaryDirectory() as temp:
            opts=self.options(OUTPUT_ROOT=str(Path(temp)/'results'),LR0=.0003,ACTION='plan')
            before=hashlib.sha256(local.FAST_OPTIONS.read_bytes()).hexdigest()
            result=execute_options(opts,lambda _:Path(temp)/'planned','interactive')
            records=list((Path(temp)/'results').rglob('终端任务选择.json'))
            self.assertEqual(len(records),1)
            r=read_json(records[0]);self.assertEqual(r['status'],'completed')
            self.assertEqual(r['effective_options']['LR0'],.0003)
            self.assertEqual(r['effective_options']['ACTION'],'plan')
            self.assertEqual(hashlib.sha256(local.FAST_OPTIONS.read_bytes()).hexdigest(),before)

    def test_engine_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as temp:
            opts=self.options(OUTPUT_ROOT=str(Path(temp)/'results'))
            def fail(_):raise RuntimeError('mechanism failure')
            with self.assertRaisesRegex(RuntimeError,'mechanism failure'):execute_options(opts,fail,'cli')
            r=read_json(next((Path(temp)/'results').rglob('终端任务选择.json')))
            self.assertEqual(r['status'],'failed');self.assertIn('mechanism failure',r['error'])

    def test_no_arguments_cancel_never_runs_configured_training(self):
        with patch.object(sys,'argv',['train_n17_local.py']),patch('builtins.input',side_effect=EOFError),patch.object(local,'run_local') as runner:
            self.assertIsNone(local.main());runner.assert_not_called()


if __name__=='__main__':unittest.main()
