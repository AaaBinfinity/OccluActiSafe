"""串行队列机制；夹具结果不代表GPU训练、模型精度或资源通过。"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import yaml
from unittest.mock import patch

from occluactisafe.experiments import detector_seed_queue as queue
from occluactisafe.experiments.run_store import read_json, write_json, sha256


SOURCE = {'source_commit': '1' * 40, 'source_fingerprint_sha256': 'mechanism-only'}


class SeedQueueTests(unittest.TestCase):
    def options(self, **values):
        return queue.queue_options(overrides=values)

    def fixture(self, directory, action='train'):
        options = self.options(OUTPUT_ROOT=str(Path(directory) / 'output'), ACTION=action)
        with patch.object(queue, 'source_info', return_value=SOURCE), patch.object(queue, 'identity', return_value={}):
            root = queue.create_queue(options)
        state = read_json(root / '队列状态.json')
        state['status'] = 'launched'
        write_json(root / '队列状态.json', state)
        return root

    def run_fixture(self, root, runner):
        with patch.object(queue, 'source_info', return_value=SOURCE):
            return queue.run_queue(root, runner=runner, wait=lambda *_: None)

    def receipt(self, root, entry, *, recipe='common', status='completed'):
        write_json(root / entry['id'] / '单项结果.json', {'status': status, 'recipe_sha256': recipe,
                   'scope': 'mechanism_only_no_training'})

    def test_default_model_order_and_unchanged_training_defaults(self):
        value = self.options()
        self.assertEqual((value['local']['MODEL'], value['seeds']), ('YOLOv8n', (17, 29, 43)))
        self.assertEqual([value['local'][k] for k in ('ACTION', 'MAX_EPOCHS', 'MIN_EPOCHS', 'PATIENCE', 'BATCH', 'WORKERS', 'LR0')],
                         ['check', 300, 15, 10, 32, 4, .001])
        self.assertEqual((value['local']['PREFETCH_FACTOR'],value['local']['VALIDATION_WORKERS']),(2,2))

    def test_only_ordered_unique_prescribed_seed_subsets_are_allowed(self):
        self.assertEqual(self.options(SEEDS=[29, 43])['seeds'], (29, 43))
        for seeds in ([], [17, 17], [43, 29], [1], [True], [17, 29, 42]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                self.options(SEEDS=seeds)

    def test_illegal_action_wait_budget_and_resource_values_are_rejected(self):
        for values in ({'ACTION': 'resume'}, {'ACTION': 'probe'}, {'WAIT_HOURS': 0},
                       {'WAIT_HOURS': float('nan')}, {'WAIT_HOURS': 169}, {'BATCH': 0}, {'LR0': float('inf')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.options(**values)

    def test_menu_selects_once_and_never_requests_individual_seeds(self):
        messages = []
        answers = iter(['2', '0.0003', '16', '3', 'y'])
        opts = queue.choose_queue(self.options(), input_fn=lambda _: next(answers), output=messages.append)
        self.assertEqual((opts['local']['MODEL'], opts['local']['LR0'], opts['local']['BATCH'], opts['local']['ACTION']),
                         ('YOLOv8n', .0003, 16, 'train'))
        self.assertEqual(opts['seeds'], (17, 29, 43))
        self.assertIn('旧seed17', '\n'.join(messages))

    def test_menu_reprompts_invalid_values_and_defaults_to_check(self):
        answers = iter(['0', 'YOLO11m', 'nan', '', '0', '', '4', '', 'y'])
        opts = queue.choose_queue(self.options(ACTION='train'), input_fn=lambda _: next(answers), output=lambda _: None)
        self.assertEqual((opts['local']['MODEL'], opts['local']['BATCH'], opts['local']['ACTION']), ('YOLO11m', 2, 'check'))

    def test_menu_cancel_eof_and_interrupt_never_create_queue(self):
        for answers in (['q'], ['', '', '', 'train', '']):
            stream = iter(answers)
            self.assertIsNone(queue.choose_queue(self.options(), input_fn=lambda _: next(stream), output=lambda _: None))
        for error in (EOFError, KeyboardInterrupt):
            def reader(_):
                raise error
            self.assertIsNone(queue.choose_queue(self.options(), input_fn=reader, output=lambda _: None))

    def test_all_seed_tasks_use_captured_default_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            plan = read_json(root / '队列计划.json')
            opts = [queue.task_options(root, plan, s) for s in (17, 29, 43)]
            self.assertEqual([o['SEED'] for o in opts], [17, 29, 43])
            self.assertEqual(len({o['options_sha256'] for o in opts}), 1)
            for key in queue.OPTION_KEYS:
                if key != 'SEED':
                    self.assertEqual(opts[0][key], opts[1][key])
            self.assertEqual(plan['training_exposure_upper_bound'], 3 * 17003 * 300)

    def test_sequential_success_and_normal_exit_after_last_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            calls = []
            def runner(directory, entry):
                state = read_json(directory / '队列状态.json')
                self.assertEqual(sum(t['status'] == 'running' for t in state['tasks']), 1)
                self.assertEqual(sum(t['status'] == 'completed' for t in state['tasks']), len(calls))
                calls.append(entry['seed'])
                self.receipt(directory, entry)
                return 0
            state = self.run_fixture(root, runner)
            self.assertEqual(calls, [17, 29, 43])
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['counts'], {'completed': 3})
            self.assertFalse(state['full_comparison_complete'])

    def test_failure_continues_to_next_seed_without_retry_or_false_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            calls = []
            def runner(directory, entry):
                calls.append(entry['seed'])
                self.receipt(directory, entry, status='failed' if entry['seed'] == 29 else 'completed')
                return 7 if entry['seed'] == 29 else 0
            state = self.run_fixture(root, runner)
            self.assertEqual(calls, [17, 29, 43])
            self.assertEqual(state['status'], 'completed_with_failures')
            self.assertEqual(state['counts'], {'completed': 2, 'failed': 1})
            self.assertEqual(state['tasks'][1]['worker_returncode'], 7)

    def test_missing_receipt_and_recipe_mismatch_are_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            def runner(directory, entry):
                if entry['seed'] != 29:
                    self.receipt(directory, entry, recipe=str(entry['seed']))
                return 0
            state = self.run_fixture(root, runner)
            self.assertEqual(state['counts'], {'completed': 1, 'failed': 2})

    def test_completed_queue_cannot_be_reexecuted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            def runner(directory, entry):
                self.receipt(directory, entry)
                return 0
            self.run_fixture(root, runner)
            with self.assertRaisesRegex(ValueError, '拒绝重跑'):
                self.run_fixture(root, runner)

    def test_plan_does_not_launch_worker_and_cannot_train_via_host(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp, 'plan')
            with self.assertRaises(ValueError), patch.object(queue, 'launch_worker') as launch:
                queue.run_queue(root)
            launch.assert_not_called()

    def test_changed_source_blocks_remaining_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            with patch.object(queue, 'source_info', return_value={**SOURCE, 'source_fingerprint_sha256': 'changed'}), self.assertRaises(ValueError):
                queue.run_queue(root, runner=lambda *_: self.fail('must not train'), wait=lambda *_: None)
            state = read_json(root / '队列状态.json')
            self.assertEqual(state['status'], 'blocked')
            self.assertTrue(all(t['status'] == 'pending' for t in state['tasks']))

    def test_changed_plan_or_snapshot_is_rejected(self):
        for filename in ('队列计划.json', '冻结本机选项.py', '冻结队列选项.py'):
            with self.subTest(file=filename), tempfile.TemporaryDirectory() as temp:
                root = self.fixture(temp)
                with (root / filename).open('ab') as stream:
                    stream.write(b'\n')
                with self.assertRaises(ValueError):
                    queue.load_queue(root)

    def test_mutated_state_cannot_remove_or_reorder_frozen_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            state = read_json(root / '队列状态.json')
            state['tasks'].reverse()
            write_json(root / '队列状态.json', state)
            with self.assertRaisesRegex(ValueError, '任务身份/顺序不同'):
                queue.load_queue(root)

    def test_changed_input_identity_is_rejected_without_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'input.txt'
            file.write_text('old')
            plan = {'source': SOURCE, 'inputs': {str(file): sha256(file)}}
            file.write_text('new')
            with patch.object(queue, 'source_info', return_value=SOURCE), self.assertRaises(ValueError):
                queue.verify_identity(plan)

    def test_busy_gpu_wait_times_out_without_touching_existing_process(self):
        plan = {'effective_options': {'OUTPUT_ROOT': 'unused', 'DEVICE': 0}}
        with (patch.object(queue, 'verify_identity'),
              patch.object(queue, 'execution_lock', side_effect=RuntimeError('busy')),
              patch.object(queue.time, 'monotonic', return_value=10)):
            with self.assertRaises(TimeoutError):
                queue.wait_gpu(plan, 9)

    def test_worker_refuses_second_execution_even_after_failed_first_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            name = 'YOLOv8n_seed17'
            with patch.object(queue, 'verify_identity'), patch.object(queue, 'run_local', side_effect=RuntimeError('mechanism failure')) as run:
                with self.assertRaises(RuntimeError):
                    queue.worker(root, name)
                with self.assertRaises(FileExistsError):
                    queue.worker(root, name)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(read_json(root / name / '单项结果.json')['status'], 'failed')

    def test_background_creation_is_detached_and_has_no_stdin_or_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            state = read_json(root / '队列状态.json')
            state['status'] = 'planned'
            write_json(root / '队列状态.json', state)
            with patch.object(queue.subprocess, 'Popen') as start:
                start.return_value.pid = 12345
                self.assertEqual(queue.start_background(root), 12345)
            self.assertEqual(start.call_args.kwargs['stdin'], subprocess.DEVNULL)
            self.assertEqual(start.call_args.kwargs['creationflags'], subprocess.CREATE_NO_WINDOW)
            with self.assertRaises(ValueError):
                queue.start_background(root)

    def test_cli_dry_run_never_creates_or_launches_queue(self):
        with patch.object(queue, 'create_queue') as create, patch.object(queue, 'start_background') as start:
            value = queue.main(['--model', 'YOLOv8n', '--seeds', '29', '43', '--action', 'train', '--dry-run'])
        self.assertEqual(value['seeds'], (29, 43))
        create.assert_not_called()
        start.assert_not_called()

    def test_actual_entry_from_foreign_cwd_is_lightweight_and_dry(self):
        result = subprocess.run([sys.executable, '-B', str(queue.ENTRY), '--model', 'YOLOv8n',
                                 '--action', 'train', '--dry-run'], cwd=Path.home(), capture_output=True,
                                encoding='utf-8', env={**queue.os.environ, 'PYTHONIOENCODING': 'utf-8'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('seeds=(17, 29, 43)', result.stdout)
        self.assertIn('未启动检查', result.stdout)

    def test_actual_detached_host_has_no_console_and_refuses_plan_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp, 'plan')
            with (root / 'mechanism.log').open('wb') as log:
                code = subprocess.run([sys.executable, '-B', str(queue.ENTRY), '--host', str(root)],
                                      stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW).returncode
            self.assertNotEqual(code, 0)
            self.assertEqual(read_json(root / '后台宿主身份.json')['console_handle'], 0)
            self.assertEqual(read_json(root / '队列状态.json')['status'], 'blocked')
            self.assertFalse((root / 'YOLOv8n_seed17').exists())

    def test_training_launcher_keeps_cuda_mapping_and_isolates_console(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / 'task.json'
            write_json(task, {'worker_cuda_visible_devices': 'GPU-fixture'})
            with patch.object(queue.subprocess, 'Popen') as start:
                start.return_value.wait.return_value = 8
                code = queue.execute_training_worker(root, {'id': 'seed17'}, task, root / 'console.txt')
            self.assertEqual(code, 8)
            self.assertEqual(start.call_args.kwargs['creationflags'], subprocess.CREATE_NO_WINDOW)
            self.assertEqual(start.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-fixture')
            self.assertIn('--model-task', start.call_args.args[0])

    def test_completed_result_requires_success_audit_and_same_task_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            opts = self.options(OUTPUT_ROOT=str(root / 'output'))['local']
            job = opts['OUTPUT_ROOT'] / 'mechanism'
            frozen = job / 'config.yaml'
            job.mkdir(parents=True)
            frozen.write_text(yaml.safe_dump({'local_profile': {'recipe_sha256': 'common'}}))
            audit = {'status': 'passed', 'actual_epochs': 15, 'image_exposures': 3, 'weights': {}}
            task = {'status': 'completed', 'development_only': False, 'model': 'YOLOv8n', 'seed': 17,
                    'source_fingerprint_sha256': SOURCE['source_fingerprint_sha256'],
                    'frozen_config': str(frozen), 'frozen_config_sha256': sha256(frozen), 'audit': audit}
            write_json(job / '本机训练任务.json', task)
            write_json(job / '本次训练核验.json', audit)
            self.assertEqual(queue.completed_result(job, opts, {'source': SOURCE})['recipe_sha256'], 'common')
            for changed in ({'status': 'failed'}, {'seed': 29}, {'development_only': True}, {'frozen_config_sha256': 'bad'}):
                write_json(job / '本机训练任务.json', {**task, **changed})
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    queue.completed_result(job, opts, {'source': SOURCE})

    def test_failed_or_pending_metrics_are_blank_in_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue.result_summary(root, {'tasks': [{'seed': 17, 'status': 'failed', 'error': 'mechanism'},
                                                  {'seed': 29, 'status': 'pending'}]})
            import csv
            with (root / '队列结果汇总.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['actual_epochs'], '')
            self.assertEqual(rows[1]['image_exposures'], '')


if __name__ == '__main__':
    unittest.main()
