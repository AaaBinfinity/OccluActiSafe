"""N17平台期早停与离线重放；共享同一规则，不能把截短训练标为早停。"""
import math

from .run_store import append_json


class PlateauStopper:
    """沿锁定原生strict-improvement/零fitness规则，另加共同最少轮数。"""
    def __init__(self, minimum_epochs, patience, trace=None):
        if type(minimum_epochs) is not int or minimum_epochs < 1 or type(patience) is not int or patience < 1:
            raise ValueError('早停最少轮数与patience必须为正整数')
        self.minimum_epochs, self.patience, self.trace = minimum_epochs, patience, trace
        self.best_fitness, self.best_epoch, self.possible_stop, self.last_epoch = 0.0, 0, False, 0
        self.last_decision = None

    def __call__(self, epoch, fitness):
        if epoch != self.last_epoch+1 or fitness is None or not math.isfinite(float(fitness)):
            raise ValueError('早停输入须为连续epoch和有限验证fitness')
        fitness = float(fitness)
        if fitness > self.best_fitness or self.best_fitness == 0:
            self.best_fitness, self.best_epoch = fitness, epoch
        delta = epoch-self.best_epoch
        self.possible_stop = epoch+1 >= self.minimum_epochs and delta >= self.patience-1
        stop = epoch >= self.minimum_epochs and delta >= self.patience
        self.last_epoch = epoch
        self.last_decision = {'epoch': epoch, 'fitness': fitness, 'best_fitness': self.best_fitness,
                              'best_epoch': self.best_epoch, 'epochs_without_improvement': delta,
                              'minimum_epochs': self.minimum_epochs, 'patience': self.patience,
                              'possible_stop': self.possible_stop, 'stop': stop}
        if self.trace is not None: append_json(self.trace, self.last_decision)
        return stop


def replay_stopping(records, policy, maximum_epochs):
    if not records or len(records) > maximum_epochs:
        raise ValueError('早停记录为空或超过预算')
    stopper = PlateauStopper(policy['minimum_epochs'], policy['patience'])
    for i, record in enumerate(records):
        decision = stopper(record['epoch'], record['fitness'])
        if stopper.last_decision != record:
            raise ValueError('早停记录与冻结规则重放不一致')
        if decision and i != len(records)-1:
            raise ValueError('已触发早停后仍继续训练')
    epochs = len(records)
    if epochs < maximum_epochs and not stopper.last_decision['stop']:
        raise ValueError('未满足早停且未达到完整预算，不能标完成')
    return {'actual_epochs': epochs, 'maximum_epochs': maximum_epochs,
            'stop_reason': 'early_stopping' if stopper.last_decision['stop'] and epochs < maximum_epochs else 'maximum_epochs',
            'best_epoch': stopper.best_epoch, 'best_fitness': stopper.best_fitness,
            'convergence_guaranteed': False}
