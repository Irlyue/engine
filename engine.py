import os
import json
import torch
import visdom
import warnings
import torch.nn as nn

from tqdm import tqdm
from . import logger
from time import time


TRAIN_LOG_FILE = 'train_log.txt'
EVAL_LOG_FILE = 'eval_log.txt'
LOGGER = None
VIZ = None


def setup_visdom(port=8097, env='engine', **kwargs):
    global VIZ
    try:
        VIZ = visdom.Visdom(port=port, env=env, **kwargs)
    except:
        VIZ = None
        log_print('==> Failed to set up visdom!')
    else:
        log_print('==> Successfully set up visdom!')


def setup_logger(name):
    global LOGGER
    if LOGGER:
        LOGGER.close()
    folder = os.path.dirname(name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    LOGGER = logger.TerminalFileLogger(name)
    LOGGER.print(f'==> Successfully set up logging file `{name}`')


def log_print(s):
    LOGGER.print(s)


class Engine:
    SAVE_PATH = 'model.pt'

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.multi_gpu_model = None

    def __call__(self, x):
        return self.multi_gpu_model(x) if self.multi_gpu_model else self.model(x)

    def train(self, data, optimizer, loss_fn, hooks=None, progress_bar=True):
        """
        :param data: iterator, gives the training batch (input, label)
        :param optimizer: torch.optim.optimizer,
        :param loss_fn: callable, loss function for training. Must return a scalar. It will
        be used as
                    loss = loss_fn(model(input), label)
        :param hooks: list,
        :param progress_bar: bool, set it to False to disable progress bar.
        """
        config = self.config
        self.on_start_train(hooks=hooks, optimizer=optimizer)
        for self.st['epoch'] in range(self.st['epoch'], config['n_epochs'] + self.st['epoch']):
            self.on_start_epoch()
            data = tqdm(data) if progress_bar else data
            for self.st['step'], (inputs, labels) in enumerate(data, start=self.st['step']):
                self.on_start_batch(inputs=inputs, labels=labels)
                optimizer.zero_grad()
                logits = self(inputs)
                loss = loss_fn(logits, labels)
                self.st['loss'] = loss.item()
                loss.backward()
                optimizer.step()
                self.on_end_batch()
            self.on_end_epoch()
        self.on_end_train()

    def evaluate(self, data, metrics, hooks=None, model_path='model-best.pt'):
        with torch.no_grad():
            self.on_start_eval(hooks=hooks, metrics=metrics, model_path=model_path)
            for inputs, labels in tqdm(data):
                logits = self.model(inputs)
                for metric in metrics:
                    metric(labels, logits)
            self.on_end_eval()

    def on_start_predict(self, hooks=None, model_path='model-best.pt'):
        self.on_start()
        self.model.eval()
        self.st = {}
        if model_path:
            self.load_model(model_path)
        else:
            warnings.warn('Not loading any checkpoint file! Could be an error!')
        for hook in hooks or []:
            hook.on_start_eval(self)

    def save_model(self, path, **others):
        path = path if os.path.isabs(path) else os.path.join(self.config['model_dir'], path)
        to_save = {
            'step': self.st['step'],
            'epoch': self.st['epoch'],
            'state_dict': self.model.state_dict()
        }
        to_save.update(others)
        torch.save(to_save, path)
        log_print('===> Model-%d saved at %s' % (to_save['step'], path))

    def load_model(self, path):
        path = self.model_path(path)
        loaded = torch.load(path)
        self.model.load_state_dict(loaded.pop('state_dict'))
        self.st.update(loaded)
        log_print('===> Model at %d step loaded!' % self.st['step'])

    def on_start_eval(self, **kwargs):
        self.model.eval()
        self.model.to(self.config['device'])
        self.st = kwargs.copy()
        for metric in self.st['metrics']:
            metric.reset()
        self.load_model(self.st['model_path'])
        for hook in self.st['hooks'] or []:
            hook.on_start_eval(self)

    def on_end_eval(self):
        log_print('***********************************')
        for metric in self.st['metrics']:
            log_print(metric)
        log_print('***********************************')

    def on_start(self):
        n_gpus = torch.cuda.device_count()
        if self.config['multi_gpus'] and n_gpus > 1:
            log_print('==> Using %d GPUs...' % n_gpus)
            self.multi_gpu_model = nn.DataParallel(self.model)
            self.multi_gpu_model.to(self.config['device'])
        else:
            self.model.to(self.config['device'])

    def on_start_train(self, **kwargs):
        self.on_start()
        self.model.train()
        self.st = {
            'epoch': 0,
            'step': 0,
            'current_value': 0.0,
            'best_value': -float('Inf'),
            'loss': 0.0
        }
        self.st.update(kwargs)

        if os.path.exists(self.model_path(self.SAVE_PATH)):
            self.load_model(self.SAVE_PATH)
        if not os.path.exists(self.config['model_dir']):
            os.makedirs(self.config['model_dir'])
        log_print('\n%s\n' % json.dumps(self.config, indent=2))
        for hook in self.st['hooks'] or []:
            hook.on_start_train(self)

    def model_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.config['model_dir'], path)

    def absolute_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.config['model_dir'], path)

    def on_end_train(self):
        log_print('Loss at final step: %.4f' % self.st['loss'])
        for hook in self.st['hooks'] or []:
            hook.on_end_train(self)

    def on_start_epoch(self):
        pass

    def on_end_epoch(self):
        for hook in self.st['hooks'] or []:
            hook.on_end_epoch(self)

    def on_start_batch(self, **kwargs):
        self.st.update(kwargs)

    def on_end_batch(self):
        for hook in self.st['hooks'] or []:
            hook.on_end_batch(self)


class Hook:
    def on_start_epoch(self):
        pass

    def on_end_epoch(self, engine):
        pass

    def on_start_batch(self):
        pass

    def on_end_batch(self, engine):
        pass

    def on_start_train(self, engine):
        pass

    def on_end_train(self, engine):
        pass

    def on_start_eval(self, engine):
        pass


class FrequencyHook(Hook):
    def __init__(self, every_steps=None, every_epochs=None):
        assert not (every_steps and every_epochs), 'Provide only one of them!'
        self.every_steps = every_steps
        self.every_epochs = every_epochs

    def do_it(self, g):
        raise NotImplementedError

    def on_end_batch(self, engine):
        if self.every_steps and engine.st['step'] % self.every_steps == 0:
            self.do_it(engine)

    def on_end_epoch(self, engine):
        if self.every_epochs and engine.st['epoch'] % self.every_epochs == 0:
            self.do_it(engine)


class ScalarSummaryHook(FrequencyHook):
    VIS_WIN_NAME = 'loss'

    def __init__(self, summaries, summary_every_steps=None, summary_every_epochs=None):
        super().__init__(summary_every_steps, summary_every_epochs)
        self.summaries = summaries

    def send_to_visdom(self, g):
        if VIZ:
            for xname, yname in self.summaries:
                x = [g.st[xname]]
                y = [g.st[yname]]
                VIZ.line(Y=y, X=x, win=yname, update='append',
                         opts={
                             'xlabel': xname,
                             'ylabel': yname,
                         })

    def do_it(self, g: Engine):
        self.send_to_visdom(g)


class LoggingHook(FrequencyHook):
    def __init__(self, keys, print_every_steps=None, print_every_epochs=None):
        super().__init__(print_every_steps, print_every_epochs)
        self.keys = keys
        self.tic = time()

    def do_it(self, g: Engine):
        toc = time()
        elapsed = toc - self.tic
        info = ', '.join('{}={}'.format(key, g.st[key]) for key in self.keys)
        log_print('{} [{:.1f} seconds]'.format(info, elapsed))
        self.tic = toc


class SaveBestModelHook(FrequencyHook):
    SAVE_PATH = 'model-best.pt'

    def __init__(self, metric, data, save_every_steps=None, save_every_epochs=1):
        super().__init__(save_every_steps, save_every_epochs)
        self.metric = metric
        self.data = data
        self.best_acc = -float('Inf')

    def do_it(self, g: Engine):
        g.model.eval()
        metric = self.metric
        metric.reset()
        with torch.no_grad():
            for inputs, labels in tqdm(self.data):
                pred = g(inputs)
                metric(labels, pred)
        if self.best_acc < metric.result:
            log_print('===> Aha, %s' % self.metric)
            self.best_acc = metric.result
            g.save_model(self.SAVE_PATH)
        self.send_to_visdom(g)
        g.model.train()

    def on_end_train(self, g: Engine):
        self.do_it(g)

    def send_to_visdom(self, g: Engine):
        if VIZ:
            VIZ.line(Y=[self.metric.result], X=[g.st['step']], win=self.metric.name, update='append',
                     opts={
                         'title': self.metric.name,
                         'xlabel': 'Step',
                         'ylabel': self.metric.name
                     })


class ExponentialMovingAverageHook(Hook):
    def __init__(self, params, beta=0.99, device=None, save_every_steps=None,
                 save_every_epochs=None):
        assert not (save_every_epochs and save_every_steps), 'Provide only one of them.'
        self.params = [item for item in params]
        self.beta = beta
        self.device = device
        self.save_every_steps = save_every_steps
        self.save_every_epochs = save_every_epochs
        self.ema = [item.detach().clone() for item in self.params]
        self.ema = [item.to(self.device) for item in self.ema] if self.device else self.ema

    def on_end_batch(self, engine):
        with torch.no_grad():
            for left, right in zip(self.ema, self.params):
                left *= self.beta
                left += (1 - self.beta) * right
        if self.save_every_steps and engine.st['step'] % self.save_every_steps == 0:
            engine.save_model(engine.SAVE_PATH, ema=self.ema)

    def on_end_epoch(self, engine):
        if self.save_every_epochs and engine.st['epoch'] % self.save_every_epochs == 0:
            engine.save_model(engine.SAVE_PATH, ema=self.ema)


class ApplyEMAHook(Hook):
    def __init__(self, params=None, save_path='model.pt'):
        self.save_path = save_path
        self.params = params

    def on_start_eval(self, engine):
        log_print('==> Applying EMA weights')
        state = torch.load(engine.model_path(self.save_path))
        with torch.no_grad():
            for left, right in zip(self.params or engine.model.parameters(), state['ema']):
                left[:] = right


class SaveModelHook(FrequencyHook):
    def __init__(self, save_every_steps=None, save_every_epochs=None):
        super().__init__(save_every_steps, save_every_epochs)

    def do_it(self, g: Engine):
        g.save_model(g.SAVE_PATH)

    def on_end_train(self, g):
        # remember to save the last checkpoint file
        self.do_it(g)
