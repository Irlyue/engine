import os
import json
import torch
import torch.nn as nn

from tqdm import tqdm


class Engine:
    SAVE_PATH = 'model.pt'

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.multi_gpu_model = None

    def __call__(self, x):
        return self.model(x)

    def train(self, data, optimizer, loss_fn, hooks=None, progress_bar=True):
        config = self.config
        self.on_start_train(hooks=hooks, optimizer=optimizer)
        for self.st['epoch'] in range(self.st['epoch'], config['n_epochs'] + self.st['epoch']):
            self.on_start_epoch()
            data = tqdm(data) if progress_bar else data
            for self.st['step'], (inputs, labels) in enumerate(data, start=self.st['step']):
                self.on_start_batch(inputs=inputs, labels=labels)
                optimizer.zero_grad()
                logits = self.multi_gpu_model(inputs) if self.multi_gpu_model else self.model(inputs)
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
        self.model.eval()
        self.model.to(self.config['device'])
        self.st = {}
        self.load_model(model_path)
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
        print('===>Model-%d saved at %s' % (to_save['step'], path))

    def load_model(self, path):
        path = self.model_path(path)
        loaded = torch.load(path)
        self.model.load_state_dict(loaded.pop('state_dict'))
        self.st.update(loaded)
        print('===>Model at %d step loaded!' % self.st['step'])

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
        print('***********************************')
        for metric in self.st['metrics']:
            print(metric)
        print('***********************************')

    def on_start_train(self, **kwargs):
        self.model.train()
        self.st = {
            'epoch': 0,
            'step': 0,
            'current_value': 0.0,
            'best_value': -float('Inf'),
            'loss': 0.0
        }
        self.st.update(kwargs)

        n_gpus = torch.cuda.device_count()
        if self.config['multi_gpus'] and n_gpus > 1:
            print('==> Using %d GPUs...' % n_gpus)
            self.multi_gpu_model = nn.DataParallel(self.model)
            self.multi_gpu_model.to(self.config['device'])
        else:
            self.model.to(self.config['device'])

        if os.path.exists(self.model_path(self.SAVE_PATH)):
            self.load_model(self.SAVE_PATH)
        if not os.path.exists(self.config['model_dir']):
            os.makedirs(self.config['model_dir'])
        with open(os.path.join(self.config['model_dir'], 'config.json'), 'wt') as f:
            json.dump(self.config, f, indent=2)
        for hook in self.st['hooks'] or []:
            hook.on_start_train(self)

    def model_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.config['model_dir'], path)

    def on_end_train(self):
        print('Loss at final step: %.4f' % self.st['loss'])
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


class LoggingHook(Hook):
    def __init__(self, keys, print_every_steps=None, print_every_epochs=None):
        assert not (print_every_steps and print_every_epochs), 'Provide only one of them!'
        self.keys = keys
        self.print_every_steps = print_every_steps
        self.print_every_epochs = print_every_epochs

    def on_end_batch(self, engine):
        if self.print_every_steps and engine.st['step'] % self.print_every_steps == 0:
            print(', '.join('{}={}'.format(key, engine.st[key]) for key in self.keys))

    def on_end_epoch(self, engine):
        if self.print_every_epochs and engine.st['epoch'] % self.print_every_epochs == 0:
            print(', '.join('{}={}'.format(key, engine.st[key]) for key in self.keys))


class SaveBestModelHook(Hook):
    SAVE_PATH = 'model-best.pt'

    def __init__(self, metric, data):
        self.metric = metric
        self.data = data
        self.best_acc = -float('Inf')

    def on_end_epoch(self, engine):
        engine.model.eval()
        metric = self.metric
        metric.reset()
        with torch.no_grad():
            for inputs, labels in tqdm(self.data):
                pred = engine.model(inputs)
                metric(labels, pred)
        if self.best_acc < metric.result:
            print('===>Aha, %s' % self.metric)
            self.best_acc = metric.result
            engine.save_model(self.SAVE_PATH)
        engine.model.train()


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
        print('==> Applying EMA weights')
        state = torch.load(engine.model_path(self.save_path))
        with torch.no_grad():
            for left, right in zip(self.params or engine.model.parameters(), state['ema']):
                left[:] = right


class SaveModelHook(Hook):
    def __init__(self, save_every_epochs=None, save_every_steps=None):
        assert not (save_every_epochs and save_every_steps), 'Provide only one of them.'
        self.save_every_epochs = save_every_epochs
        self.save_every_steps = save_every_steps

    def on_end_epoch(self, engine):
        if self.save_every_epochs and engine.st['epoch'] % self.save_every_epochs == 0:
            engine.save_model(engine.SAVE_PATH)

    def on_end_batch(self, engine):
        if self.save_every_steps and engine.st['step'] % self.save_every_steps == 0:
            engine.save_model(engine.SAVE_PATH)

    def on_end_train(self, engine):
        # remember to save the last checkpoint file
        engine.save_model(engine.SAVE_PATH)
