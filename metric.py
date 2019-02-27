import torch
import numpy as np
import torch.nn as nn


class Metric:
    def __init__(self, name):
        self.name = name

    @property
    def result(self):
        raise NotImplementedError

    def __repr__(self):
        return '{}={}'.format(self.name, self.result)

    def reset(self):
        raise NotImplementedError

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AccuracyMetric(Metric):
    def __init__(self, func=None):
        super().__init__('Accuracy')
        self.func = func
        self.reset()

    def reset(self):
        self.total = 0
        self.correct = 0

    def __call__(self, gt, pred):
        gt, pred = (gt, pred) if self.func is None else self.func(gt, pred)
        self.correct += gt.eq(pred).sum().item()
        self.total += gt.numel()

    @property
    def result(self):
        return self.correct / self.total


class CrossEntropyLossMetric(Metric):
    def __init__(self, func=None):
        super().__init__('CELoss')
        self.func = func
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.count = 0

    def __call__(self, gt, pred):
        gt, pred = (gt, pred) if self.func is None else self.func(gt, pred)
        self.total_loss += self.loss_fn(pred, gt)
        self.count += 1

    @property
    def result(self):
        return self.total_loss / self.count


class MeanIoUMetric(Metric):
    def __init__(self, n_classes, func=None, ignore_pixel=255):
        super().__init__('mIoU')
        self.n_classes = n_classes
        self.func = func
        self.ignore_pixel = ignore_pixel
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def __call__(self, gt, pred):
        gt, pred = (gt, pred) if self.func is None else self.func(gt, pred)
        self.cm += self._confusion_matrix(gt, pred)

    def _confusion_matrix(self, gt, pred):
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        gt, pred = [item.astype(np.int64) for item in [gt, pred]]
        flag = (gt != self.ignore_pixel)
        counts = gt[flag] + pred[flag] * self.n_classes
        cm = np.histogram(counts.flatten(), bins=range(self.n_classes**2 + 1))[0]
        cm = cm.reshape((self.n_classes, self.n_classes)).T
        return cm

    @property
    def result(self):
        tp = self.cm.diagonal()
        total = self.cm.sum(axis=1) + self.cm.sum(axis=0) - tp
        self.ious = tp * 1.0 / total
        return self.ious.mean()

    def __repr__(self):
        return '{}={}'.format(self.name, self.result)


class MeanAPMetric(Metric):
    def __init__(self, func=None):
        super().__init__('mAP')
        self.func = func
        self.reset()

    def reset(self):
        self.preds = []
        self.gts = []

    def __call__(self, gt, pred):
        gt, pred = self.func(gt, pred) if self.func else (gt, pred)
        self.gts.append(gt)
        self.preds.append(pred)

    @property
    def result(self):
        preds = np.vstack(self.preds)
        gts = np.vstack(self.gts)
        n_classes = gts.shape[1]
        self.aps = np.array([self.calc_average_precision(gts[:, i].flatten(), preds[:, i].flatten())
                             for i in range(n_classes)])
        return self.aps.mean()

    @staticmethod
    def calc_average_precision(gt, pred):
        prec, rec = MeanAPMetric.interpolated_precision_recall_pairs(gt, pred)
        ap = np.sum((rec[1:] - rec[:-1]) * prec[:-1])
        return ap

    @staticmethod
    def interpolated_precision_recall_pairs(gt, pred):
        prec, rec = MeanAPMetric.precision_recall_pairs(gt, pred)
        prec = np.r_[0, prec, 0]
        rec = np.r_[0, rec, 1]
        # calculate the interpolated precision value
        for i in reversed(range(0, prec.size-1)):
            prec[i] = max(prec[i], prec[i+1])
        return prec, rec

    @staticmethod
    def precision_recall_pairs(gt, pred):
        si = np.argsort(-pred)   # sort in descending order
        tp = (gt[si] == 1)
        fp = (gt[si] == 0)
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        prec = tp / (tp + fp)
        rec = tp / np.sum(gt)
        return prec, rec
