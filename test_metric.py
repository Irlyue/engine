import torch
import metric
import unittest

import numpy as np
from sklearn.metrics import confusion_matrix


class TestAccuracyMetric(unittest.TestCase):

    def test_100_percent(self):
        y = torch.randint(10, size=(100, 16))
        with metric.AccuracyMetric() as mt:
            for a, b in zip(y, y):
                mt(a, b)
        self.assertEqual(mt.result, 1.0)

    def test_90_percent(self):
        y1 = torch.randint(10, size=(100, 16))
        y2 = y1.clone()
        y2[90:] += 1
        with metric.AccuracyMetric() as mt:
            for a, b in zip(y1, y2):
                mt(a, b)
        self.assertEqual(mt.result, 0.9)

    def test_0_percent(self):
        y1 = torch.randint(10, size=(100, 16))
        y2 = y1.clone() + 1
        with metric.AccuracyMetric() as mt:
            for a, b in zip(y1, y2):
                mt(a, b)
        self.assertEqual(mt.result, 0.0)

    def test_func_100_percent(self):
        def _func(_x, _y):
            return _x, _y.argmax(dim=1)
        logits = torch.randn(100, 16, 10)
        labels = logits.argmax(dim=2)
        with metric.AccuracyMetric(_func) as mt:
            for a, b in zip(labels, logits):
                mt(a, b)
        self.assertEqual(mt.result, 1.0)


class TestMeanIoUMetric(unittest.TestCase):

    def test_2_classes_with_numpy_input(self):
        y_true = np.random.randint(2, size=(100,))
        y_pred = np.random.randint(2, size=(100,))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        iou0 = tn / (tn + fn + fp)
        iou1 = tp / (tp + fp + fn)
        miou = (iou0 + iou1) / 2.
        with metric.MeanIoUMetric(2) as mt:
            mt(y_true, y_pred)
        self.assertEqual(miou, mt.result)
        self.assertEqual(iou0, mt.ious[0])
        self.assertEqual(iou1, mt.ious[1])

    def test_20_classes_with_numpy_input(self):
        C = 20
        y_true = np.random.randint(C, size=(1000, 16))
        y_pred = np.random.randint(C, size=(1000, 16))
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
        tp = cm.diagonal()
        total = cm.sum(axis=1) + cm.sum(axis=0) - tp
        ious = tp * 1.0 / total
        miou = ious.mean()

        with metric.MeanIoUMetric(C) as mt:
            for yt, yp in zip(y_true, y_pred):
                mt(yt, yp)
        self.assertEqual(miou, mt.result)
        self.assertTrue(np.all(cm == mt.cm))

    def test_2_classes_with_tensor_input(self):
        y_true = torch.randint(2, size=(100,))
        y_pred = torch.randint(2, size=(100,))
        tp = torch.sum((y_true == 1) & (y_pred == 1)).to(torch.float64)
        tn = torch.sum((y_true == 0) & (y_pred == 0)).to(torch.float64)
        fp = torch.sum((y_pred == 1) & (y_true == 0)).to(torch.float64)
        fn = torch.sum((y_pred == 0) & (y_true == 1)).to(torch.float64)
        iou0 = tn / (tn + fn + fp)
        iou1 = tp / (tp + fp + fn)
        miou = (iou0 + iou1) / 2.
        iou0, iou1, miou = [item.item() for item in [iou0, iou1, miou]]
        with metric.MeanIoUMetric(2) as mt:
            mt(y_true, y_pred)
        self.assertEqual(miou, mt.result)
        self.assertEqual(iou0, mt.ious[0])
        self.assertEqual(iou1, mt.ious[1])

    def _test_20_classes_with_tensor_input(self, C, device, dtype):
        y_true = torch.randint(C, size=(1000, 16), device=device, dtype=dtype)
        y_pred = torch.randint(C, size=(1000, 16), device=device, dtype=dtype)
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
        tp = cm.diagonal()
        total = cm.sum(axis=1) + cm.sum(axis=0) - tp
        ious = tp * 1.0 / total
        miou = ious.mean()

        with metric.MeanIoUMetric(C) as mt:
            for yt, yp in zip(y_true, y_pred):
                mt(yt, yp)
        self.assertEqual(miou, mt.result)
        self.assertTrue(np.all(cm == mt.cm))

    def test_20_classes_with_tensor_input_cpu(self):
        self._test_20_classes_with_tensor_input(20, 'cpu', torch.int64)

    def test_20_classes_with_tensor_input_cuda(self):
        if torch.cuda.is_available():
            self._test_20_classes_with_tensor_input(20, 'cuda', torch.int64)

    def test_20_classes_with_tensor_input_uint8(self):
        self._test_20_classes_with_tensor_input(20, 'cpu', torch.uint8)


if __name__ == '__main__':
    unittest.main()
