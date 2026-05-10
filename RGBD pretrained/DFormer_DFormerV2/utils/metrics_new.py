import torch
import numpy as np
from torch import Tensor
from typing import Tuple

class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update_hist(self, hist):
        self.hist += hist.to(self.hist.device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(
            target[keep] * self.num_classes + pred[keep], 
            minlength=self.num_classes**2
        ).view(self.num_classes, self.num_classes)

    def compute_stats(self):
        """Računa sve metrike za CSV logiranje"""
        h = self.hist.float()
        diag = h.diag()
        row_sum = h.sum(1) # Ground truth po klasi
        col_sum = h.sum(0) # Predikcija po klasi
        total = h.sum()
        eps = 1e-6

        # 1. IoU sa stabilnošću
        iu = diag / (row_sum + col_sum - diag + eps)
        miou = iu[1:].mean().item()

        # 2. Overall Pixel Accuracy
        overall_acc = diag.sum() / (total + eps)

        # 3. Pixel Accuracy FG (Globalni omjer za sve osim backgrounda)
        fg_correct = diag[1:].sum()
        fg_total = row_sum[1:].sum()
        pixel_acc_fg = (fg_correct / fg_total).item() if fg_total > 0 else 0.0

        # 4. Frequency Weighted IoU
        freq = row_sum / (total + eps)
        fwiou = (freq * iu).sum().item()
        
        # FW-IoU Foreground samo
        if fg_total > 0:
            freq_fg = row_sum[1:] / fg_total
            fwiou_fg = (freq_fg * iu[1:]).sum().item()
        else:
            fwiou_fg = 0.0

        # 5. Recall po klasama (diag / row_sum)
        cl_recall = diag / (row_sum + eps)

        return {
            "miou": miou * 100,
            "overall_acc": overall_acc.item() * 100,
            "pixel_acc_fg": pixel_acc_fg * 100,
            "fwiou": fwiou * 100,
            "fwiou_fg": fwiou_fg * 100,
            "cl_recall": (cl_recall * 100).cpu().numpy().tolist(),
            "ious": (iu * 100).cpu().numpy().tolist()
        }

    # Zadržavamo ove metode radi kompatibilnosti s postojećim train.py pozivima
    def compute_iou(self):
        res = self.compute_stats()
        return [round(x, 2) for x in res["ious"]], round(res["miou"], 2)

    def compute_pixel_acc(self):
        res = self.compute_stats()
        return [], round(res["overall_acc"], 2)

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) + 1e-6)
        return [], round(f1.mean().item() * 100, 2)