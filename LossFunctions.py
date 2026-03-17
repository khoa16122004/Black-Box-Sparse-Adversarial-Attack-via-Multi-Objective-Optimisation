import numpy as np
import torch
import math


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def to_pytorch(tensor_image):
    return torch.from_numpy(tensor_image).permute(2, 0, 1)


class UnTargeted:
    def __init__(self, model, true, unormalize=False, to_pytorch=False):
        self.model = model
        self.true = true
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch

    def get_label(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        return y

    def __call__(self, img):

        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y != self.true else False

        f_true = math.log(math.exp(preds[self.true]) + 1e-30)
        preds[self.true] = -math.inf

        f_other = math.log(math.exp(max(preds)) + 1e-30)
        # return [is_adversarial, float(f_true - f_other)]
        return [is_adversarial, float(f_other - f_true)]

    def batch(self, imgs):
        imgs = np.asarray(imgs, dtype=np.float32)
        if self.unormalize:
            imgs = imgs * 255.

        if self.to_pytorch:
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2)
            preds = self.model.predict(x)
            if isinstance(preds, torch.Tensor):
                preds_t = preds.detach().cpu()
            else:
                preds_t = torch.from_numpy(np.asarray(preds))

            y = torch.argmax(preds_t, dim=1)
            true_scores = preds_t[:, self.true]
            masked = preds_t.clone()
            masked[:, self.true] = -torch.inf
            other_scores = torch.max(masked, dim=1).values
            margins = other_scores - true_scores

            return [[bool(y[i].item() != self.true), float(margins[i].item())] for i in range(preds_t.shape[0])]

        preds = self.model.predict(imgs)
        preds = np.asarray(preds)
        y = np.argmax(preds, axis=1)
        true_scores = preds[:, self.true]
        masked = preds.copy()
        masked[:, self.true] = -np.inf
        other_scores = np.max(masked, axis=1)
        margins = other_scores - true_scores
        return [[bool(y[i] != self.true), float(margins[i])] for i in range(preds.shape[0])]


class Targeted:
    def __init__(self, model, true, target, unormalize=False, to_pytorch=False):
        self.model = model
        self.true = true
        self.target = target
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch

    def get_label(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        return y

    def __call__(self, img):

        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y == self.target else False
        #print("current label %d target label %d" % (y, self.target))
        f_target = preds[self.target]
        #preds[self.true] = -math.inf

        f_other = math.log(sum(math.exp(pi) for pi in preds))
        return [is_adversarial, f_other - f_target]

    def batch(self, imgs):
        imgs = np.asarray(imgs, dtype=np.float32)
        if self.unormalize:
            imgs = imgs * 255.

        if self.to_pytorch:
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2)
            preds = self.model.predict(x)
            if isinstance(preds, torch.Tensor):
                preds_t = preds.detach().cpu()
            else:
                preds_t = torch.from_numpy(np.asarray(preds))

            y = torch.argmax(preds_t, dim=1)
            f_target = preds_t[:, self.target]
            f_other = torch.logsumexp(preds_t, dim=1)
            vals = f_other - f_target
            return [[bool(y[i].item() == self.target), float(vals[i].item())] for i in range(preds_t.shape[0])]

        preds = self.model.predict(imgs)
        preds = np.asarray(preds)
        y = np.argmax(preds, axis=1)
        f_target = preds[:, self.target]
        f_other = np.log(np.sum(np.exp(preds), axis=1) + 1e-30)
        vals = f_other - f_target
        return [[bool(y[i] == self.target), float(vals[i])] for i in range(preds.shape[0])]
