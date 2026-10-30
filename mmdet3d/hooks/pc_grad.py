import torch
from mmcv.runner import Hook, HOOKS


@HOOKS.register_module()
class PCGradHook(Hook):
    """
    Asymmetric PCGrad for Detection + Occupancy
    """

    def __init__(
        self,
        det_loss_key,
        occ_loss_key,
        shared_param_prefix,
        eps=1e-6,
        verbose=False,
    ):
        if isinstance(det_loss_key, str):
            det_loss_key = [det_loss_key]
        if isinstance(occ_loss_key, str):
            occ_loss_key = [occ_loss_key]

        self.det_loss_key = det_loss_key
        self.occ_loss_key = occ_loss_key
        self.shared_param_prefix = shared_param_prefix
        self.eps = eps
        self.verbose = verbose

    def _match_loss(self, losses, prefixes):
        return [
            v for k, v in losses.items()
            if any(k.startswith(p) for p in prefixes)
        ]

    def after_backward(self, runner):
        model = runner.model
        breakpoint()

        # ===============================
        # 1. 从 model 中拿 loss dict
        # ===============================
        if not hasattr(model, 'losses'):
            if self.verbose:
                print('[PCGradHook] model has no attribute losses')
            return

        losses = model.losses

        det_losses = self._match_loss(losses, self.det_loss_key)
        occ_losses = self._match_loss(losses, self.occ_loss_key)

        if len(det_losses) == 0 or len(occ_losses) == 0:
            if self.verbose:
                print('[PCGradHook] Skip: missing det or occ loss')
            return

        loss_det = sum(det_losses)
        loss_occ = sum(occ_losses)

        # ===============================
        # 2. 找 shared parameters
        # ===============================
        shared_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(pref) for pref in self.shared_param_prefix):
                if p.grad is not None:
                    shared_params.append(p)

        if len(shared_params) == 0:
            if self.verbose:
                print('[PCGradHook] Skip: no shared parameters')
            return

        # ===============================
        # 3. 重新算 task gradients
        # ===============================
        g_det = torch.autograd.grad(
            loss_det,
            shared_params,
            retain_graph=True,
            allow_unused=True,
        )

        g_occ = torch.autograd.grad(
            loss_occ,
            shared_params,
            retain_graph=True,
            allow_unused=True,
        )

        # ===============================
        # 4. PCGrad (Det dominant)
        # ===============================
        conflict_cnt = 0
        for p, gd, go in zip(shared_params, g_det, g_occ):
            if gd is None or go is None:
                continue

            dot = torch.sum(gd * go)
            if dot < 0:
                proj = dot / (gd.norm() ** 2 + self.eps)
                go = go - proj * gd
                conflict_cnt += 1

            p.grad = gd + go

        if self.verbose:
            print(
                f'[PCGradHook] applied on {len(shared_params)} params, '
                f'conflicts resolved: {conflict_cnt}'
            )
