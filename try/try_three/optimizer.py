import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math


class Optimizer:
    """3D高斯泼溅优化器"""

    def __init__(self, gaussian_model, args):
        self.gaussian_model = gaussian_model
        self.args = args

        # 设置优化器
        self.setup_optimizer()

        # 学习率调度器
        self.setup_schedulers()

    def setup_optimizer(self):
        """设置优化器参数"""
        params = []

        # 位置参数
        params.append({
            'params': [self.gaussian_model._xyz],
            'lr': self.args.position_lr_init * self.gaussian_model.spatial_lr_scale,
            'name': 'xyz'
        })

        # 特征参数（球谐函数）
        params.append({
            'params': [self.gaussian_model._features_dc],
            'lr': self.args.feature_lr,
            'name': 'f_dc'
        })

        params.append({
            'params': [self.gaussian_model._features_rest],
            'lr': self.args.feature_lr / 20.0,
            'name': 'f_rest'
        })

        # 不透明度参数
        params.append({
            'params': [self.gaussian_model._opacity],
            'lr': self.args.opacity_lr,
            'name': 'opacity'
        })

        # 缩放参数
        params.append({
            'params': [self.gaussian_model._scaling],
            'lr': self.args.scaling_lr,
            'name': 'scaling'
        })

        # 旋转参数
        params.append({
            'params': [self.gaussian_model._rotation],
            'lr': self.args.rotation_lr,
            'name': 'rotation'
        })

        # 创建优化器
        self.optimizer = optim.Adam(params, lr=0.0, eps=1e-15)

        # 位置学习率衰减参数
        self.xyz_scheduler_args = {
            'max_steps': self.args.position_lr_max_steps,
            'lr_init': self.args.position_lr_init,
            'lr_final': self.args.position_lr_final
        }

    def setup_schedulers(self):
        """设置学习率调度器"""

        # 位置学习率衰减函数
        def xyz_lr_lambda(step):
            if step < self.xyz_scheduler_args['max_steps']:
                t = step / self.xyz_scheduler_args['max_steps']
                lr = self.xyz_scheduler_args['lr_init'] * (1 - t) + self.xyz_scheduler_args['lr_final'] * t
            else:
                lr = self.xyz_scheduler_args['lr_final']

            # 应用空间尺度因子
            return lr * self.gaussian_model.spatial_lr_scale / self.args.position_lr_init

        # 创建调度器
        self.schedulers = []

        # 为每个参数组创建单独的调度器
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                scheduler = LambdaLR(self.optimizer, lr_lambda=xyz_lr_lambda)
                self.schedulers.append(scheduler)
                break

    def step(self, iteration):
        """执行优化步骤"""
        # 更新学习率
        self.update_learning_rate(iteration)

        # 执行优化步骤
        self.optimizer.step()

        # 更新调度器
        for scheduler in self.schedulers:
            scheduler.step()

    def update_learning_rate(self, iteration):
        """更新学习率（手动实现）"""
        # 位置学习率衰减
        if iteration < self.xyz_scheduler_args['max_steps']:
            t = iteration / self.xyz_scheduler_args['max_steps']
            lr = self.xyz_scheduler_args['lr_init'] * (1 - t) + self.xyz_scheduler_args['lr_final'] * t
        else:
            lr = self.xyz_scheduler_args['lr_final']

        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = lr * self.gaussian_model.spatial_lr_scale
                break

    def zero_grad(self, set_to_none=True):
        """清零梯度"""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """获取优化器状态字典"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """加载优化器状态字典"""
        self.optimizer.load_state_dict(state_dict)


class AdaptiveLearningRate:
    """自适应学习率调度器"""

    def __init__(self, initial_lr, min_lr=1e-6, patience=100, factor=0.5):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor

        self.best_loss = float('inf')
        self.counter = 0
        self.current_lr = initial_lr

    def step(self, current_loss):
        """根据当前损失调整学习率"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.counter = 0
                print(f"Reducing learning rate to {self.current_lr}")

        return self.current_lr

    def get_lr(self):
        """获取当前学习率"""
        return self.current_lr


def exponential_decay_lr(initial_lr, decay_rate, decay_steps):
    """指数衰减学习率函数"""

    def lr_lambda(step):
        return initial_lr * (decay_rate ** (step / decay_steps))

    return lr_lambda


def cosine_annealing_lr(initial_lr, T_max, eta_min=0):
    """余弦退火学习率函数"""

    def lr_lambda(step):
        if step < T_max:
            return eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * step / T_max))
        else:
            return eta_min

    return lr_lambda


def warmup_cosine_annealing_lr(initial_lr, warmup_steps, T_max, eta_min=0):
    """预热+余弦退火学习率函数"""

    def lr_lambda(step):
        if step < warmup_steps:
            # 线性预热
            return initial_lr * (step / warmup_steps)
        elif step < T_max:
            # 余弦退火
            step_adj = step - warmup_steps
            T_max_adj = T_max - warmup_steps
            return eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * step_adj / T_max_adj))
        else:
            return eta_min

    return lr_lambda
