import torch
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    梯度下降 根据超参数的内容name会自适应改变
    主要即‘SGD’、‘SGD_momentum’、‘SGD_momentum_nesterov’三种
    注意若要使用nesterov需要在超参数字典内加上'nesterov': True
    需要设置的超参数：
    SGD: lr:float
    SGD_momentum : lr momentum:float
    SGD_momentum_nesterov : lr momentum:float nesterov:bool
    """
    def __init__(self, params, hyperparams):
        super(SGD, self).__init__(params, hyperparams)
        self.name = 'SGD'
        if hyperparams.get('momentum') is not None:
            if hyperparams.get('nesterov'):
                self.name = 'SGD_momentum_nesterov'
            else:
                self.name = 'SGD_momentum'
            # 动量法需要存储累积梯度，因此需要初始化vt张量，值得注意的是，vt需要对应每一个参数的维度
            self.state_momentum = []
            for p1 in self.param_groups:
                for p2 in p1['params']:
                    feature_dim = p2.shape
                    vt = torch.zeros(feature_dim).to(p2.device)
                    self.state_momentum.append(vt)
        else:
            if hyperparams.get('nesterov'):
                raise RuntimeError("不能在不使用momentum的情况下设置nesterov加速梯度！")

    @torch.no_grad()
    def step(self, closure=False):
        # 普通SGD直接减梯度学习率的积
        if self.defaults.get('momentum') is None:
            for p1 in self.param_groups:
                for p2 in p1['params']:
                    p2.data.sub_(p1['lr'] * p2.grad)
        else:  # 存储累计梯度，momentum控制过去的量
            for p1 in self.param_groups:
                for p2, v in zip(p1['params'], self.state_momentum):
                    v[:] = p1['momentum'] * v + p1['lr'] * p2.grad
                    # nesterov修正参数，多减一个动量与累计梯度的积
                    if self.defaults.get('nesterov'):
                        p2[:] -= (1 + p1['momentum']) * v
                    else:
                        p2[:] -= v


class AdaGrad(Optimizer):
    """
    自适应学习率
    超参数： lr:float
    """
    def __init__(self, params, hyperparams):
        super(AdaGrad, self).__init__(params, hyperparams)
        self.state_G = []
        self.name = 'AdaGrad'
        for p1 in self.param_groups:
            for p2 in p1['params']:
                feature_dim = p2.shape
                gtii = torch.zeros(feature_dim).to(p2.device)
                self.state_G.append(gtii)

    @torch.no_grad()
    def step(self, closure=False):
        epsilon = 1e-8  # 保证分母不为0
        for p1 in self.param_groups:
            for p2, g_tii in zip(p1['params'], self.state_G):
                # 累积平方梯度
                g_tii[:] += torch.square(p2.grad)
                p2[:] -= p1['lr'] * p2.grad / torch.sqrt(g_tii + epsilon)


class AdaDelta(Optimizer):
    """
    自适应学习率
    超参数： gamma: float 一般为0.9 类似动量
    """
    def __init__(self, params, hyperparams):
        super(AdaDelta, self).__init__(params, hyperparams)
        self.state_Eg = []
        self.name = 'AdaDelta'
        for p1 in self.param_groups:
            for p2 in p1['params']:
                feature_dim = p2.shape
                gtii = torch.zeros(feature_dim).to(p2.device)
                deltatii = torch.zeros(feature_dim).to(p2.device)
                self.state_Eg.append((gtii, deltatii))

    @torch.no_grad()
    def step(self, closure=False):
        epsilon = 1e-8
        for p1 in self.param_groups:
            for p2, (Eg_t, delta_t) in zip(p1['params'], self.state_Eg):
                # 梯度平方与参数差平方的衰减平均
                Eg_t[:] = p1['gamma'] * Eg_t + (1 - p1['gamma']) * torch.square(p2.grad)
                delta_theta_t = torch.sqrt(delta_t + epsilon) * p2.grad / torch.sqrt(Eg_t + epsilon)
                p2[:] -= delta_theta_t
                delta_t[:] = p1['gamma'] * delta_t + (1 - p1['gamma']) * torch.square(delta_theta_t)


class RMSProp(Optimizer):
    """
    初代的AdaDelta
    超参数： lr:float
    """
    def __init__(self, params, hyperparams):
        super(RMSProp, self).__init__(params, hyperparams)
        self.state_Eg = []
        self.name = 'RMSProp'
        for p1 in self.param_groups:
            for p2 in p1['params']:
                feature_dim = p2.shape
                gtii = torch.zeros(feature_dim).to(p2.device)
                self.state_Eg.append(gtii)

    @torch.no_grad()
    def step(self, closure=False):
        epsilon = 1e-8
        for p1 in self.param_groups:
            for p2, Eg_t in zip(p1['params'], self.state_Eg):
                Eg_t[:] = 0.9 * Eg_t + 0.1 * torch.square(p2.grad)
                p2[:] -= p1['lr'] * p2.grad / torch.sqrt(Eg_t + epsilon)


class Adam(Optimizer):
    def __init__(self, params, hyperparams):
        super(Adam, self).__init__(params, hyperparams)
        self.state_adam = []
        self.name = 'Adam'
        # 时间步优化器内记录
        self.t = 1
        for p1 in self.param_groups:
            for p2 in p1['params']:
                feature_dim = p2.shape
                mt = torch.zeros(feature_dim).to(p2.device)
                vt = torch.zeros(feature_dim).to(p2.device)
                self.state_adam.append((mt, vt))

    @torch.no_grad()
    def step(self, closure=False):
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8  # 几个衰减平均的参数默认值 一般不需修改
        for p1 in self.param_groups:
            for p2, (mt, vt) in zip(p1['params'], self.state_adam):
                mt[:] = beta1 * mt + (1 - beta1) * p2.grad
                vt[:] = beta2 * vt + (1 - beta2) * torch.square(p2.grad)
                mt_hat = mt / (1 - beta1 ** self.t)
                vt_hat = vt / (1 - beta2 ** self.t)
                p2[:] -= p1['lr'] * mt_hat / (torch.sqrt(vt_hat) + epsilon)
        self.t += 1


class AdaMax(Optimizer):
    def __init__(self, params, hyperparams):
        super(AdaMax, self).__init__(params, hyperparams)
        self.state_adam = []
        self.name = 'AdaMax'
        self.t = 1
        for p1 in self.param_groups:
            for p2 in p1['params']:
                feature_dim = p2.shape
                mt = torch.zeros(feature_dim).to(p2.device)
                ut = torch.zeros(feature_dim).to(p2.device)
                self.state_adam.append((mt, ut))

    @torch.no_grad()
    def step(self, closure=False):
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for p1 in self.param_groups:
            for p2, (mt, ut) in zip(p1['params'], self.state_adam):
                mt[:] = beta1 * mt + (1 - beta1) * p2.grad
                temp = torch.cat([(beta2 * ut).unsqueeze(0), (p2.grad.abs() + epsilon).unsqueeze(0)], 0)
                ut[:] = torch.amax(temp, 0, keepdim=False)
                mt_hat = mt / (1 - beta1 ** self.t)
                p2[:] -= p1['lr'] * mt_hat / ut
        self.t += 1


class NAdam(Optimizer):
    def __init__(self, params, hyperparams):
        super(NAdam, self).__init__(params, hyperparams)
        self.state_adam = []
        self.t = 1
        self.name = 'NAdam'
        for p1 in self.param_groups:
            for p2 in p1['params']:
                feature_dim = p2.shape
                mt = torch.zeros(feature_dim).to(p2.device)
                vt = torch.zeros(feature_dim).to(p2.device)
                self.state_adam.append((mt, vt))

    @torch.no_grad()
    def step(self, closure=False):
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for p1 in self.param_groups:
            for p2, (mt, vt) in zip(p1['params'], self.state_adam):
                mt[:] = beta1 * mt + (1 - beta1) * p2.grad
                vt[:] = beta2 * vt + (1 - beta2) * torch.square(p2.grad)
                mt_hat = mt / (1 - beta1 ** self.t)
                vt_hat = vt / (1 - beta2 ** self.t)
                p2[:] -= p1['lr'] * (beta1 * mt_hat + (1 - beta1) * p2.grad) / (1 - beta1 ** self.t) / (torch.sqrt(vt_hat) + epsilon)
        self.t += 1
