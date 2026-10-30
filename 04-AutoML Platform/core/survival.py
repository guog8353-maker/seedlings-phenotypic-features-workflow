"""生存分析模块：CoxPH、DeepSurv、加权 Cox。"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class BaseSurvivalModel:
    name: str = ""

    def fit(self, X: pd.DataFrame, time: pd.Series, event: pd.Series,
            weights: pd.Series = None):
        raise NotImplementedError

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_survival_function(self, X: pd.DataFrame):
        raise NotImplementedError


class CoxPHSurvival(BaseSurvivalModel):
    name = "CoxPH"

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.model_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X, time, event, weights=None):
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            raise ImportError("lifelines 未安装，请执行：pip install lifelines")

        df = X.copy()
        df["time"] = time.values
        df["event"] = event.values
        if weights is not None:
            df["weight"] = weights.values
        else:
            df["weight"] = 1.0

        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        df[numeric_cols] = self.scaler_.fit_transform(df[numeric_cols])

        self.model_ = CoxPHFitter(penalizer=self.penalizer)
        self.model_.fit(
            df,
            duration_col="time",
            event_col="event",
            weights_col="weight",
            robust=True,
        )
        return self

    def predict_risk(self, X):
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        X_scaled = X.copy()
        X_scaled[numeric_cols] = self.scaler_.transform(X[numeric_cols])
        return -self.model_.predict_partial_hazard(X_scaled).values

    def predict_survival_function(self, X):
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        X_scaled = X.copy()
        X_scaled[numeric_cols] = self.scaler_.transform(X[numeric_cols])
        return self.model_.predict_survival_function(X_scaled)


class WeightedCoxPH(CoxPHSurvival):
    name = "WeightedCoxPH"

    def fit(self, X, time, event, weights):
        return super().fit(X, time, event, weights)


class DeepSurvModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepSurv(BaseSurvivalModel):
    name = "DeepSurv"

    def __init__(self, hidden_dims=(64, 64), dropout=0.2, lr=1e-3,
                 num_epochs=100, batch_size=256, random_state=42,
                 device: str = None):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_ = StandardScaler()
        self.model_ = None

    def _negative_cox_partial_likelihood(self, risk_scores, time, event):
        """Cox 偏似然的负值（用于 PyTorch）。"""
        risk_scores = risk_scores.squeeze()
        time = time.squeeze()
        event = event.squeeze()
        order = torch.argsort(time, descending=True)
        risk_scores = risk_scores[order]
        event = event[order]
        log_risk = torch.logcumsumexp(risk_scores, dim=0)
        pl = risk_scores - log_risk
        return -torch.sum(pl * event)

    def fit(self, X, time, event, weights=None):
        torch.manual_seed(self.random_state)

        X_arr = self.scaler_.fit_transform(X.astype(float))
        time_arr = time.values.astype(float)
        event_arr = event.values.astype(float)

        X_tensor = torch.FloatTensor(X_arr).to(self.device)
        time_tensor = torch.FloatTensor(time_arr).to(self.device)
        event_tensor = torch.FloatTensor(event_arr).to(self.device)

        self.model_ = DeepSurvModel(X_arr.shape[1], self.hidden_dims, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X_tensor, time_tensor, event_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.num_epochs):
            for xb, tb, eb in loader:
                optimizer.zero_grad()
                risk = self.model_(xb)
                loss = self._negative_cox_partial_likelihood(risk, tb, eb)
                loss.backward()
                optimizer.step()
        return self

    def predict_risk(self, X):
        X_arr = self.scaler_.transform(X.astype(float))
        X_tensor = torch.FloatTensor(X_arr).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            risk = self.model_(X_tensor).cpu().numpy().squeeze()
        return risk

    def predict_survival_function(self, X):
        # 简化：返回基于 Breslow 估计的基线生存函数
        raise NotImplementedError("DeepSurv 的生存函数需结合基线风险估计，当前版本仅输出风险评分")


SURVIVAL_REGISTRY = {
    "CoxPH": CoxPHSurvival,
    "WeightedCoxPH": WeightedCoxPH,
    "DeepSurv": DeepSurv,
}


def compute_c_index(risk_scores: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """计算 Concordance Index。"""
    try:
        from lifelines.utils import concordance_index
    except ImportError:
        raise ImportError("lifelines 未安装")
    return concordance_index(time, -risk_scores, event)


class SurvivalRunner:
    def __init__(self, name: str, params: dict = None):
        if name not in SURVIVAL_REGISTRY:
            raise ValueError(f"未知生存模型：{name}")
        self.name = name
        self.params = params or {}
        self.model_: Optional[BaseSurvivalModel] = None

    def fit(self, X: pd.DataFrame, time: pd.Series, event: pd.Series,
            weights: pd.Series = None):
        cls = SURVIVAL_REGISTRY[self.name]
        self.model_ = cls(**self.params)
        self.model_.fit(X, time, event, weights)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_risk(X)

    def evaluate(self, X: pd.DataFrame, time: pd.Series, event: pd.Series) -> Dict:
        risk = self.predict_risk(X)
        c_index = compute_c_index(risk, time.values, event.values)
        metrics = {"c_index": round(c_index, 4)}
        try:
            metrics["brier"] = round(compute_brier_score(self.model_, X, time, event), 4)
        except Exception:
            metrics["brier"] = None
        try:
            metrics["calibration_slope"] = round(compute_calibration_slope(risk, time, event), 4)
        except Exception:
            metrics["calibration_slope"] = None
        return metrics


def compute_brier_score(model, X: pd.DataFrame, time: pd.Series, event: pd.Series, eval_time: float = None) -> float:
    """基于 Kaplan-Meier 逆概率加权（IPCW）的简化 Brier score。

    在 eval_time 处计算。若未提供，使用事件时间的 median。
    """
    from lifelines import KaplanMeierFitter
    import numpy as np

    t = time.values.astype(float)
    e = event.values.astype(float)
    if eval_time is None:
        eval_time = np.median(t)

    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=e)
    surv_prob = kmf.predict(eval_time)
    if surv_prob <= 0:
        surv_prob = 1e-6

    # 获取模型在 eval_time 处的生存概率
    try:
        surv_funcs = model.predict_survival_function(X)
        predicted_surv = np.array([fn(eval_time) for fn in surv_funcs])
    except Exception:
        # 若无法获取生存函数，使用风险评分近似
        risk = model.predict_risk(X)
        predicted_surv = 1 / (1 + np.exp(risk - np.mean(risk)))

    # IPCW 近似
    event_before = (t <= eval_time) & (e == 1)
    censored_before = (t <= eval_time) & (e == 0)
    observed_after = t > eval_time

    brier = 0.0
    n = len(t)
    for i in range(n):
        if event_before[i]:
            brier += (predicted_surv[i]) ** 2 / surv_prob
        elif observed_after[i]:
            brier += (1 - predicted_surv[i]) ** 2 / surv_prob
        elif censored_before[i]:
            brier += (1 - predicted_surv[i]) ** 2 / surv_prob
    return float(brier / n)


def compute_calibration_slope(risk_scores: np.ndarray, time: pd.Series, event: pd.Series, eval_time: float = None) -> float:
    """用风险评分分位数与观察事件率拟合校准斜率（简化版）。"""
    import numpy as np
    from sklearn.linear_model import LinearRegression

    t = time.values.astype(float)
    e = event.values.astype(float)
    if eval_time is None:
        eval_time = np.median(t)

    observed = ((t <= eval_time) & (e == 1)).astype(int)
    risk = np.asarray(risk_scores).ravel()

    # 按风险评分分 10 组
    n_bins = 10
    sorted_idx = np.argsort(risk)
    bin_size = max(1, len(risk) // n_bins)
    x_mean, y_mean = [], []
    for i in range(0, len(sorted_idx), bin_size):
        idx = sorted_idx[i:i + bin_size]
        x_mean.append(risk[idx].mean())
        y_mean.append(observed[idx].mean())

    x = np.array(x_mean).reshape(-1, 1)
    y = np.array(y_mean)
    if len(x) < 2:
        return 0.0
    reg = LinearRegression().fit(x, y)
    return float(reg.coef_[0])
