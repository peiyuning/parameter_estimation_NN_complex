# tools.py 物理审计和清理记录

日期：2026-04-20

## 当前原则

- TLS 保留 renewal/iid baseline。
- shelving/lambda/v/ladder 只保留 nonrenewal kernel 工具。
- 非 TLS 的 scalar waiting-time、scalar Fisher、scalar Torch theory、toy iid mixture 全部从 `tools.py` 删除。
- three-level 训练或 benchmark 不能再通过 scalar waiting density 或 scalar Laplace transform 静默运行。

## 已删除的非 TLS iid 分支

- 人造等待时间 / Fisher 诊断。
- 双分支 TLS 混合诊断。
- three-level scalar spectral waiting-time builder。
- shelving/lambda/v/ladder scalar waiting-time wrappers。
- shelving/lambda/v/ladder scalar one-interval Fisher wrappers。
- three-level Torch scalar waiting-time / scalar Laplace transform。
- `build_model_theory_torch()` 的 three-level 诊断 override。
- trajectory generator 里的 three-level renewal 模式。

## 仍保留的物理对象

- TLS renewal trajectory generation and analytic waiting density。
- TLS Torch waiting-time / Laplace theory。
- three-level kernel matrix：
  `build_noniid_three_level_kernel_model`,
  `evaluate_detected_kernel_matrix`,
  `noniid_three_level_kernel_matrix_complex`,
  `noniid_three_level_determinant_complex`。
- three-level stationary one-interval marginal：
  `noniid_three_level_aggregated_waiting_time_real`。
  这个只用于图示/诊断，不是完整序列 likelihood。

## 本轮新增修正

- `QuantumModel.computeLikelihood()` 被保留，但已去掉旧的 `tau*r` 人工因子；现在和 TLS 闭式 iid product 烟测一致。
- 删除了旧的 `generate_prob_from_model_list()`、`prob_no_click()`。
- `QuantumModel` 现在支持显式 `detected_indices`，MC 轨迹会用 QuTiP `col_which` 过滤观测通道。
- 修正了 `simulateTrajectoryFixedJumps()` 的分段续跑逻辑：跨段 waiting time 现在保留无跳跃时间，不再把下一段平移到上一跳时刻。
- TLS 入口不再暴露可调 `gamma`；全文件使用 `gamma=1` 单位制。
- `build_jump_kernel_spectral_model()` 现在是 channel-to-channel kernel；`observed_channels` 和 `hidden_channels` 显式区分。
- `kernel_transition_matrix()` 默认自适应加长积分窗口，减少 stationary marginal 被截尾污染的风险。
- `trapz_integral()` 替代 `np.trapz`，兼容当前 NumPy 环境。

## 当前逐项判定

| 区域 | 当前判定 | 说明 |
| --- | --- | --- |
| `QuantumModel` | OK/谨慎使用 | 适合单符号观测记录的 MC 和递归 likelihood；多符号 non-iid 仍应使用 kernel matrix。 |
| `generate_clicks_TLS`, `generate_clicks_TLS_fixed_time` | OK/TLS-only | 单通道 TLS 轨迹；`simulateTrajectoryFixedJumps()` 的跨段等待时间已修正。 |
| `compute_log_likelihood_analytical`, `compute_likelihood_analytical` | OK/TLS-only | iid product of TLS waiting density；不暴露 `gamma` 参数。 |
| `compute_likelihood_analytical_Classical` | 诊断-only/TLS-only | 均值的高斯近似，不是 exact trajectory likelihood。 |
| `get_estimates_Bayesian` | TLS-only | 只扫 `Delta`，默认固定 `Omega`；不能用于 two-parameter three-level。 |
| `compute_waiting_time`, `tls_waiting_time_real` | OK/TLS-only | TLS waiting density；递归 likelihood 与闭式 product 烟测一致；单位为 `gamma=1`。 |
| `zero_mode_profile`, `compute_weighted_interaction_matrix`, persistence helpers | 诊断-only | 拓扑/图示工具，不是物理生成模型或 likelihood。 |
| `_build_three_level_channels` | OK/需论文同步 | 显式返回 physical channels、reset states 和 channel names；gamma_aux 是可选额外 leakage。 |
| `build_jump_kernel_spectral_model` | 核心可用 | 构造 `K_ij(t)=Tr[J_i exp(L0 t) rho_j]`，其中 `j` 是上一观测通道，`i` 是下一观测通道。 |
| `build_noniid_three_level_kernel_model` | 核心可用但观测模型需选择 | 默认观测所有物理 emission channel；如果只观测 bright photon，需要显式传 `observed_channels`。 |
| `kernel_transition_matrix`, `kernel_conditional_transition_matrix`, `kernel_stationary_distribution` | OK | 转移矩阵积分现在自适应加长；条件化版本用于最终 click 概率不足 1 的情形。 |
| `noniid_three_level_aggregated_waiting_time_real` | 诊断-only | stationary single-interval marginal，不是完整 nonrenewal sequence likelihood。 |
| `build_model_theory_torch` | OK/TLS-only | 现在只返回 TLS scalar theory；three-level scalar Torch theory 已删。 |

## 仍需下一步补的东西

1. 为 three-level nonrenewal 训练写真正的 kernel-matrix likelihood 或 kernel-matrix Torch theory。
2. 给 `build_noniid_three_level_kernel_model()` 的观测 instrument 写清楚物理定义。
3. 进一步决定旧的 TLS Bayesian/CLT helper 是否还需要保留；它们现在只应作为 TLS baseline 使用。
