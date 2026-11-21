# 开发、测试与日常运行

日常两路论文推荐由 Mac mini 系统任务运行；个人 Windows 关机不影响调度。Mac 需要持续开机联网。运行目录为 `~/projects/paper-radar`，私有配置为 `~/.config/paper-radar/environment.json`，日志和运行回执为 `~/.local/state/paper-radar/`。

`cn.lhf.paper-materials` 和 `cn.lhf.paper-quantum` 每五分钟检查一次，分别从北京时间18:30、18:38开始准备，最早19:00、19:08投递。保留论文筛选、模型备援、图片、PDF精读、去重、失败重试与未知发送结果保护。

个人私有仓库 `feiniao0071/paper-radar` 用于开发和 CI；组织仓库 `QuantumAIMaterials/paper-radar` 接收通过检查的相同提交。组织仓库 Actions 受组织策略限制不表示 Mac 调度停止。

发布顺序：

1. 提交代码并推送个人仓库。`CI` 对所有分支的代码推送和 PR 运行，也可手动触发；仅状态 JSON 变更不触发。
2. 核对个人仓库该提交 SHA 的 `CI` 为 `completed/success`。本地测试、旧 SHA 或其他仓库通过不能代替此检查。
3. 使用 `python3 scripts/publish-organization.py` 同步当前分支到组织仓库；脚本检查当前 SHA 的个人仓库 CI，拒绝未通过的提交和强制覆盖。组织现有 PR 继续在原分支审阅。
4. 如需更新 Mac，按已验证的暂存、锁定、切换和回读流程发布；推送 Git 不会自动覆盖生产状态。

仓库中的论文 delivery workflows 只保留手动入口，默认预览。实发还要求 `RADAR_CLOUD_DELIVERY_ENABLED=true`，仅供受控切换运行主机时使用。不得在 Mac 正常调度时启动云端实发，也不得用组织或个人仓库中的旧 JSON 覆盖 Mac 最新状态、投递账本和待核队列。

首次创建个人仓库只复制代码历史，不复制生产密钥、财务记录或聊天数据。
