# 统一发布入口

共用工具：[feiniao0071/lab-release-tools](https://github.com/feiniao0071/lab-release-tools)。后台只在Mac mini运行，每5分钟检查个人main对应CI，通过后快进组织配置的接收分支。开发电脑关机不阻止此检查。

本项目提交代码后：

```sh
lab-release submit radar
lab-release check radar
lab-release sync radar
lab-release status radar
```

`submit`把当前开发checkout的已提交版本普通推送到个人main；有分叉则拒绝，不强推。随后等待CI，Mac定时任务自动同步。`check`实时核验但不推送，`sync`立即尝试一次，`status`显示带时间的缓存结果。

原`python scripts/publish-organization.py --check`及默认发布命令调用同一工具，并核当前本地SHA必须等于个人main。它不再按本地分支名任意选择组织目标。

接收分支：显微镜main；财务feat/finance-overview-migration-20261118（PR #12）；论文fix/model-unavailable-fallback（PR #3）。保留审阅流程，不自动合并组织main。组织分叉/超前/分支缺失则停止；CI失败、跳过或缺失任务不推送。组织论文仓库保持原有公开属性，其他此处列出的仓库私有。

共用工具使用固定安装版本，凭据保存在Mac本机gh；Actions没有组织写入凭据。旧WSL显微镜同步timer已停用。既有Git remote别名保持，统一命令直接使用核定的仓库和分支，不受别名差异影响。

这里只发布代码。正式服务更新、备份、回退和实机验收仍走本项目原有部署流程；不把GitHub版本或CI成功当成现场已经更新。
