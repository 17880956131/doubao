# 豆宝 dsh-go 插件（可分发）

豆宝手机本地数字人的 dsh-go 原生插件：输入框胶囊、状态条、面板、回合尾注；自带 Python 服务并在需要时自动拉起。

## 安装

### 方式一：本地 tarball（不需要 npm 账号）

```sh
dsh plugin --profile web add ./doubao-digital-human-0.1.0.tgz
```

或指向解压后的目录：

```sh
dsh plugin --profile web add ./doubao-plugin
```

安装后 App 里出现「豆宝」胶囊和面板；`lib/` 改动需重启引擎。

### 方式二：npm（发布后）

```sh
dsh plugin --profile web add doubao-digital-human
```

## 发布到 npm（需要 npm 账号）

```sh
npm login
npm publish --access public     # 包名 doubao-digital-human
```

发布后：
1. 用户安装：`dsh plugin --profile web add doubao-digital-human`
2. 想出现在「插件市场」：把包名/版本加进你使用的市场索引（用户添加索引地址后才可见）

## 服务目录查找顺序

1. 环境变量 `DOUBAO_HOME`
2. 插件自带的 `server/`（本包已内置，约 2.6 MB）
3. `~/workspace/digital-human`
4. `~/doubao`
5. `/root/workspace/digital-human`

数据写在服务目录的 `data/` 下，不会随插件发布。

## 自启动

- 默认开启：插件加载时若 8100 没响应，会拉起 `python3 server.py` 并尝试守护进程
- 关闭：服务目录 `data/ui_params.json` 里 `plugin_autostart: false`

## 卸载

```sh
dsh plugin --profile web remove doubao-digital-human
```

再删除插件目录即可。
