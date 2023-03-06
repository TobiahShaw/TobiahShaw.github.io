---
layout: post
title: Github kex_exchange_identification 
tag: other
---

# Github 发生 kex_exchange_identification 解决办法

## 现象：

配置完 SSH key 当时能用，重启后就一直报 ```kex_exchange_identification: Connection closed by remote host```。

## 解决方案

1. 验证连接性```ssh -T git@github.com -p 22```

2. 编辑文件```~/.ssh/config```，增加以下内容指定端口和IdentityFile：

```
# github
Host github.com
HostName github.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_ed25519
Port 22
```
## 最终结论

如果指定 ssh key 和端口都无法解决，发生后又能恢复的，考虑是网络原因导致的。
