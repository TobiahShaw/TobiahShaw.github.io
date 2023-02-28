---
layout: post
title: Github kex_exchange_identification 
tag: other
---

# Github 发生 kex_exchange_identification 解决办法

## 现象：

配置完 SSH key 当时能用，重启后就一直报 ```kex_exchange_identification: Connection closed by remote host```。

## 解决方案

尝试指定端口号22.编辑文件```~/.ssh/config```，增加以下内容：

```
# github
Host github.com
HostName github.com
Port 22
```
