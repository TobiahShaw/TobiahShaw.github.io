---
layout: post
title: 如何在 markdown 文件中使用 mermaid
tag: other
---
# 在 Vs Code 中使用（推荐）：

1. 安装插件 Markdown all in one;
2. 安装插件 Markdown Preview Mermaid Support;
3. 重启 Vs Code;
4. 使用代码块来编写 mermaid 并进行预览、输出。

# 实例

```markdown
    ```mermaid
    graph TD;
        A-->B;
        A-->C;
        B-->D;
        C-->D;
    ```
```
