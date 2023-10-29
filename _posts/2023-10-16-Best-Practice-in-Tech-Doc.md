---
layout: post
title: 技术文档的最佳实践
tag: other
---

# 技术文档的最佳实践

编写技术文档时开发过程中常见的需求。一份好的开发文档可以有很多作用：

1. 帮助使用者快速入门；
2. 指导开发者进行项目编译；
3. 为开发者做维护进行指南等。

## 痛点

传统技术文档使用 Word 或者一些在线工具或多或少存在了一些问题，例如：

1. 文档和项目分开维护、导致版本管理困难；
2. 文档中的图表往往使用其他软件绘制，其他人在没有源文件的情况下难以维护；
3. 文档中包含较多的资源文件会导致整个文档的大小变大，下载缓慢、占用资源；
4. 使用多软件编写文档，软件付费策略各不相同合规性难以保证；
5. 维护文档的软件的跨平台性。

## 解决方案及其优势

针对以上问题，在实际工作中，我找到了一种经济可行的方法。

1. 利用 Markdown 来编写文档，实际工程项目中很多都会包含```README.MD```文件，我们可以使用 Markdown 将文档和项目同步进行维护；
2. 使用对 Markdown 友好的 Mermaid 来处理所需图标，以用来解决图表维护、文档大小的问题；
3. 使用开源/免费软件来进行文档维护。

### 软件选择

1. 编辑器 [VS code](https://code.visualstudio.com/)，其 [license](https://code.visualstudio.com/license?lang=zh-cn) 授予您使用该软件的某些权利，微软保留所有其他权利。您可以使用该软件任意数量的副本，以开发并测试您的应用程序，包括在您的内部公司网络内部署；
2. Markdown 插件（用于预览，可选，建议安装）选择 [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)，其 [license](https://marketplace.visualstudio.com/items/yzhang.markdown-all-in-one/license) 为MIT；
3. 图表处理使用 [Mermaid](https://mermaid.js.org/)， 其 [license](https://github.com/mermaid-js/mermaid/blob/develop/LICENSE) 为MIT；
4. Mermaid 插件（用于预览，可选）选择 [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid)，其 [license](https://marketplace.visualstudio.com/items/bierner.markdown-mermaid/license) 为MIT；

以上软件、插件、项目均提供Windows、Linux、MacOS 甚至 Web 端支持。

## 实例

| version | author | date | comment |
| - | - | - | - |
| 1.0 | mermaid group | 2023-10-22 | demo |

```
- [Mermaid项目文档](#mermaid项目文档)
  - [一、简介](#一简介)
  - [二、示例](#二示例)
    - [流程图](#流程图)
    - [时序图](#时序图)
    - [类图](#类图)
    - [状态图](#状态图)
    - [ER图](#er图)
    - [四象限图](#四象限图)
    - [用户体验旅程图](#用户体验旅程图)
    - [甘特图](#甘特图)
    - [git图](#git图)
```

### Mermaid项目文档

Mermaid 被提名并获得了 JS Open Source Awards (2019) 的 "The most exciting use of technology" 奖项!!!

#### 一、简介

Mermaid 是一个基于 Javascript 的图表绘制工具，通过解析类 Markdown 的文本语法来实现图表的创建和动态修改。Mermaid 诞生的主要目的是让文档的更新能够及时跟上开发进度。

#### 二、示例

下面是一些可以使用 Mermaid 创建的图表示例。点击 [语法](https://mermaid-js.github.io/mermaid/#/n00b-syntaxReference) 查看详情。

##### 流程图

```
flowchart LR
A[Hard] -->|Text| B(Round)
B --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
```

<pre class="mermaid">
flowchart LR
A[Hard] -->|Text| B(Round)
B --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
</pre>

##### 时序图

```
sequenceDiagram
Alice->>John: Hello John, how are you?
loop Healthcheck
    John->>John: Fight against hypochondria
end
Note right of John: Rational thoughts!
John-->>Alice: Great!
John->>Bob: How about you?
Bob-->>John: Jolly good!
```

<pre class="mermaid">
sequenceDiagram
Alice->>John: Hello John, how are you?
loop Healthcheck
    John->>John: Fight against hypochondria
end
Note right of John: Rational thoughts!
John-->>Alice: Great!
John->>Bob: How about you?
Bob-->>John: Jolly good!
</pre>

##### 类图

```
classDiagram
    note "From Duck till Zebra"
    Animal <|-- Duck
    note for Duck "can fly\ncan swim\ncan dive\ncan help in debugging"
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal()
    Animal: +mate()
    class Duck{
        +String beakColor
        +swim()
        +quack()
    }
    class Fish{
        -int sizeInFeet
        -canEat()
    }
    class Zebra{
        +bool is_wild
        +run()
    }
```

<pre class="mermaid">
classDiagram
    note "From Duck till Zebra"
    Animal <|-- Duck
    note for Duck "can fly\ncan swim\ncan dive\ncan help in debugging"
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal()
    Animal: +mate()
    class Duck{
        +String beakColor
        +swim()
        +quack()
    }
    class Fish{
        -int sizeInFeet
        -canEat()
    }
    class Zebra{
        +bool is_wild
        +run()
    }
</pre>

##### 状态图

```
stateDiagram-v2
[*] --> Still
Still --> [*]
Still --> Moving
Moving --> Still
Moving --> Crash
Crash --> [*]
```

<pre class="mermaid">
stateDiagram-v2
[*] --> Still
Still --> [*]
Still --> Moving
Moving --> Still
Moving --> Crash
Crash --> [*]
</pre>

##### ER图

```
erDiagram
    CAR ||--o{ NAMED-DRIVER : allows
    CAR {
        string registrationNumber
        string make
        string model
    }
    PERSON ||--o{ NAMED-DRIVER : is
    PERSON {
        string firstName
        string lastName
        int age
    }
```

<pre class="mermaid">
erDiagram
    CAR ||--o{ NAMED-DRIVER : allows
    CAR {
        string registrationNumber
        string make
        string model
    }
    PERSON ||--o{ NAMED-DRIVER : is
    PERSON {
        string firstName
        string lastName
        int age
    }
</pre>

##### 四象限图

```
quadrantChart
    title Reach and engagement of campaigns
    x-axis Low Reach --> High Reach
    y-axis Low Engagement --> High Engagement
    quadrant-1 We should expand
    quadrant-2 Need to promote
    quadrant-3 Re-evaluate
    quadrant-4 May be improved
    Campaign A: [0.3, 0.6]
    Campaign B: [0.45, 0.23]
    Campaign C: [0.57, 0.69]
    Campaign D: [0.78, 0.34]
    Campaign E: [0.40, 0.34]
    Campaign F: [0.35, 0.78]
```

<pre class="mermaid">
quadrantChart
    title Reach and engagement of campaigns
    x-axis Low Reach --> High Reach
    y-axis Low Engagement --> High Engagement
    quadrant-1 We should expand
    quadrant-2 Need to promote
    quadrant-3 Re-evaluate
    quadrant-4 May be improved
    Campaign A: [0.3, 0.6]
    Campaign B: [0.45, 0.23]
    Campaign C: [0.57, 0.69]
    Campaign D: [0.78, 0.34]
    Campaign E: [0.40, 0.34]
    Campaign F: [0.35, 0.78]
</pre>

##### 用户体验旅程图

```
journey
  title My working day
  section Go to work
    Make tea: 5: Me
    Go upstairs: 3: Me
    Do work: 1: Me, Cat
  section Go home
    Go downstairs: 5: Me
    Sit down: 3: Me
```

<pre class="mermaid">
journey
  title My working day
  section Go to work
    Make tea: 5: Me
    Go upstairs: 3: Me
    Do work: 1: Me, Cat
  section Go home
    Go downstairs: 5: Me
    Sit down: 3: Me
</pre>

##### 甘特图

```
gantt
    title A Gantt Diagram
    dateFormat YYYY-MM-DD
    section Section
        A task          :a1, 2014-01-01, 30d
        Another task    :after a1, 20d
    section Another
        Task in Another :2014-01-12, 12d
        another task    :24d
```

<pre class="mermaid">
gantt
    title A Gantt Diagram
    dateFormat YYYY-MM-DD
    section Section
        A task          :a1, 2014-01-01, 30d
        Another task    :after a1, 20d
    section Another
        Task in Another :2014-01-12, 12d
        another task    :24d
</pre>

##### git图

```
---
title: Example Git diagram
---
gitGraph
   commit
   commit
   branch develop
   checkout develop
   commit
   commit
   checkout main
   merge develop
   commit
   commit
```

<pre class="mermaid">
---
title: Example Git diagram
---
gitGraph
   commit
   commit
   branch develop
   checkout develop
   commit
   commit
   checkout main
   merge develop
   commit
   commit
</pre>

## 参考链接

