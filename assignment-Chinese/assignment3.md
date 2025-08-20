---
layout: page
title: 作业3
mathjax: true
permalink: /assignments2025/assignment3/
---

<span style="color:red">本作业截止时间为**2025年5月30日（星期五）太平洋标准时间晚上11:59**。</span>

包含Colab笔记本的初始代码可从[此处下载](https://drive.google.com/file/d/14J1IBXY50431YBbOWmPpYngudbnIqUFP/view?usp=drive_link)。

- [设置](#setup)
- [目标](#goals)
- [问题1：使用Transformer进行图像描述生成](#q1-使用transformer进行图像描述生成)
- [问题2：用于图像分类的自监督学习](#q2-用于图像分类的自监督学习)
- [问题3：去噪扩散概率模型](#q3-去噪扩散概率模型)
- [问题4：CLIP与Dino](#q4-clip与dino)
- [提交作业](#submitting-your-work)

### 设置

开始作业前，请熟悉[推荐的工作流程]({{site.baseurl}}/setup-instructions/#working-remotely-on-google-colaboratory)。你还应该观看下面的Colab操作教程。

<iframe style="display: block; margin: auto;" width="560" height="315" src="https://www.youtube.com/embed/DsGd2e9JNH4" title="YouTube视频播放器" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**注意**：请定期保存你的笔记本（`File -> Save`），以免在你离开作业时Colab虚拟机断开连接，导致进度丢失。

虽然我们不正式支持本地开发，但我们添加了一个<b>requirements.txt</b>文件，你可以用它来设置虚拟环境。

完成所有Colab笔记本（除了`collect_submission.ipynb`）后，请按照[提交说明](#submitting-your-work)进行操作。

### 目标

在本作业中，你将实现语言网络并将其应用于COCO数据集的图像描述生成任务。然后，你将了解自监督学习，以自动学习未标记数据集的视觉表示。接下来，你将实现扩散模型（DDPMs）并将其应用于图像生成。最后，你将探索CLIP和DINO这两种自监督学习方法，它们利用大量未标记数据来学习视觉表示。

本作业的目标如下：

- 理解并实现Transformer网络。将它们与CNN网络结合用于图像描述生成。
- 理解如何利用自监督学习技术来辅助图像分类任务。
- 实现并理解扩散模型（DDPMs），并将其应用于图像生成。
- 实现并理解CLIP和DINO这两种自监督学习方法，它们利用大量未标记数据来学习视觉表示。

**本作业的大部分内容将使用PyTorch。**

### 问题1：使用Transformer进行图像描述生成

笔记本`Transformer_Captioning.ipynb`将指导你实现Transformer模型，并将其应用于COCO数据集的图像描述生成任务。

### 问题2：用于图像分类的自监督学习

在笔记本`Self_Supervised_Learning.ipynb`中，你将学习如何利用自监督预训练在图像分类任务上获得更好的性能。**首次打开笔记本时，请前往`Runtime > Change runtime type`，并将`Hardware accelerator`设置为`GPU`。**

### 问题3：去噪扩散概率模型

在笔记本`DDPM.ipynb`中，你将实现去噪扩散概率模型（DDPM）并将其应用于图像生成。

### 问题4：CLIP与Dino

在笔记本`CLIP_DINO.ipynb`中，你将实现CLIP和DINO这两种自监督学习方法，它们利用大量未标记数据来学习视觉表示。

### 提交作业

**重要提示**：请确保提交的笔记本已经运行过，并且单元格的输出是可见的。

完成所有笔记本并填写好必要的代码后，你需要按照以下说明提交作业：

**1.** 在Colab中打开`collect_submission.ipynb`并执行笔记本中的单元格。

这个笔记本/脚本将：

* 生成一个包含你的代码（`.py`和`.ipynb`）的zip文件，名为`a3_code_submission.zip`。
* 将所有笔记本转换为一个单一的PDF文件，名为`a3_inline_submission.pdf`。

如果此步骤的提交成功，你将看到以下显示信息：

`### 完成！请将a3_code_submission.zip和a3_inline_submission.pdf提交至Gradescope。 ###`

**2.** 将PDF和zip文件提交至Gradescope。

请记住在提交到Gradescope之前，将`a3_code_submission.zip`和`a3_inline_submission.pdf`下载到本地。
