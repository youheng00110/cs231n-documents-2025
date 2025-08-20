---
layout: page
title: 作业2
mathjax: true
permalink: /assignments2025/assignment2/
---

<span style="color:red">本作业截止时间为**2025年5月7日（星期三）太平洋标准时间晚上11:59**。</span>

包含Colab笔记本的 starter 代码可从[此处下载]({{site.hw_2_colab}})。

- [设置](#setup)
- [目标](#goals)
- [问题1：批标准化](#q1-批标准化)
- [问题2：Dropout](#q2-dropout)
- [问题3：卷积神经网络](#q3-卷积神经网络)
- [问题4：基于PyTorch的CIFAR-10任务](#q4-基于pytorch的cifar-10任务)
- [问题5：使用 vanilla RNN 进行图像 caption 生成](#q5-使用-vanilla-rnn-进行图像-caption-生成)
- [提交作业](#submitting-your-work)

### 设置

开始作业前，请熟悉[推荐的工作流程]({{site.baseurl}}/setup-instructions/#working-remotely-on-google-colaboratory)。你还应该观看下面的Colab操作教程。

<iframe style="display: block; margin: auto;" width="560" height="315" src="https://www.youtube.com/embed/DsGd2e9JNH4" title="YouTube视频播放器" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**注意**：请定期保存你的笔记本（`File -> Save`），以免在你离开作业时Colab虚拟机断开连接，导致进度丢失。

虽然我们不正式支持本地开发，但我们添加了一个<b>requirements.txt</b>文件，你可以用它来设置虚拟环境。

完成所有Colab笔记本（除了`collect_submission.ipynb`）后，请按照[提交说明](#submitting-your-work)进行操作。

### 目标

在本作业中，你将练习编写反向传播代码，以及训练神经网络和卷积神经网络。本作业的目标如下：

- 实现**批标准化**和**层标准化**以训练深度网络。
- 实现**Dropout**来正则化网络。
- 理解**卷积神经网络**的架构，并练习训练它们。
- 获得使用主流深度学习框架**PyTorch**的经验。
- 理解并实现RNN网络。将它们与CNN网络结合用于图像caption生成。


### 问题1：批标准化

在笔记本`BatchNormalization.ipynb`中，你将实现批标准化，并使用它来训练深度全连接网络。

### 问题2：Dropout

笔记本`Dropout.ipynb`将帮助你实现Dropout，并探索其对模型泛化的影响。

### 问题3：卷积神经网络

在笔记本`ConvolutionalNetworks.ipynb`中，你将实现几个常用于卷积网络中的新层。

### 问题4：基于PyTorch的CIFAR-10任务

在这一部分，你将使用PyTorch——一个流行且强大的深度学习框架。

打开`PyTorch.ipynb`。在那里，你将学习该框架的工作原理，最终目标是设计一个卷积网络并在CIFAR-10上训练，以获得尽可能好的性能。

### 问题5：使用 vanilla RNN 进行图像 caption 生成
笔记本`RNN_Captioning_pytorch.ipynb`将指导你实现vanilla循环神经网络，并将其应用于COCO数据集的图像caption生成任务。

### 提交作业

**重要提示**：请确保提交的笔记本已经运行过，并且单元格的输出是可见的。

完成所有笔记本并填写好必要的代码后，你需要按照以下说明提交作业：

**1.** 在Colab中打开`collect_submission.ipynb`并执行笔记本中的单元格。

这个笔记本/脚本将：

* 生成一个包含你的代码（`.py`和`.ipynb`）的zip文件，名为`a2_code_submission.zip`。
* 将所有笔记本转换为一个单一的PDF文件。

如果此步骤的提交成功，你将看到以下显示信息：

`### 完成！请将a2_code_submission.zip和a2_inline_submission.pdf提交至Gradescope。 ###`

**_注意：当你完成所有笔记本后，请确保你最近的内核执行顺序是按时间顺序的，否则可能会给Gradescope自动评分器带来问题。如果不是这样，你应该重启该笔记本的内核，并使用“运行时”菜单中的“重启并运行全部”选项重新运行笔记本中的所有单元格。_**

**2.** 将PDF和zip文件提交至[Gradescope](https://www.gradescope.com/courses/1012166)。

请记住在提交到Gradescope之前，将`a2_code_submission.zip`和`a2_inline_submission.pdf`下载到本地。
