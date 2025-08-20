---
layout: page
title: 作业1
mathjax: true
permalink: /assignments2025/assignment1/
---

<span style="color:red">本作业截止时间为**2025年4月23日（星期三）太平洋时间晚上11:59**。</span>

包含Colab笔记本的 starter 代码可从[此处下载]({{site.hw_1_colab}})。

- [设置](#setup)
- [目标](#goals)
- [问题1：k近邻分类器](#q1-k近邻分类器)
- [问题2：实现Softmax分类器](#q2-实现softmax分类器)
- [问题3：两层神经网络](#q3-两层神经网络)
- [问题4：更高层次的表示：图像特征](#q4-更高层次的表示-图像特征)
- [问题5：训练全连接网络](#q5-训练全连接网络)
- [提交作业](#submitting-your-work)

### 设置

请观看下面的Colab操作教程，熟悉推荐的工作流程：

<iframe style="display: block; margin: auto;" width="560" height="315" src="https://www.youtube.com/embed/DsGd2e9JNH4" title="YouTube视频播放器" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**注意**：请定期保存你的笔记本（`File -> Save`），以免在你离开作业时Colab虚拟机断开连接，导致进度丢失。

完成所有Colab笔记本（除了`collect_submission.ipynb`）后，请按照[提交说明](#submitting-your-work)进行操作。

### 目标

在本作业中，你将练习搭建一个基于k近邻或SVM/Softmax分类器的简单图像分类流水线。本作业的目标如下：

- 理解基本的**图像分类流水线**和数据驱动方法（训练/预测阶段）。
- 理解训练/验证/测试**分割**以及验证数据在**超参数调优**中的使用。
- 熟练掌握使用numpy编写高效的**向量化**代码。
- 实现并应用k近邻（**kNN**）分类器。
- 实现并应用**Softmax**分类器。
- 实现并应用**两层神经网络**分类器。
- 实现并应用**全连接网络**分类器。
- 理解这些分类器之间的差异和权衡。
- 初步理解使用**更高层次的表示**（而非原始像素，例如颜色直方图、方向梯度直方图（HOG）特征等）带来的性能提升。

### 问题1：k近邻分类器

笔记本**knn.ipynb**将指导你实现kNN分类器。

### 问题2：实现Softmax分类器

笔记本**softmax.ipynb**将指导你实现Softmax分类器。

### 问题3：两层神经网络

笔记本**two_layer_net.ipynb**将指导你实现两层神经网络分类器。

### 问题4：更高层次的表示：图像特征

笔记本**features.ipynb**将探讨使用更高层次的表示相对于使用原始像素值所带来的改进。

### 问题5：训练全连接网络

笔记本**FullyConnectedNets.ipynb**将指导你实现全连接网络。

### 提交作业

**重要提示**：请确保提交的笔记本已经运行过，并且单元格的输出是可见的。

完成所有笔记本并填写好必要的代码后，你需要按照以下说明提交作业：

**1.** 在Colab中打开`collect_submission.ipynb`并执行笔记本中的单元格。

这个笔记本/脚本将：

* 生成一个包含你的代码（`.py`和`.ipynb`）的zip文件，名为`a1_code_submission.zip`。
* 将所有笔记本转换为一个单一的PDF文件。

如果此步骤的提交成功，你将看到以下显示信息：

`### 完成！请将a1_code_submission.zip和a1_inline_submission.pdf提交至Gradescope。 ###`

**2.** 将PDF和zip文件提交至[Gradescope](https://www.gradescope.com/courses/1012166)。

请记住在提交到Gradescope之前，将`a1_code_submission.zip`和`a1_inline_submission.pdf`下载到本地。
