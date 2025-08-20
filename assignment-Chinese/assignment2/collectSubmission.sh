#!/bin/bash
# 注意：不要编辑此文件——可能导致提交不完整
set -euo pipefail  # 启用严格错误检查（遇到错误立即退出）

# 需要包含的代码文件列表
CODE=(
	"cs231n/layers.py"
	"cs231n/classifiers/fc_net.py"
	"cs231n/optim.py"
	"cs231n/solver.py"
	"cs231n/classifiers/cnn.py"
	"cs231n/im2col_cython.pyx"
  	"cs231n/classifiers/rnn_pytorch.py"
)

# 这些笔记本理想情况下应按问题顺序排列，
# 以便生成的pdf按问题顺序展示
NOTEBOOKS=(
	"BatchNormalization.ipynb"
	"Dropout.ipynb"
	"ConvolutionalNetworks.ipynb"
	"PyTorch.ipynb"
  	"RNN_Captioning_pytorch.ipynb"
)
FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )  # 合并所有需要检查的文件

LOCAL_DIR=`pwd`  # 当前工作目录
ASSIGNMENT_NO=1  # 作业编号
ZIP_FILENAME="a2_code_submission.zip"  # 代码提交压缩包名称
PDF_FILENAME="a2_inline_submission.pdf"  # 笔记本导出的PDF名称

# 终端输出颜色配置
C_R="\e[31m"  # 红色
C_G="\e[32m"  # 绿色
C_BLD="\e[1m"  # 加粗
C_E="\e[0m"    # 重置颜色

# 检查所有必需文件是否存在
for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}必需文件 ${FILE} 未找到，退出。${C_E}"
		exit 0
	fi
done

echo -e "### 正在压缩文件 !!! ###"
rm -f ${ZIP_FILENAME}  # 删除已存在的压缩包（如果有）
# 将笔记本、所有Python文件、Cython文件和saved目录压缩（排除makepdf.py）
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . \( -name '*.py' -o -name '*.pyx' \)) "cs231n/saved" -x "makepdf.py"

echo -e "### 正在生成PDF ###"
# 调用Python脚本将笔记本导出为PDF
python makepdf.py --notebooks "${NOTEBOOKS[@]}" --pdf_filename "${PDF_FILENAME}"

echo -e "### 完成！请将 ${ZIP_FILENAME} 和 ${PDF_FILENAME} 提交到Gradescope。 ###"
