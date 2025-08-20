#!/bin/bash
# 注意：请勿修改此文件——否则可能导致提交不完整
set -euo pipefail          # 遇到错误立即退出；未定义变量时报错；管道出错也视为失败

# 需要提交的 Python 源代码文件列表
CODE=(
	"cs231n/classifiers/k_nearest_neighbor.py"
	"cs231n/classifiers/linear_classifier.py"
	"cs231n/classifiers/softmax.py"
	"cs231n/classifiers/fc_net.py"
	"cs231n/optim.py"
	"cs231n/solver.py"
	"cs231n/layers.py"
)

# 这些 notebook 最好按问题顺序排列，
# 以便生成的 PDF 也按题目顺序呈现
NOTEBOOKS=(
	"knn.ipynb"
	"softmax.ipynb"
	"two_layer_net.ipynb"
	"features.ipynb"
	"FullyConnectedNets.ipynb"
)

# 合并 CODE 与 NOTEBOOKS 数组，得到所有需要检查/打包的文件
FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )

LOCAL_DIR=`pwd`            # 当前工作目录
ASSIGNMENT_NO=1            # 作业编号
ZIP_FILENAME="a1_code_submission.zip"   # 生成的代码压缩包名称
PDF_FILENAME="a1_inline_submission.pdf" # 生成的 PDF 名称

# 颜色输出控制
C_R="\e[31m"   # 红色
C_G="\e[32m"   # 绿色
C_BLD="\e[1m"  # 加粗
C_E="\e[0m"    # 重置颜色

# 检查所有必需文件是否存在；若缺失则退出
for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}必需文件 ${FILE} 未找到，脚本终止。${C_E}"
		exit 0
	fi
done

# 打包代码
echo -e "### 正在压缩文件 ###"
rm -f ${ZIP_FILENAME}  # 若已存在旧压缩包则删除
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") "cs231n/saved" -x "makepdf.py"

# 调用 makepdf.py 把所有 notebook 渲染成单个 PDF
echo -e "### 正在生成 PDF ###"
python makepdf.py --notebooks "${NOTEBOOKS[@]}" --pdf_filename "${PDF_FILENAME}"

# 完成提示
echo -e "### 完成！请将 ${ZIP_FILENAME} 和 ${PDF_FILENAME} 提交到 Gradescope。 ###"
