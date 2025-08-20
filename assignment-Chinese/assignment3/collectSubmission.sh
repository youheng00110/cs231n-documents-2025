#!/bin/bash
# 启用严格模式：遇到错误、未定义变量或管道错误时终止脚本
set -euo pipefail

# 需要打包的代码文件列表
CODE=(
	# Transformer实现文件
	"cs231n/transformer_layers.py"
	"cs231n/classifiers/transformer.py"
	"cs231n/captioning_solver_transformer.py"

	# 自监督学习实现文件
	"cs231n/simclr/contrastive_loss.py"
	"cs231n/simclr/data_utils.py"
	"cs231n/simclr/utils.py"
	"cs231n/simclr/model.py"

	# DDPM实现文件
	"cs231n/unet.py"
	"cs231n/gaussian_diffusion.py"
	"cs231n/ddpm_trainer.py"
	"cs231n/emoji_dataset.py"
	"cs231n/clip_dino.py"
)

# 需要处理的笔记本文件列表
NOTEBOOKS=(
	"Transformer_Captioning.ipynb"
	"Self_Supervised_Learning.ipynb"
	"DDPM.ipynb"
	"CLIP_DINO.ipynb"
)

# 需要转换为PDF的文件列表
PDFS=(
  	"Transformer_Captioning.ipynb"
	"Self_Supervised_Learning.ipynb"
	"DDPM.ipynb"
	"CLIP_DINO.ipynb"
)

# 合并所有需要处理的文件
FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )
# 压缩包文件名
ZIP_FILENAME="a3_code_submission.zip"
# PDF文件名
PDF_FILENAME="a3_inline_submission.pdf"

# 检查所有必要文件是否存在
for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "必需的文件 ${FILE} 未找到，正在退出。"
		exit 0
	fi
done

echo -e "### 正在压缩文件 ###"
# 删除已存在的压缩包（如果有）
rm -f ${ZIP_FILENAME}
# 打包笔记本文件和所有Python文件（排除makepdf.py）
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") -x "makepdf.py"

echo -e "### 正在生成PDF ###"
# 使用Python脚本将笔记本转换为PDF
python makepdf.py --notebooks "${PDFS[@]}" --pdf_filename "${PDF_FILENAME}"

echo -e "### 完成！请将 ${ZIP_FILENAME} 和 ${PDF_FILENAME} 提交到Gradescope。 ###"