import argparse  # 用于解析命令行参数
import os  # 用于文件和目录操作
import subprocess  # 用于执行外部命令

try:
    from PyPDF2 import PdfMerger  # 导入PDF合并工具

    MERGE = True  # 标记可以进行PDF合并
except ImportError:
    # 如果未安装PyPDF2，打印提示信息并禁用合并功能
    print("未找到PyPDF2库。将保留单独的PDF文件，不进行合并。")
    MERGE = False


def main(files, pdf_name):
    # 构建nbconvert命令的基础参数列表
    os_args = [
        "jupyter",
        "nbconvert",
        "--log-level",
        "CRITICAL",  # 只显示严重错误日志
        "--to",
        "pdf",  # 转换目标格式为PDF
    ]
    # 遍历每个输入文件，执行转换命令
    for f in files:
        os_args.append(f)  # 添加当前文件到命令参数
        subprocess.run(os_args)  # 执行命令将笔记本转换为PDF
        print(f"已创建PDF文件：{f}。")
    # 如果可以合并PDF
    if MERGE:
        # 生成每个输入文件对应的PDF文件名（去掉原扩展名，加上.pdf）
        pdfs = [f.split(".")[0] + ".pdf" for f in files]
        merger = PdfMerger()  # 创建PDF合并器实例
        # 将每个PDF文件添加到合并器
        for pdf in pdfs:
            merger.append(pdf)
        # 将合并后的内容写入目标PDF文件
        merger.write(pdf_name)
        merger.close()  # 关闭合并器
        # 删除转换过程中生成的单个PDF文件
        for pdf in pdfs:
            os.remove(pdf)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数：--notebooks，接收一个或多个笔记本文件名（有序列表，用于生成有序PDF）
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    # 添加参数：--pdf_filename，指定合并后PDF的文件名（必需）
    parser.add_argument("--pdf_filename", type=str, required=True)
    args = parser.parse_args()  # 解析命令行参数
    # 调用主函数，传入笔记本文件列表和目标PDF文件名
    main(args.notebooks, args.pdf_filename)
