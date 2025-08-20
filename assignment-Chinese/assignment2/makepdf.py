import argparse  # 用于解析命令行参数
import os  # 用于文件和目录操作
import subprocess  # 用于执行外部命令

try:
    from PyPDF2 import PdfMerger  # 导入PDF合并工具

    MERGE = True  # 标记可以进行PDF合并
except ImportError:
    print("未找到PyPDF2库。将不合并PDF文件。")
    MERGE = False  # 标记无法进行PDF合并


def main(files, pdf_name):
    # 构建Jupyter Notebook转换为PDF的命令参数列表
    os_args = [
        "jupyter",
        "nbconvert",
        "--log-level",
        "CRITICAL",  # 只显示严重错误日志
        "--to",
        "pdf",  # 转换为PDF格式
    ]
    for f in files:
        os_args.append(f)  # 添加要转换的笔记本文件
        subprocess.run(os_args)  # 执行转换命令
        os_args.pop()  # 移除最后添加的文件名，为下一个文件做准备
        print("已创建PDF文件：{}。".format(f))
    if MERGE:
        # 生成每个笔记本对应的PDF文件名列表
        pdfs = [f.split(".")[0] + ".pdf" for f in files]
        merger = PdfMerger()  # 创建PDF合并器对象
        for pdf in pdfs:
            merger.append(pdf)  # 添加PDF文件到合并队列
        merger.write(pdf_name)  # 将合并后的内容写入目标PDF文件
        merger.close()  # 关闭合并器
        # 删除单个的PDF文件
        for pdf in pdfs:
            os.remove(pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # 传入明确的笔记本参数，以便提供有序列表并生成有序的PDF
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)  # 笔记本文件列表参数
    parser.add_argument("--pdf_filename", type=str, required=True)  # 输出PDF文件名参数
    args = parser.parse_args()  # 解析命令行参数
    main(args.notebooks, args.pdf_filename)  # 调用主函数执行转换和合并操作
