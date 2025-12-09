import os
import re
import numpy as np
import json
from multiprocessing import Pool, cpu_count
from typing import List, Generator, Tuple, Dict, Any
import argparse
from glob import glob # 用于查找文件
from tokenizer import BPETokenizer



# 全局变量占位符，用于在多进程中存储 Tokenizer 实例
global_tokenizer = None 

# --- 多进程工作函数（保持不变）---

def tokenize_and_save_chunk(task: Tuple[str, str]):
    """
    工作进程函数：对单个文档进行分词并保存为 .bin 文件。
    """
    doc_text, output_path = task
    
    try:
        if global_tokenizer is None:
            raise RuntimeError("Tokenizer not initialized in worker process.")
        
        token_ids: list[int] = global_tokenizer.encode(doc_text)
        tokens_array = np.array(token_ids, dtype=np.uint16)
        tokens_array.tofile(output_path)
        
        # 仅在需要时打印，避免过多输出
        # print(f"Worker {os.getpid()}: ✅ Saved {len(token_ids)} tokens to {output_path}")

    except Exception as e:
        print(f"Worker {os.getpid()}: ❌ Error processing document for {output_path}: {e}")
        raise # 重新抛出异常，让主进程可以捕获


def worker_init(vocab_path: str, merges_path: str, special_tokens: List[str]):
    """
    工作进程初始化函数：在每个子进程中加载 BPETokenizer 实例。
    """
    global global_tokenizer
    try:
        global_tokenizer = BPETokenizer.from_files(
            vocab_filepath=vocab_path, 
            merges_filepath=merges_path, 
            special_tokens=special_tokens
        )
    except Exception as e:
        print(f"Worker process {os.getpid()} failed to initialize tokenizer: {e}")
        # 允许进程初始化失败，但后续任务会失败
        pass # 不要在这里 raise，让 Pool 启动


# --- 流式读取和分割函数（保持不变）---

def read_and_split_stream(input_filepath: str, special_token: str) -> Generator[str, None, None]:
    """
    流式读取大文件，并使用特殊 token 作为分隔符生成完整的文档字符串。
    """
    buffer: List[str] = []
    token_pattern = re.compile(re.escape(special_token))

    print(f"Stream reading file: {input_filepath}")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = token_pattern.search(line)
            
            if match:
                pre_token_text = line[:match.start()]
                current_doc = "".join(buffer) + pre_token_text
                current_doc += match.group(0)
                yield current_doc
                buffer = []
                
                post_token_text = line[match.end():]
                if post_token_text:
                    buffer.append(post_token_text)
            else:
                buffer.append(line)
                
    if buffer:
        yield "".join(buffer)


# --- 新的合并函数 ---

def merge_token_files(output_dir: str, base_filename: str, task_count: int) -> str:
    """
    读取所有分词后的文档小文件，按顺序合并成一个大文件。
    """
    print("\n--- Starting file merging process ---")
    
    # 最终输出文件名：与输入文件同名，后缀为 .bin
    final_output_filename = f"{base_filename}.bin"
    final_output_path = os.path.join(output_dir, final_output_filename)
    
    # 构建所有小文件的路径列表，确保顺序正确
    # 文件名是 base_filename_doc_0.bin, base_filename_doc_1.bin, ...
    input_files = []
    for i in range(task_count):
        input_files.append(os.path.join(output_dir, f"{base_filename}_doc_{i}.bin"))
        
    # 使用 Python 的二进制写入模式和 NumPy 的 fromfile 高效处理
    with open(final_output_path, 'wb') as outfile:
        total_tokens = 0
        
        for i, filepath in enumerate(input_files):
            if not os.path.exists(filepath):
                 print(f"Warning: File not found: {filepath}. Skipping.")
                 continue

            # 1. 使用 np.fromfile 读取 uint16 数据
            token_data = np.fromfile(filepath, dtype=np.uint16)
            total_tokens += len(token_data)
            
            # 2. 将二进制数据直接写入最终文件
            # tobytes() 比 np.concatenate() 内存效率更高
            outfile.write(token_data.tobytes())
            
            # 3. (可选) 清理临时小文件
            os.remove(filepath)
            
            if (i + 1) % 1000 == 0:
                print(f"Merged {i+1}/{task_count} files...")
    
    print(f"--- Merging complete! Total tokens: {total_tokens}. ---")
    print(f"Final binary file saved to: {final_output_path}")
    
    return final_output_path


# --- 主控函数（修改为调用合并）---

def tokenize_file_parallel(input_filepath: str, output_dir: str, vocab_path: str, merges_path: str, special_token: str, num_workers: int):
    """
    主控函数：流式读取文件，分块生成任务，多进程执行分词，然后合并。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    
    # 1. 初始化进程池
    num_workers = min(num_workers, cpu_count())
    print(f"Starting tokenization with {num_workers} worker processes...")

    pool = Pool(
        processes=num_workers, 
        initializer=worker_init, 
        initargs=(vocab_path, merges_path, [special_token])
    )
    
    # 2. 流式生成任务并提交
    document_generator = read_and_split_stream(input_filepath, special_token)
    results = []
    task_count = 0
    
    for doc_text in document_generator:
        output_filename = f"{base_filename}_doc_{task_count}.bin"
        output_path = os.path.join(output_dir, output_filename)
        task = (doc_text, output_path)
        
        res = pool.apply_async(tokenize_and_save_chunk, args=(task,))
        results.append(res)
        task_count += 1
        
        # 周期性打印任务提交进度
        if task_count % 5000 == 0:
             print(f"Submitted {task_count} documents to pool.")

    # 3. 关闭进程池并等待所有任务完成
    pool.close()
    
    # 强制等待所有异步任务完成
    for res in results:
        try:
            res.get()
        except Exception as e:
            # 捕获并报告子进程中的任何异常
            print(f"An error occurred in a worker process: {e}")
            
    pool.join()
    
    print(f"--- All {task_count} documents tokenization tasks finished. ---")
    
    # 4. 执行合并操作
    if task_count > 0:
        merge_token_files(output_dir, base_filename, task_count)
    else:
        print("No documents were processed, skipping merge.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parallel BPE Tokenization Script for Large Files.")
    parser.add_argument("--input_file", required=True, help="Path to the input TXT file (large file friendly).")
    parser.add_argument("--output_dir", default="./tokenized_output", help="Directory to save the final .bin file and temporary files.")
    parser.add_argument("--vocab_path", required=True, help="Path to the vocab.json file.")
    parser.add_argument("--merges_path", required=True, help="Path to the merges.json file.")
    parser.add_argument("--special_token", default="<|endoftext|>", help="The special token used to split documents.")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of worker processes to use (default: CPU count).")
    
    args = parser.parse_args()

    tokenize_file_parallel(
        input_filepath=args.input_file,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_token=args.special_token,
        num_workers=args.workers
    )