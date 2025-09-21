# 导入 Self 用于类型提示，List 用于清晰表示列表类型
from typing import Self, Optional, List, Iterable, Iterator, BinaryIO
import tiktoken
from abc import ABC
from dataclasses import dataclass, field
from collections import defaultdict
import os
import time
import heapq
import regex as re
from multiprocessing import Process, Queue, Pool
import logging
# 配置日志输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,  # 设置最低级别为 INFO
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # 日志输出到文件
        logging.StreamHandler()  # 日志输出到控制台
    ]
)
# 获取一个Logger实例 (推荐使用 __name__ 来区分不同模块的Logger)
logger = logging.getLogger(__name__)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    e.g. 实际的分界点在12 20 分块的点为5 10 15 20 第一块和第二块有重叠，返回了相同的分界点，经过排序去重之后结果会少
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def BPE_train_example():

    def merge(word_freq: dict[tuple, int], pair: tuple[bytes, bytes], new_token: bytes):
        # 遍历每个word，进行merge
        new_word_freq = defaultdict(int)
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freq[tuple(new_word)] = freq
        return new_word_freq

        # 在一个简单的例子上完成BPE训练
    text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

    # 1. 初始化词表
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    vocab[0] = "<|endoftext|>".encode("utf-8")
    vocab.update(dict({1+i: bytes([i]) for i in range(256)}))
    # 2. 预分词，使用空白字符分词获取词频率表，词以bytes元组表示
    words = text.split()
    word_freq = defaultdict(int)
    for word in words:
        word_freq[tuple((bytes([byte]) for byte in word.encode("utf-8")))] += 1
    # 3. 进行merge
    num_merges = 6
    near_bytes_freq = defaultdict(int)
    vocab_len = len(vocab)
    for i in range(num_merges):
        # 3.1 遍历全部词，找到相邻两个字节出现次数最多的
        for word, freq in word_freq.items():
            for byte1, byte2 in zip(word, word[1:]):
                # 这里不能将两个bytes拼接之后放进字典，因为不知道是两个bytes是怎样的
                near_bytes_freq[(byte1, byte2)] += freq
        # 3.2 出现次数最多的选择出来作为new_token，加入到词汇表和merges中，随后进行merge
        # max函数key返回一个元组max函数会依次比较元组里面的每个元素对应的值
        max_bytes = max(near_bytes_freq.keys(),
                        key=lambda key: (near_bytes_freq.get(key), key[0]+key[1]))
        new_token = max_bytes[0]+max_bytes[1]
        vocab[vocab_len+i] = new_token
        pair = (max_bytes[0], max_bytes[1])
        merges.append(pair)
        word_freq = merge(word_freq, pair, new_token)
        near_bytes_freq.clear()

    print(f"vocab:{vocab}")
    print(f"merges:{merges}")


@dataclass
class LinkedNode:
    value: bytes = None
    token_id: int = 0
    preNode: Self = None
    nextNode: Self = None

    def __str__(self):
        node = self
        nodeStr = b""
        while node is not None:
            nodeStr += node.value
            nodeStr += b"-"
            node = node.nextNode
        return nodeStr.decode("utf-8")[:-1]


@dataclass
class PairInStr:
    string: str = None
    # count用于计数这个pair在str中出现了多少次，如果次数为0那么需要跳过
    count: int = 0

    def __hash__(self) -> int:
        return self.string.__hash__()

    def __eq__(self, other) -> bool:
        return self.string == other.string


@dataclass
class PairItem:
    """_summary_
    封装了字节对，字节对数量以及字节对比较方式，方便使用堆取出最大的元素
    """
    pair: tuple[bytes, bytes] = None
    count: int = 0

    def __lt__(self, other):
        # 优先比较count数量,由于python中的heapq是小顶堆，需要把判断翻转
        if self.count != other.count:
            return self.count > other.count
        # count数量相同比较两个字节的字典序
        if self.pair[0] != other.pair[0]:
            return self.pair[0] > other.pair[0]
        return self.pair[1] > other.pair[1]

    def __eq__(self, other):
        return self.count == other.count and self.pair[0] == other.pair[0] and self.pair[1] == other.pair[1]


@dataclass
class ProcessMessageBlock:
    # 子进程编号
    idx: int = 0
    # 子进程处理时间
    process_time: float = 0.0
    # 子进程消息
    msg: str = ""
    # 子进程处理区间起点
    start: Optional[int] = None
    # 子进程处理区间终点
    end: Optional[int] = None
    # pretokenize返回多个分块的预分词结果
    pretokenized_chunks: defaultdict = None
    # BPE Train merge过程子进程返回多个分块的merge结果
    merged_chunks: List[defaultdict] = None
    # merge过程中统计相邻字节对出现数据
    near_bytes: Optional[defaultdict] = None
    # merge过程中统计新token相邻字节对数量
    new_token_freq: Optional[defaultdict] = None

    def set_msg(self, msg: str) -> Self:
        """设置消息，并返回自身以便链式调用。"""
        self.msg = msg  # 直接赋值比 __setattr__ 更常用和推荐
        return self

    def set_process_time(self, process_time: float) -> Self:
        """子进程完成任务需要时间，进行性能分析"""
        self.process_time = process_time
        return self

    def set_pretokenized_chunks(self, pretokenized_chunks: defaultdict) -> Self:
        """设置预分词块，并返回自身以便链式调用。"""
        self.pretokenized_chunks = pretokenized_chunks
        return self

    def set_idx(self, idx: int) -> Self:
        """设置子进程编号，并返回自身以便链式调用。"""
        self.idx = idx
        return self

    def set_start(self, start: int) -> Self:
        """设置处理区间起点，并返回自身以便链式调用。"""
        self.start = start
        return self

    def set_end(self, end: int) -> Self:
        """设置处理区间终点，并返回自身以便链式调用。"""
        self.end = end
        return self

    def set_merged_chunks(self, merged_chunks: List[defaultdict]) -> Self:
        """设置合并后的分块，并返回自身以便链式调用。"""
        self.merged_chunks = merged_chunks
        return self

    def set_near_bytes(self, near_bytes: defaultdict) -> Self:
        """设置相邻字节对统计数据，并返回自身以便链式调用。"""
        self.near_bytes = near_bytes
        return self

    def set_new_token_freq(self, new_token_freq: defaultdict) -> Self:
        """设置新token相邻字节对统计数据，并返回自身以便链式调用。"""
        self.new_token_freq = new_token_freq
        return self


def train_chunk_pretokenize(input_path: str, idx: int, blockes: list[int],  special_tokens) -> list[defaultdict]:
    """_summary_

    Args:
        idx (int): 子进程编号
        blockes (list[int]): 子程需要处理的块
        queue (Queue): 主进程与子进程间通信的消息队列，子进程只需要向队列中插入数据
    """
    message_block = ProcessMessageBlock(idx=idx)
    try:
        if blockes is None:
            msg = f"worker-{idx} 需要处理的块为空，退出处理"
            return message_block.set_msg(msg)
        # logger.info(f"worker-{idx} 接收到任务，处理：{blockes}")
        with open(input_path, "rb") as f:
            # 1. 每个进程只需要处理一个块，需要首先使用special_tokens对文本进行分割成多段
            f.seek(blockes[0])
            text_to_split = f.read(blockes[1]-blockes[0]).decode("utf-8")
            blocks = re.split(
                re.escape("|".join(special_tokens)), text_to_split)
            if len(blocks) == 0:
                msg = f"worker-{idx} 需要处理的块为空，退出处理"
                return message_block.set_msg(msg)
            # 2. 分割成多段，对每段进行预分词，每段单独保存
            multi_block_word_freq = defaultdict(int)
            for block in blocks:
                for word in re.finditer(
                        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", block):
                    # 需要加上去掉\r，否则会出现问题
                    multi_block_word_freq[word.group(0).replace('\r', '')] += 1
            msg = "处理成功"
            message_block.set_msg(
                msg).set_pretokenized_chunks(multi_block_word_freq)
    except Exception as e:
        msg = f"处理过程异常:{e}"
        message_block.set_msg(msg)
    return message_block


def BPE_train_pretokenize(pool, input_path: str, special_tokens: list[str], num_processes: int = 4, ) -> tuple[defaultdict[int], defaultdict[LinkedNode]]:
    """
    多进程完成BPE预分词
    """
    # 1. 对文档进行分块，获取到文档的块边界
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # 2. 使用多个进程并行处理，进行预分词
    # 2.1 为每个进程分配任务
    results = pool.starmap(train_chunk_pretokenize, [(input_path,
                                                      i, boundaries[i:min(i+2, len(boundaries))] if i < len(boundaries)-1 else None,  special_tokens) for i in range(num_processes)])
    # 2.3 使用进程池就不需要持续监听消息队列，直接遍历results即可
    all_word_freq = defaultdict(int)
    for message_block in results:
        if message_block.pretokenized_chunks is None:
            msg = f"worker-{message_block.idx} 处理失败,info:{message_block.msg}"
            logger.info(msg)
        else:
            processed_result = message_block.pretokenized_chunks
            for word in processed_result:
                all_word_freq[word] += processed_result[word]
    # 2.4 从每个进程拿到的是str以及对应的出现频率，上面已经汇总了结果，下面需要建立起对应的双向链表方便后续merge
    word_list = defaultdict(LinkedNode)
    for word, _ in all_word_freq.items():
        preNode = None
        for byte in word.encode("utf-8"):
            node = LinkedNode()
            node.value = bytes([byte])
            node.preNode = preNode
            if preNode is None:
                word_list[word] = node
            else:
                preNode.nextNode = node
            preNode = node
    return all_word_freq, word_list


def train_BPE(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """_summary_

    Args:
        input_path (str | os.PathLike): 数据文件路径
        vocab_size (int): 词表最大大小
        special_tokens (list[str]): 特殊tokens

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: tokenizer参数，分别为词表和merges
    """
    def updatePair(pair, delta, string):
        """_summary_

        辅助函数，修改pair的count以及出现的字符串 
        """
        near_bytes_count[pair] += delta
        if delta < 0:
            # 无论如何，只要 delta < 0，这个 string 就不再包含这个 pair
            # 所以从 set 中移除它
            # 这里有问题，因为一个字符串中可能出现多次pair，因此不能直接将字符串丢弃
            pair_exist_str[pair][string] -= 1  # 这里已经做了
        if near_bytes_count[pair] <= 0:
            # 如果频率为0或负数，说明这个 pair 已经不再有效
            # 此时应该从 near_bytes_count 和 pair_exist_str 中彻底移除
            near_bytes_count.pop(pair, None)  # 使用 .pop(key, None) 防止 KeyErorr
            pair_exist_str.pop(pair, None)  # 同时移除 pair_exist_str 中的对应项
        elif delta > 0:
            # 只有在 delta > 0 且 pair 频率仍然有效时，才添加到 set
            pair_exist_str[pair][string] += 1
        # 日志记录
        # logger.info(
        #     f"Updated pair {pair}: delta={delta}, pre_count={pre_count},new_count={near_bytes_count.get(pair, 0)}, string={string},string_freq={all_word_freq[string]}")

    start_time = time.time()
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    # 1. 初始化词表,为了和测试用例相符，需要调换顺序，先放256个bytes，再放特殊token
    vocab.update({i: bytes([i]) for i in range(256)})
    vocab_len = len(vocab)
    vocab.update({vocab_len+idx: special_token.encode("utf-8")
                 for idx, special_token in enumerate(special_tokens)})
    # 使用进程池避免重复创建进程开销，创建进程池
    # 在这里创建进程池，它只会被创建一次
    num_processes = 4
    with Pool(
        processes=num_processes,
    ) as shared_pool:  # 将进程池命名为 shared_pool
        # 2. 预分词，预分词结果为所有单词的频率以及每个单词对应的双向链表
        start_time = time.time()
        all_word_freq, word_list = BPE_train_pretokenize(
            shared_pool, input_path, special_tokens, num_processes)
        end_time = time.time()
        logger.info(
            f"-----预分词时间：{end_time-start_time}，文档块数：{len(all_word_freq)}----")
        # 3. 统计相邻字节对频率，需要遍历每个单词对应的双向链表，同时需要统计每个字节对出现的单词【关键优化】
        start_time = time.time()
        near_bytes_count = defaultdict(int)
        pair_exist_str = defaultdict(lambda: defaultdict(int))
        for word, head in word_list.items():
            while head.nextNode is not None:
                pair = (head.value, head.nextNode.value)
                near_bytes_count[pair] += all_word_freq[word]
                pair_str_count = pair_exist_str[pair][word]
                pair_exist_str[pair][word] = pair_str_count+1
                head = head.nextNode
        end_time = time.time()
        logger.info(
            f"-----相邻字节对统计时间：{end_time-start_time}----")
        # 4. 进行merge
        # 4.1 建堆，使用自定义类封装pair和比较的逻辑，使用堆取出数量最多的pair，注意取出的pair可能是过时的，需要判断
        start_time = time.time()
        heap = []
        for pair, freq in near_bytes_count.items():
            item = PairItem(pair=pair, count=freq)
            heapq.heappush(heap, item)
        end_time = time.time()
        logger.info(
            f"-----建堆时间：{end_time-start_time}----")
        vocab_len = len(vocab)
        nums_to_merge = vocab_size - vocab_len
        start_time = time.time()
        idx = 0
        while idx < nums_to_merge:
            item: PairItem = heapq.heappop(heap)
            pair = item.pair
            # 堆中的数据可能会过时，因此取出的时候需要进行比对
            if item.count != near_bytes_count[pair]:
                continue
            # 没过期就进行merge
            # 4.2 针对相关的单词遍历双向链表进行merge，merge过程中修改相邻字节对数量，字节对出现字符串，将
            # 出现修改的pair重新放进堆中
            new_token = pair[0]+pair[1]
            merges.append(pair)
            vocab[vocab_len+idx] = new_token
            related_strs = list(pair_exist_str[pair].keys())
            for string in related_strs:
                # 说明pair在该字符串中没有出现了
                if pair_exist_str[pair][string] == 0:
                    continue
                node: LinkedNode = word_list[string]
                str_freq = all_word_freq[string]
                while node is not None and node.nextNode is not None:
                    nextNode = node.nextNode
                    if node.value == pair[0] and nextNode.value == pair[1]:
                        node.value = new_token
                        node.nextNode = nextNode.nextNode
                        # 需要修改的内容：1. 相邻字节对频率统计表 2. 字节对出现字符串统计表 3. 修改之后重新放进堆中
                        updatePair(pair, -str_freq, string)
                        if node.preNode is not None:
                            prePair = (node.preNode.value, pair[0])
                            newPair = (node.preNode.value, new_token)
                            updatePair(prePair, -str_freq, string)
                            updatePair(newPair, str_freq, string)
                            heapq.heappush(heap, PairItem(
                                prePair, near_bytes_count[prePair]))
                            heapq.heappush(heap, PairItem(
                                newPair, near_bytes_count[newPair]))
                        if nextNode.nextNode is not None:
                            nextNode.nextNode.preNode = node
                            nexPair = (pair[1], nextNode.nextNode.value)
                            newPair = (new_token, nextNode.nextNode.value)
                            updatePair(nexPair, -str_freq, string)
                            updatePair(newPair, str_freq, string)
                            heapq.heappush(heap, PairItem(
                                nexPair, near_bytes_count[nexPair]))
                            heapq.heappush(heap, PairItem(
                                newPair, near_bytes_count[newPair]))
                        continue
                    node = node.nextNode
            near_bytes_count.pop(pair, 0)
            pair_exist_str.pop(pair, None)
            idx += 1
    end_time = time.time()
    logger.info(
        f"-----merge时间：{end_time-start_time}----")
    return (vocab, merges)


@dataclass
class BPETokenizerParam:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is not None:
            self.special_tokens = set(special_tokens)
        else:
            self.special_tokens = set()
        # stoi占用内存：2.6MB
        self.stoi = self.reverse_vocab(vocab)
        self.merges_rank = self.compute_merges_rank()

    def reverse_vocab(self, vocab):
        stoi = defaultdict(int)
        for key, value in vocab.items():
            stoi[value] = key
        return stoi

    def compute_merges_rank(self):
        # 预先计算merges的rank，方便后续查询
        merges_rank = defaultdict(lambda:  defaultdict(int))
        for idx, pair in enumerate(self.merges):
            merges_rank[pair[0]][pair[1]] = idx+1
        return merges_rank

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """_summary_

        Args:
            vocab_filepath (str): _description_
            merges_filepath (str): _description_
            special_tokens (list[str] | None, optional): _description_. Defaults to None.
        由于不知道文件格式就当成json文件去读取
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            import json
            vocab_data = json.load(f)
            # 确保vocab的键是整数（JSON会将其转换为字符串）
            cls.vocab = {int(k): v.encode('utf-8') if isinstance(v, str) else v
                         for k, v in vocab_data.items()}
        cls.stoi = cls.reverse_vocab(cls.vocab)

        cls.merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f1:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pairs = line.split(" ")
                # 将两个token转换为bytes
                token1, token2 = pairs[0].encode(
                    'utf-8'), pairs[1].encode('utf-8')
                cls.merges.append((token1, token2))

        cls.merges_rank = cls.compute_merges_rank()
        cls.special_tokens = set(special_tokens)
        return cls

    def split_by_special_tokens(self, text: str) -> Iterator[str]:
        if not self.special_tokens:
            # 如果没有特殊 token，直接对整个文本进行预分词
            yield from self.pretokenize(text)
            return

        # 1. 构造正则表达式，用于匹配特殊 token
        sorted_tokens = sorted(list(self.special_tokens),
                               key=len, reverse=True)
        # 注意这里用的是非捕获分组 (?:...)
        pattern = re.compile(
            '(?:' + '|'.join(map(re.escape, sorted_tokens)) + ')')

        # 2. 使用 re.finditer 迭代查找特殊 token
        last_end = 0
        for match in pattern.finditer(text):
            # 获取特殊 token 的起始和结束位置
            start, end = match.span()
            # 获取特殊 token 之前的文本块
            if start > last_end:
                yield from self.pretokenize(text[last_end:start])

            # 返回特殊 token 本身
            yield match.group(0)
            last_end = end

        # 3. 处理最后一个特殊 token 之后的剩余文本
        if last_end < len(text):
            yield from self.pretokenize(text[last_end:])

    def pretokenize(self, text: str) -> Iterator[str]:
        for word_match in re.finditer(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", text
        ):
            yield word_match.group(0).replace('\r', '')

    def encode(self, text: str) -> list[int]:
        # return self.slow_encode(text)
        # return self.greedy_match_encode(text)
        return self.faster_encode(text)

    def greedy_match_encode(self, text: str) -> list[int]:
        # 这种实现无法通过测试，经过研究原因在于如果对于unicode字符' 🙃'，其转换成字节数组为b' \xf0\x9f\x99\x83'
        # 其中 \xf0不在字典中，但是 \xf0\x9f在字典中，但是到\xf0已经匹配不到了
        # 其实除了上面的问题，贪婪方法进行encode还是不能通过，因为顺序和tiktoken有所不同，例如' Leland'，贪婪匹配应该分成' Le'和'land'，但是
        # 实际上是分成了' L'和'eland'
        blocks_list = self.split_by_special_tokens(text)
        result = []
        for block in blocks_list:
            if block == "":
                continue
            if block in self.special_tokens:
                result.append(self.stoi[block.encode("utf-8")])
                continue
            # 对单词进行贪婪匹配，尽可能匹配最长的
            block_bytes = block.encode("utf-8")
            # 这里匹配的逻辑需要修改，例如abc是一个词，但是合并的顺序是bc abc，这样从前向后找ab不在词表中但是abc是在词表中的
            # 改成从后向前去匹配尽可能长的子串
            start = 0
            end = len(block_bytes)
            while start < len(block_bytes):
                # 外层循环遍历起点
                while end > start:
                    # 内层循环遍历终点
                    sub_str = block_bytes[start:end]
                    if sub_str in self.stoi.keys():
                        result.append(self.stoi[sub_str])
                        if end == len(block_bytes):
                            # end如果是词的末尾，这个词就处理结束了
                            start = len(block_bytes)
                            break
                        else:
                            start = end
                            end = len(block_bytes)
                            break
                    else:
                        end -= 1
        return result

    def slow_encode(self, text: str) -> list[int]:
        blocks_list = self.split_by_special_tokens(text)
        # 3. 预分词之后对每个block转换成bytes的形式，便于merge
        # 使用一个字典存放所有单词，避免重复运算
        word_list = defaultdict(LinkedNode)
        for word in blocks_list:
            if word in self.special_tokens:
                continue
            node: LinkedNode = word_list[word]
            # 对每个词建立一个唯一的链表
            for byte in word.encode("utf-8"):
                # 遍历字节数组得到的byte的值为一个int值，需要转换成字节数组再进行后续操作
                node.value = bytes([byte])
                node.token_id = self.stoi[bytes([byte])]
                newNode = LinkedNode()
                node.nextNode = newNode
                newNode.preNode = node
                node = newNode
            node.preNode.nextNode = None
        # 4. 完成merge，需要遍历merges，每个merge对对所有单词进行merge
        # 当前算法低效的点在于对于每个merge对遍历了全部的单词，不知道后续是不是有更高效的做法或者有更好的数据结构
        # a.对于每个pair单词中不一定有，可以通过拼接之后判断word中是否存在来判断
        for pair in self.merges:
            for word, head in word_list.items():
                node: LinkedNode = head
                while node is not None and node.nextNode is not None:
                    nextNode = node.nextNode
                    if node.value == pair[0] and nextNode.value == pair[1]:
                        # 需要将这对进行merge
                        newValue = pair[0]+pair[1]
                        # 这里不小心把nextNode赋值给了value(python不检查类型)
                        node.value = newValue
                        node.token_id = self.stoi[newValue]
                        node.nextNode = nextNode.nextNode
                        if nextNode.nextNode is not None:
                            nextNode.nextNode.preNode = node
                        continue
                    node = node.nextNode
        # 5. merge完成之后收集结果到列表中
        encoded_result = []
        for word in blocks_list:
            # 特殊token直接添加
            if word in self.special_tokens:
                encoded_result.append(self.stoi[word.encode("utf-8")])
            else:
                head: LinkedNode = word_list[word]
                while head is not None:
                    encoded_result.append(head.token_id)
                    head = head.nextNode
        return encoded_result

    def faster_encode(self, text: str) -> list[int]:

        # 内部类，存放进列表中
        @dataclass
        class ListNode:
            # 约定value如果为None说明该节点已经被删除了（惰性删除）
            value: bytes = None
            # -1表示是链表的最后一个节点
            nextIdx: int = -1
            # -1表示是链表的第一个节点
            preIdx: int = -1

            def __str__(self) -> str:
                return str({
                    "value": self.value,
                    "nextIdx": self.nextIdx,
                    "preIdx": self.preIdx
                })

        @dataclass
        class HeapItem:
            rank: int = 0
            pair: tuple[bytes, bytes] = None
            # idx表示pair首个字节的下标
            idx: int = -1

            def __lt__(self, other):
                # rank小的先合并
                if self.rank != other.rank:
                    return self.rank < other.rank
                # rank相同index小的先合并
                return self.idx < other.idx

            def __eq__(self, other):
                return self.rank == other.rank and self.pair == other.pair and self.idx == other.idx

        # 1. 首先获取按特殊token进行分块以及预分词之后的结果,已经修改为迭代器方式，避免存储全部结果
        blocks_list = self.split_by_special_tokens(text)
        result = []
        for block in blocks_list:
            # 2. 处理每个词，对于非特殊token这次不采用建立链表的方式，而是采用数组中存放链表的方式
            if block in self.special_tokens:
                result.append(self.stoi[block.encode("utf-8")])
                continue
            # 3. 需要遍历这个词，一方面建立成链表（以数组方式存储），另一方面建立优先级队列
            word_list: list[ListNode] = []
            priority_queue: list[HeapItem] = []
            for idx, byte in enumerate(block.encode("utf-8")):
                node = ListNode(bytes([byte]), -1, -1)
                if idx > 0:
                    node.preIdx = idx-1
                    word_list[idx-1].nextIdx = idx
                    pair = (word_list[idx-1].value, node.value)
                    rank = self.merges_rank[pair[0]][pair[1]]
                    item = HeapItem(rank, pair, idx-1)
                    priority_queue.append(item)
                word_list.append(node)
            heapq.heapify(priority_queue)
            # 4. 开始迭代merge，同样，堆中的元素使用惰性更新，pop出来的元素需要检查是否有效
            while len(priority_queue) != 0:
                item: HeapItem = heapq.heappop(priority_queue)
                # 4.1 检查item是否有效,首先rank=0说明没有这个merge对
                if item.rank == 0:
                    continue
                pair = item.pair
                first_node = word_list[item.idx]
                second_node = word_list[first_node.nextIdx] if first_node.nextIdx != -1 else None
                # 4.2 检查item对应位置的pair是否还是一样，如果不一样说明发生了合并，item已经过期
                if pair != (first_node.value, second_node.value if second_node is not None else None):
                    continue
                # 4.3 item有效继续更新
                # 4.3.1 首先修改链表，合并value，修改前后链接
                first_node.value += second_node.value
                second_node.value = None
                first_node.nextIdx = second_node.nextIdx
                if second_node.nextIdx != -1:
                    word_list[second_node.nextIdx].preIdx = item.idx
                # 4.3.2 把新出现的pair添加到优先级队列中
                if first_node.preIdx != -1:
                    pre_pair_idx = first_node.preIdx
                    pre_pair = (
                        word_list[pre_pair_idx].value, first_node.value)
                    rank = self.merges_rank[pre_pair[0]][pre_pair[1]]
                    new_item = HeapItem(rank, pre_pair, pre_pair_idx)
                    heapq.heappush(priority_queue, new_item)
                if first_node.nextIdx != -1:
                    next_pair = (first_node.value,
                                 word_list[first_node.nextIdx].value)
                    rank = self.merges_rank[next_pair[0]][next_pair[1]]
                    new_item = HeapItem(rank, next_pair, item.idx)
                    heapq.heappush(priority_queue, new_item)
            # 5. 从第一个节点找到最后一个节点，把每个token对应的token id加入到最终结果中
            cur_idx = 0
            while cur_idx != -1:
                node = word_list[cur_idx]
                result.append(self.stoi[node.value])
                cur_idx = node.nextIdx
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # encode_iterable思路比较简单，就是一个个单词去编码
        # iterable里面传入的字符串实际上是一段文本
        for text in iterable:
            result = self.encode(text)
            for token_id in result:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        # 1. 将token IDs转换为字节序列
        byte_sequence = b''
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
            else:
                # 处理未知token ID，可以添加替换字节或抛出错误
                continue

        # 2. 将字节序列解码为字符串，自动处理错误
        try:
            text = byte_sequence.decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(str(e))

        return text


if __name__ == "__main__":
    file_path = "/home/cong/Projs/assignment1-basics/cs336_basics/address.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    blank_str = ""
    # unicode_chr = "😊"
    unicode_chr = "🙃"
    ascii_str = "Hello,how are you?"
    from tests.test_tokenizer import VOCAB_PATH, MERGES_PATH, get_tokenizer_from_vocab_merges_path, test_address_matches_tiktoken, test_unicode_string_matches_tiktoken, test_german_matches_tiktoken
    # tokenizer = get_tokenizer_from_vocab_merges_path(
    #     vocab_path=VOCAB_PATH,
    #     merges_path=MERGES_PATH,
    #     special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    # )
    # test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # token_ids = tokenizer.encode(test_string)
    # decoded = tokenizer.decode(token_ids)
    # test_overlapping_special_tokens()
    # test_unicode_string_matches_tiktoken()
    test_german_matches_tiktoken()
