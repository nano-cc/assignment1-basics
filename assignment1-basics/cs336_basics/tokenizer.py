# 导入 Self 用于类型提示，List 用于清晰表示列表类型
from typing import  Optional, List, Iterable, Iterator, BinaryIO
from abc import ABC
from dataclasses import dataclass, field
from collections import defaultdict
import os
import time
import heapq
import regex as re
from multiprocessing import Process, Queue, Pool
import logging
import json  # 引入 json 模块

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

# --- BPE 字节-字符映射函数及全局变量 (用户提供) ---


def get_bpe_byte_char_maps():
    """
    创建遵循GPT-2风格的字节-字符双向映射表。
    """
    byte_to_char = {}
    char_to_byte = {}

    # 1. 识别需要映射的161个字节
    bytes_to_map = []
    # 控制字符: 0x00 - 0x1F
    bytes_to_map.extend(range(0x00, 0x20))
    # DEL: 0x7F
    bytes_to_map.append(0x7F)
    # 扩展/非ASCII字节: 0x80 - 0xFF
    bytes_to_map.extend(range(0x80, 0x100))

    # PUA区域的起始码点，用于映射这161个字节
    PUA_START = 0xE000

    # 2. 处理需要映射的161个字节
    for i, byte_val in enumerate(bytes_to_map):
        pua_char = chr(PUA_START + i)
        byte_to_char[byte_val] = pua_char
        char_to_byte[pua_char] = byte_val

    # 3. 处理可打印的ASCII字符 (0x20 - 0x7E)
    for byte_val in range(0x20, 0x7F):
        # 映射到它们自身，即 chr(byte_val)
        char = chr(byte_val)
        byte_to_char[byte_val] = char
        char_to_byte[char] = byte_val

    return byte_to_char, char_to_byte


# 生成全局映射表
BYTE_TO_CHAR, CHAR_TO_BYTE = get_bpe_byte_char_maps()

# --- BPETokenizer 类定义 ---


@dataclass
class LinkedNode:
    value: bytes = None
    token_id: int = 0
    preNode = None
    nextNode = None

    def __str__(self):
        node = self
        nodeStr = b""
        while node is not None:
            nodeStr += node.value
            nodeStr += b"-"
            node = node.nextNode
        return nodeStr.decode("utf-8")[:-1]


@dataclass
class BPETokenizerParam:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]  = None


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):

        # 1. 基础属性初始化
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = set()

        # 2. 处理 Special Tokens: 分配 ID 并更新 self.vocab
        if special_tokens is not None:

            # 找到当前最大的 Token ID
            max_id = max(vocab.keys()) if vocab else -1
            next_id = max_id + 1

            for token_str in special_tokens:
                self.special_tokens.add(token_str)
                token_bytes = token_str.encode("utf-8")

                # 检查 token 是否已经存在（通过检查其 bytes 形式是否在 vocab 的值中），
                # 尽管这不是最快的方法，但能确保安全。如果需要更严格的性能，可以依赖 stoi。
                # 由于 stoi 尚未计算，我们直接假定 special tokens 需要新 ID，并添加到 vocab 中。

                # 如果 token_bytes 已经在某个 ID 下（例如，一个普通词），理论上不应再分配新 ID。
                # 但对于 Special Tokens，标准做法是为其分配最高 ID，即使它恰好与某个词冲突
                # (因为它在 pretokenize 阶段优先级最高)。这里我们简单地分配新 ID。

                self.vocab[next_id] = token_bytes
                next_id += 1

        # 3. 计算 stoi 和 merges_rank (使用包含 special tokens 的完整 self.vocab)
        # stoi占用内存：2.6MB
        self.stoi = self.reverse_vocab(self.vocab)
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

    @staticmethod
    def _convert_chars_to_bytes(text: str, char_to_byte: dict[str, int]) -> bytes:
        """
        使用 char_to_byte 映射表将包含可打印字符的字符串转换为原始字节序列。
        """
        raw_bytes = []
        for char in text:
            if char in char_to_byte:
                raw_bytes.append(char_to_byte[char])
            else:
                logger.warning(
                    f"Character '{char}' not found in CHAR_TO_BYTE map. Skipped.")
                pass
        return bytes(raw_bytes)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]  = None) -> 'BPETokenizer':
        """
        从文件加载词汇表和合并规则。使用全局的 CHAR_TO_BYTE 映射将 JSON 中的
        可打印字符转换回原始字节。
        """
        # 假设全局的 CHAR_TO_BYTE 已经可用
        global CHAR_TO_BYTE

        # 1. 加载 Vocab 文件 (JSON 格式)
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

            vocab: dict[int, bytes] = {}
            for k_str, v_char_seq in vocab_data.items():
                token_id = int(k_str)
                # 使用辅助方法将字符序列转换回原始的 bytes
                token_bytes = cls._convert_chars_to_bytes(
                    v_char_seq, CHAR_TO_BYTE)
                vocab[token_id] = token_bytes

        # 2. 加载 Merges 文件 (JSON 格式)
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f1:
            merges_data = json.load(f1)

            for pair_list in merges_data:
                token1_char_seq, token2_char_seq = pair_list[0], pair_list[1]

                # 使用辅助方法将字符序列转换回原始的 bytes
                token1_bytes = cls._convert_chars_to_bytes(
                    token1_char_seq, CHAR_TO_BYTE)
                token2_bytes = cls._convert_chars_to_bytes(
                    token2_char_seq, CHAR_TO_BYTE)

                merges.append((token1_bytes, token2_bytes))

        # 3. 实例化 BPETokenizer 并返回
        special_tokens_list = special_tokens if special_tokens is not None else []
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens_list)

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
                if second_node is None or pair != (first_node.value, second_node.value):
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
            # 需要找到链表的头节点，即 preIdx 为 -1 且 value 不为 None 的第一个节点
            head_idx = -1
            for i, node in enumerate(word_list):
                if node.preIdx == -1 and node.value is not None:
                    head_idx = i
                    break

            cur_idx = head_idx
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
            text = "Decoding Error"  # 添加一个默认值以防万一

        return text


if __name__ == "__main__":
    tokenizer = BPETokenizer.from_files(vocab_filepath="/home/cong/Projs/assignment1-basics/trained/tokenizer/vocab_size_10000/vocab_tinystories.json",
                                        merges_filepath="/home/cong/Projs/assignment1-basics/trained/tokenizer/vocab_size_10000/merges_tinystories.json",
                                        special_tokens=["<|endoftext|>", "<s>", "</s>"])
    text = "helloworld conglx <s>user</s> root k😊!!!!"
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))
