import json


def get_bpe_byte_char_maps():
    """
    创建遵循GPT-2风格的字节-字符双向映射表。

    规则：
    1. 可打印ASCII字符 (0x20-0x7E) 映射到自身。
    2. 其他161个字节 (控制字符, DEL, 扩展ASCII) 映射到Unicode PUA区域。
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

# with open("BYTE_TO_CHAR.json", "w", encoding="utf-8") as f:
#     json.dump(BYTE_TO_CHAR, f, ensure_ascii=False, indent=4)

# with open("CHAR_TO_BYTE.json", "w", encoding="utf-8") as f:
#     json.dump(CHAR_TO_BYTE, f, ensure_ascii=False, indent=4)


# with open("/data/lyl/temp/assignment1-basics/trained/tokenizer/vocab_tinystories.json", "r", encoding="utf-8") as f:
#     vocab = json.load(f)
#     for k, v in vocab.items():
#         v: list[int]
#         s: str = ""
#         for byte in v:
#             s += BYTE_TO_CHAR[byte]
#         vocab[k] = s
#     with open("/data/lyl/temp/assignment1-basics/trained/tokenizer/vocab_tinystories_readable.json", "w", encoding="utf-8") as f:
#         json.dump(vocab, f, ensure_ascii=False, indent=4)

# with open("/data/lyl/temp/assignment1-basics/trained/tokenizer/vocab_owt.json", "r", encoding="utf-8") as f:
#     vocab = json.load(f)
#     for k, v in vocab.items():
#         v: list[int]
#         s: str = ""
#         for byte in v:
#             s += BYTE_TO_CHAR[byte]
#         vocab[k] = s
#     with open("/data/lyl/temp/assignment1-basics/trained/tokenizer/vocab_owt_readable.json", "w", encoding="utf-8") as f:
#         json.dump(vocab, f, ensure_ascii=False, indent=4)

# with open("/data/lyl/temp/assignment1-basics/trained/tokenizer/vocab_size_10000/pre_merges_owt.json", "r", encoding="utf-8") as f:
#     merges = json.load(f)
#     new_merges = []
#     for pair in merges:
#         bytes1 = b''
#         bytes2 = b''
#         for byte in pair[0]:
#             bytes1 += BYTE_TO_CHAR[byte].encode("utf-8")
#         for byte in pair[1]:
#             bytes2 += BYTE_TO_CHAR[byte].encode("utf-8")
#         new_merges.append((bytes1.decode(), bytes2.decode()))
#     with open("/data/lyl/temp/assignment1-basics/trained/tokenizer/vocab_size_10000/merges_owt.json", "w", encoding="utf-8") as f1:
#         json.dump(new_merges, f1, ensure_ascii=False, indent=4)


print(CHAR_TO_BYTE[BYTE_TO_CHAR[238]])
print(BYTE_TO_CHAR[238])
