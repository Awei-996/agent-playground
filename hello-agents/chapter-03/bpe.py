import re, collections

"""
----第一次统计，元数据是：{'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
    # 第一组'h u g </w>' 贡献：('h','u') +1，('u','g') +1，('g','</w>') +1
    # 第二组'p u g </w>' 贡献：('p','u') +1，('u','g') +1，('g','</w>') +1
    # 第三组'p u n </w>' 贡献：('p','u') +1，('u','n') +1，('n','</w>') +1
    # 第四组'b u n </w>' 贡献：('b','u') +1，('u','n') +1，('n','</w>') +1
    # 最终pairs = {('h','u'): 1, ('u','g'): 2, ('g','</w>'): 2, ('p','u'): 2, ('u','n'): 2, ('n','</w>'): 2, ('b','u'): 1}
----第二次统计，元数据是：{'h ug </w>': 1, 'p ug </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
    # 第一组'h ug </w>' 贡献：('h','ug') +1，('ug','</w>') +1
    # 第二组'p ug </w>' 贡献：('p','ug') +1，('ug','</w>') +1
    # 第三组'p u n </w>' 贡献：('p','u') +1，('u','n') +1，('n','</w>') +1
    # 第四组'b u n </w>' 贡献：('b','u') +1，('u','n') +1，('n','</w>') +1
    # 最终pairs = {('h','ug'): 1, ('ug','</w>'): 2, ('p','ug'): 1, ('u','n'): 2, ('n','</w>'): 2, ('b','u'): 1,('p','u'): 1}
----第三次统计，元数据是：{'h ug</w>': 1, 'p ug</w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
    # 第一组'h ug</w>' 贡献：('h','ug</w>') +1
    # 第二组'p ug</w>' 贡献：('p','ug</w>') +1
    # 第三组'p u n </w>' 贡献：('p','u') +1，('u','n') +1，('n','</w>') +1
    # 第四组'b u n </w>' 贡献：('b','u') +1，('u','n') +1，('n','</w>') +1
    # 最终pairs = {('h','ug</w>'): 1, ('p','ug</w>'): 1, ('u','n'): 2, ('n','</w>'): 2, ('b','u'): 1,('p','u'): 1}
----第四次统计，元数据是：{'h ug</w>': 1, 'p ug</w>': 1, 'p un </w>': 1, 'b un </w>': 1}
    # 第一组'h ug</w>' 贡献：('h','ug</w>') +1
    # 第二组'p ug</w>' 贡献：('p','ug</w>') +1
    # 第三组'p un </w>' 贡献：('p','un') +1，('un','</w>') +1
    # 第四组'b un </w>' 贡献：('b','un') +1，('un','</w>') +1
    # 最终pairs = {('h','ug</w>'): 1, ('p','ug</w>'): 1, ('p','un'): 1, ('un','</w>'): 2, ('b','un'): 1}
最终结果：元数据是：{'h ug</w>': 1, 'p ug</w>': 1, 'p un</w>': 1, 'b un</w>': 1}
"""

def get_stats(vocab):
    """统计词元对频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            # 累计词元对频率
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """合并词元对"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# 准备语料库，每个词末尾加上</w>表示结束，并切分好字符
vocab = {'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
num_merges = 4 # 设置合并次数

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    # 找到频率最高的词元对
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"第{i+1}次合并: {best} -> {''.join(best)}")
    print(f"新词表（部分）: {list(vocab.keys())}")
    print("-" * 20)