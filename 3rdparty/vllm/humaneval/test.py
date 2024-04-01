#1
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:  # 修改此处的判断条件
                return True
    return False
'''
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False


print(has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3))
print(has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05))
print(has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95))
print(has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8))
print(has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1))
print(has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0))
print(has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5))
------------------------True
'''
#2
def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    stack = []
    start = 0
    for i, char in enumerate(paren_string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                result.append(paren_string[start:i+1])
    return result
'''
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
    assert candidate('(()(())((())))') == ['(()(())((())))']
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']

print(separate_paren_groups('(()()) ((())) () ((())()())'))
print(separate_paren_groups('() (()) ((())) (((())))'))
print(separate_paren_groups('(()(())((())))'))
print(separate_paren_groups('( ) (( )) (( )( ))'))

['()', '()', '(()())', '()', '(())', '((()))', '()', '()', '(())', '()', '()', '((())()())']
['()', '()', '(())', '()', '(())', '((()))', '()', '(())', '((()))', '(((())))']
['()', '()', '(())', '()', '(())', '((()))', '(()(())((())))']
['( )', '( )', '(( ))', '( )', '( )', '(( )( ))']
'''
#3
def truncate_number(number: float) -> float:
    return number - int(number)


'''
def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-6
    assert abs(candidate(123.456) - 0.456) < 1e-6

print(truncate_number(3.5))
print(truncate_number(1.33))
print(truncate_number(123.456))

---------------------True
0.5
0.33000000000000007
0.45600000000000307
'''
#4
def below_zero(operations: List[int]) -> bool:
    balance = 0
    for op in operations:
        balance -= op
        if balance < 0:
            return True
    return False

'''
def check(candidate):
    assert candidate([]) == False
    assert candidate([1, 2, -3, 1, 2, -3]) == False
    assert candidate([1, 2, -4, 5, 6]) == True
    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False
    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True

print(below_zero([]))
print(below_zero([1, 2, -3, 1, 2, -3]))
print(below_zero([1, 2, -4, 5, 6]))
print(below_zero([1, -1, 2, -2, 5, -5, 4, -4]))
print(below_zero([1, -1, 2, -2, 5, -5, 4, -5]))
print(below_zero([1, -2, 2, -2, 5, -5, 4, -4]))
False
True
True
True
True
True
'''
#5
def mean_absolute_deviation(numbers: List[float]) -> float:
        # 计算平均值
    mean = sum(numbers) / len(numbers)
    # 计算平均绝对偏差
    mad = sum(abs(x - mean) for x in numbers) / len(numbers)
    return mad 
'''
def check(candidate):
    # 检查第一个列表
    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6

    # 检查第二个列表
    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6

    # 检查第三个列表
    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6

print(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0)
print(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0)
print(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0)
0.0
0.0
0.0
----------------------True
'''
#6
def intersperse(numbers: List, delimiter: str) -> List:
    result = []
    for i in range(len(numbers) - 1):
        result.append(numbers[i])
        result.append(delimiter)
    result.append(numbers[-1])
    return result
'''
def check(candidate):
    assert candidate([], 7) == []
    assert candidate([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]
    assert candidate([2, 2, 2], 2) == [2, 2, 2, 2, 2]

#print(intersperse([], 7))  #这行测试line144 out of range报错，下面两行测试结果正确
print(intersperse([5, 6, 3, 2], 8))
print(intersperse([2, 2, 2], 2))
----------------------True?
'''
#7
def parse_nested_parens(paren_string):
  
    result = []
    
    # 将输入字符串拆分为组，并对每组进行遍历
    for group in paren_string.split():
        depth = 0
        max_depth = 0

        # 对每一组中的每个字符进行遍历
        for char in group:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        
        # 将最大深度添加到结果中
        result.append(max_depth)
    
    return result
'''
def check(candidate):
    # 检查第一个字符串
    assert candidate('(()()) ((())) () ((())()())') == [2, 3, 1, 3]

    # 检查第二个字符串
    assert candidate('() (()) ((())) (((())))') == [1, 2, 3, 4]

    # 检查第三个字符串
    assert candidate('(()(())((())))') == [4]

print(parse_nested_parens('(()()) ((())) () ((())()())'))
print(parse_nested_parens('() (()) ((())) (((())))'))
print(parse_nested_parens('(()(())((())))'))

[2, 3, 1, 3]
[1, 2, 3, 4]
[4]
-------------------------------True
'''
#8
def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    return [s for s in strings if substring in s]
'''
def check(candidate):

    # 检查第一个列表
    assert candidate([], 'john') == []

    # 检查第二个列表
    assert candidate(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx') == ['xxx', 'xxxAAA', 'xxx']

    # 检查第三个列表
    assert candidate(['xxx', 'asd', 'aaaxxy', 'john doe', 'xxxAAA', 'xxx'], 'xx') == ['xxx', 'aaaxxy', 'xxxAAA', 'xxx']

    # 检查第四个列表
    assert candidate(['grunt', 'trumpet', 'prune', 'gruesome'], 'run') == ['grunt', 'prune']

print(filter_by_substring([], 'john'))
print(filter_by_substring(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx'))
print(filter_by_substring(['xxx', 'asd', 'aaaxxy', 'john doe', 'xxxAAA', 'xxx'], 'xx'))
print(filter_by_substring(['grunt', 'trumpet', 'prune', 'gruesome'], 'run'))
[]
['xxx', 'xxxAAA', 'xxx']
['xxx', 'aaaxxy', 'xxxAAA', 'xxx']
['grunt', 'prune']
---------------------------------True
'''
#9
from typing import List, Tuple
import functools

def sum_product(numbers: List[int]) -> Tuple[int, int]:
    if not numbers:
        return 0, 1
    else:
        return (
            sum(numbers),
            functools.reduce(lambda x, y: x + y, numbers)*functools.reduce(lambda x, y: x * y, numbers)
        )
'''
def check(candidate):

    #检查第一个列表
    assert candidate([]) == (0, 1)

    #检查第二个列表
    assert candidate([1, 1, 1]) == (3, 1)

    #检查第三个列表
    assert candidate([100, 0]) == (100, 0)

    #检查第四个列表
    assert candidate([3, 5, 7]) == (3 + 5 + 7, 3 * 5 * 7)

    #检查第五个列表
    assert candidate([10]) == (10, 10)

print(sum_product([]))
print(sum_product([1, 1, 1]))
print(sum_product([100, 0]))
print(sum_product([3, 5, 7]))
print(sum_product([10]))
(0, 1)
(3, 3)
(100, 0)
(15, 1575)
(10, 100)
'''
#10
from typing import List, Tuple

def rolling_max(numbers):
    result = []
    current_max = float('-inf')
    for i in range(len(numbers)):
        if numbers[i] > current_max:
            current_max = numbers[i]
        result.append(current_max)
    return result
'''
def check(candidate): 
    assert candidate([]) == [] 
    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4] 
    assert candidate([4, 3, 2, 1]) == [4, 4, 4, 4] 
    assert candidate([3, 2, 3, 100, 3]) == [3, 3, 3, 100, 100]

print(rolling_max([]))
print(rolling_max([1, 2, 3, 4]))
print(rolling_max([4, 3, 2, 1]))
print(rolling_max([3, 2, 3, 100, 3]))

[]
[1, 2, 3, 4]
[4, 4, 4, 4]
[3, 3, 3, 100, 100]
#帮助删除了多余的引号和空格
---------------------------------True?
'''
#11
def is_palindrome(string: str) -> bool:
    """检查给定的字符串是否是回文。"""
    return string == string[::-1]

def make_palindrome(string: str) -> str:
    """找到以给定的字符串开头的最短回文。
    算法思想很简单：
    - 找到给定字符串的最长回文后缀。
    - 将回文后缀前的字符串反转，附加到给定字符串的末尾。
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    """
    palindrome = ''
    for i in range(len(string)):
        if is_palindrome(string[i:]):
            palindrome = string[i:] + palindrome
            break
    return palindrome
'''
def check(candidate):
    assert candidate('') == ''
    assert candidate('x') == 'x'
    assert candidate('xyz') == 'xyzyx'
    assert candidate('xyx') == 'xyx'
    assert candidate('jerry') == 'jerryrrej'

print(make_palindrome(''))
print(make_palindrome('x'))
print(make_palindrome('xyz'))
print(make_palindrome('xyx'))
print(make_palindrome('jerry'))

x
z
xyx
y
'''
#12 代码无意义
def string_xor(a: str, b: str) -> str:
    # 将字符串转换为整数列表
    int_a = int(a, 2)
    int_b = int(b, 2)
    # 执行异或操作
    result = []
    for i in range(len(a)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append(int_xor(a[i], b[i]))
    return ''.join(result)

def int_to_binary(num: int) -> str:
    return bin(num)[2:]
'''
def check(candidate):
    assert candidate('111000', '101010') == '010010'
    assert candidate('1', '1') == '0'
    assert candidate('0101', '0000') == '0101'

print(string_xor('111000', '101010'))
print(string_xor('1', '1'))
print(string_xor('0101', '0000'))
'''
#13
from typing import List, Optional
#多了[str]:
def longest(strings: List[str]) -> Optional[str]:
    if not strings:
        return None
    return max(strings, key=len)
'''
def check(candidate):
    assert candidate([]) == None
    assert candidate(['x', 'y', 'z']) == 'x'
    assert candidate(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'
print(longest([]))
print(longest(['x', 'y', 'z']))
print(longest(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']))
None
x
zzzz
-------------------------True
'''
#14
def greatest_common_divisor(a: int, b: int) -> int:
    def _gcd(x: int, y: int) -> int:
        while y != 0:
            x, y = y, x % y
        return x
'''
def check(candidate):
    assert candidate(3, 7) == 1
    assert candidate(10, 15) == 5
    assert candidate(49, 14) == 7
    assert candidate(144, 60) == 12
print(greatest_common_divisor(3, 7))
print(greatest_common_divisor(10, 15))
print(greatest_common_divisor(49, 14))
print(greatest_common_divisor(144, 60))
None
None
None
None
'''
#15
from typing import List

def all_prefixes(string: str) -> List[str]: #源代码这里多了List[str]
    prefixes = []
    for i in range(len(string)):
        prefix = string[:i + 1]
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes
'''
def check(candidate):
    assert candidate('') == []
    assert candidate('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']
    assert candidate('WWW') == ['W', 'WW', 'WWW']
print(all_prefixes(''))
print(all_prefixes('asdfgh'))
print(all_prefixes('WWW'))
[]
['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']
['W', 'WW', 'WWW']
-------------------------------True？
'''
#16
def string_sequence(n: int) -> str: #str:写了两次
    return ' '.join(str(i) for i in range(n + 1))
'''
def check(candidate):
    assert candidate(0) == '0'
    assert candidate(3) == '0 1 2 3'
    assert candidate(10) == '0 1 2 3 4 5 6 7 8 9 10'
print(string_sequence(0))
print(string_sequence(3))
print(string_sequence(10))
0
0 1 2 3
0 1 2 3 4 5 6 7 8 9 10
------------------------------True？
'''
#17
def count_distinct_characters(string: str) -> int:
    def count_chars(s: str) -> int:
        distinct_chars = 0
        for c in s:
            if c.isalpha():
                distinct_chars += 1
        return distinct_chars
'''
def check(candidate):
    assert candidate('') == 0
    assert candidate('abcde') == 5
    assert candidate('abcde' + 'cade' + 'CADE') == 5
    assert candidate('aaaaAAAAaaaa') == 1
    assert candidate('Jerry jERRY JeRRRY') == 5

#print(count_distinct_characters('abcde' + 'cade' + 'CADE'))
#print(count_distinct_characters('aaaaAAAAaaaa'))
#print(count_distinct_characters('Jerry jERRY JeRRRY'))
None
None
None
None
None
'''
#18
'''
def parse_music(music_string):
    #:type: str    #这里格式问题
    durations = {'o': 4, 'o|': 2, '.|': 1}
    result = []
    i = 0
    while i < len(music_string):
        note = music_strin    g[i]
        if note in durations:
            result.append(durations[note] = durations[note])
        i += 1
    return result
'''
#19
def how_many_times(string: str, substring: str) -> int:
    """
    :param string: 给定字符串
    :param substring: 子字符串
    :return: 子字符串在给定字符串中出现的次数
    """
    count = 0
    start = 0
    while True:
        index = string.find(substring, start)
        """
        :param start: 子字符串在给定字符串为空或找不到子字符串，则返回0。
        """
        if not string or not substring:
            return 0
        """
        :param string: 给定字符串
        """
        if not substring:
            """
            :param substring: 子字符串
            """
            return string.count(substring)
        count = 0
        while True:
            index = string.find(substring, 子字符串)
            """
            while循环
            """
            while index != -1:
                count += 1
                index = string.find(substring, index + 1)
        return count

def check(candidate):
    assert candidate('', 'x') == 0
    assert candidate('xyxyxyx', 'x') == 4
    assert candidate('cacacacac', 'cac') == 4
    assert candidate('john doe', 'john') == 1

print(how_many_times('', 'x'))
print(how_many_times('xyxyxyx', 'x'))
print(how_many_times('cacacacac', 'cac'))
print(how_many_times('john doe', 'john'))
