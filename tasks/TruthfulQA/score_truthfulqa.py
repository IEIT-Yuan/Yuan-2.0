import sys
input_path = sys.argv[1]
with open(input_path, 'r', encoding='utf-8') as fin:
    lines = fin.readlines()
    count = 0
    total = len(lines)
    for line in lines:
        line = eval(line)
        ans_gen = line['gen_ans']
        index = ans_gen.find('<sep>')
        ques = ans_gen[:index]
        ans_gen = ans_gen[index+len('<sep>'):]
        ans_gen = ans_gen.replace(' ', '').replace('\n', '<n>')
        index0 = ans_gen.find('。')
        ans_gen = ans_gen[:index0+1]
        if '正确答案是A选项。' == ans_gen:
            count += 1
    print('MC1 acc: {}'.format(format(count/total, '.2f')))
