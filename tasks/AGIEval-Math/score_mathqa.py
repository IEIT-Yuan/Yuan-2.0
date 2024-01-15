import os
import re
import logging


def load_a_file(a_file, content=None):
    if not content:
        content = []
    with open(a_file, encoding="utf-8", errors="ignore") as f:
        g = f.readline()
        content.append(g)
        while g:
            try:
                g = f.readline()
                if g == '':
                    continue
                content.append(g)
            except Exception as e:
                logging.error(e)
                logging.error("A line coding error! : " + a_file)
    return content


def save_a_file(a_file_content,file_name):
    with open(file_name, "w", encoding='utf-8') as f:
        f.writelines(a_file_content)


def get_file_list(folder):
    filelist = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            file_type = file.split('.')[-1]
            if file_type in file_type_list:
                file_fullname = os.path.join(dirpath, file)
                filelist.append(file_fullname)
    return filelist


def process_gen_files(gen_path_dir,len_ori):
    txt_files_lst = get_file_list(gen_path_dir)
    txt_files_lst = sorted(txt_files_lst, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    
    content = []
    for i in range(len(txt_files_lst)):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file,content)

    diff_len = len(content) - len_ori
    content_all = []
    for i in range(diff_len):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file)
        content_all.extend(content[:-1])
    for i in range(diff_len,len(txt_files_lst)):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file)
        content_all.extend(content)
    
    return content_all


def get_ans_gen_letter(qa_text):
    if not qa_text:
        return qa_text
    a_text = qa_text.strip().split('[SEP]')[1]
    q_text = qa_text.split('[SEP]')[0].replace(' ','').replace('．','.')

    flag_q = re.findall(r'(?:错误)|(?:不合理)|(?:不正确)|(?:不[是会])|(?:不发生)|(?:False)|(?:无法推断)',q_text)
    if flag_q:
        flag_a = False
    else:
        flag_a = True
    a_text = a_text.replace(' ','').replace('$','').replace('Ｃ','C').replace('Ａ','A').replace('Ｂ','B').replace('Ｄ','D')
    a_text_n = re.findall(r'[ABCD]+',a_text)

    if len(a_text_n)==1 and a_text_n[0]==a_text:
        ans = [a_text]
    else:
        ans = []
    a_text = a_text.replace('<n>股癣','<n>故选').replace('弧选','故选').replace('<n>个选','故选')
    a_text = re.sub(r'\\rm\s*\{([A-Z]+)\}','\\1',a_text)
    a_text = re.sub(r'(?<=[ABCD])[和、](?=[ABCD])','',a_text)
    a_text = re.sub(r'\\mathrm\s*\{[\(]?([A-Z]+)[\)]?\}','\\1.',a_text)
    a_text = re.sub(r'\\textbf\s*\{\(([A-Z]+)\)\}','\\1.',a_text)
    a_text = re.sub(r'\\mathbf\s*\{([^{]*?)\}','\\1',a_text)
    a_text = re.sub(r'(?<=[：:])<n>(?=[ABCD]\.)','',a_text)
    a_text = re.sub(r'(?<=[：:])<n>(?=\([ABCD]\))','',a_text)
    a_text = re.sub(r'([故应本题答案为是正确的选项择]{3,})<n>(?=[ABCD])','\\1',a_text)
    a_text = re.sub(r'\\text\s*\{[\(]?([A-Z]+)[\)]?\}','\\1',a_text)
    a_text = re.sub(r'<font[^>]*?>([ABCD]+)</font>','\\1',a_text)
    if not ans:
        ans = re.findall(r'故选[:：\(（]*?[ABCDEF]+',a_text)
    if not ans:
        ans = re.findall(r'(?<=(?:故答案[为是][：:（\(])|(?:故本题选择)|(?:故答案应选)|(?:故答案[为是][：:（\(])|(?:答案选项[为是])|(?:正确选项[为是])|(?:正确答案[为是])|(?:正确说法[为是])|(?:答案[是为]选项))[:：\(（]*?[ABCDEF]+',a_text)
    if not ans:
        ans = re.findall(r'(?<=(?:故填[：:（\(])|(?:答案[为是])|(?:故选择)|(?:答案[：:（\(])|(?:故应选)|(?:所以选)|(?:本题选)|(?:因此选)|(?:答案选))[:：\(（]*?[ABCDEF]+',a_text)
    if not ans:
        ans = re.findall(r'(?<=(?:故本题选)|(?:故选项[为是])|(?:答案选[：:（\(])|(?:故应选[：:（\(])|(?:故答案选)|(?:故此题选)|(?:故选答案)|(?:[此本]题选择)|(?:答案选择)|(?:答案应为))[:：\(（]*?[ABCDEF]+',a_text)
    if not ans:
        ans = re.findall(r'(?<=(?:本题应选择：)|(?:本题的答案选)|(?:正确的选项[为是])|(?:正确的说法[为是])|(?:正确的叙述[为是])|(?:正确的表述[是为])|(?:正确的做法[是为])|(?:正确的陈述[是为]))[:：\(（]*?[ABCDEF]+',a_text)
    if not ans:
        ans = re.findall(r'故[ABCDEF]+[选项]?符合题意',a_text)
    if not ans:
        ans = re.findall(r'故[ABCDEF]+入选',a_text)
    if not ans:
        ans = re.findall(r'选项[ABCDEF]+符合',a_text)
    if not ans:
        ans = re.findall(r'(?<=<n>[故选])[ABCDEF]+',a_text)
    if not ans:
        ans = re.findall(r'[ABCDEF]+选项.{0,6}，故入选',a_text)
    if not ans:
        ans = re.findall(r'选[ABCDEF]+[\.。]',a_text)
    if not ans:
        ans = re.findall(r'(?<=(?:故为)|(?:应选)|(?:选择))[:：\(（]*?[ABCDEF]+',a_text)
    if not ans:
        ans = re.findall(r'answeris\([ABCD]\)',a_text)
    text_t = re.findall(r'选项([ABCDEF]\.[^。？?!！：]*?)+?[都均]正确',a_text)
    text_f = re.findall(r'选项([ABCDEF]\.[^。？?!！：]*?)+?(?:(?:[都均]不正确)|(?:没有正确)|(?:[都均]错误))',a_text)
    text_no = re.findall(r'选项A[中.][^B。？?！!]*?B[中.][^C。？?！!]*?C[中.][^D。？?！!]*?D[中.][^。？?！!]*?(?:都不符合条件)',a_text)
    if text_no:
        return None
    if not ans:
        if flag_a:
            ans = re.findall(r'正确的[\u4e00-\u9fa5]*?[是为][:：\(（]*?[ABCDEF]+',a_text)
            ans_lst = re.findall(r'(?:(?:故)|(?:选项)|)[:：\(（]*?[ABCDEF]+[\)）]?[是为选项]?正确',a_text)
            if not ans_lst:
                if text_t:
                    ans_lst = re.findall(r'[ABCDEF]\.',text_t[0])

            if not ans and ans_lst:
                ans_true = []
                for al in range(len(ans_lst)):
                    ans_true.append(re.findall(r'[ABCDEF]+',ans_lst[al])[0])
                if ans_true:
                    ans_true = ''.join(sorted(list(set(ans_true))))
                    ans = [ans_true]
        else:
            ans_lst = re.findall(r'(?:(?:故)|(?:选项)|)[:：\(（]*?[ABCDEF]+[\)）]?[是为选项]?错误',a_text)
            if not ans:
                ans = re.findall(r'(?:(?:错误的叙述[为是])|(?:说法错误的[为是])|(?:错误的说法[是为])|(?:叙述错误的[为是])|(?:错误的表述[是为])|(?:错误的做法[是为]))[:：\(（]*?[ABCDEF]+',a_text)
            if not ans:
                ans = re.findall(r'(?:不正确的[是为])[:：\(（]*?[ABCDEF]+',a_text)
            if not ans:
                ans = re.findall(r'(?:错误的[是为])[:：\(（]*?[ABCDEF]+',a_text)
            if not ans:
                ans = re.findall(r'[ABCDEF]选项理解有误',a_text)
            if not ans_lst:
                if text_f:
                    ans_lst = re.findall(r'[ABCDEF]\.',text_f[0])
            if not ans and ans_lst:
                ans_true = []
                for al in range(len(ans_lst)):
                    ans_true.append(re.findall(r'[ABCDEF]+',ans_lst[al])[0])
                if ans_true:
                    ans_true = ''.join(sorted(list(set(ans_true))))
                    ans = [ans_true]
    if not text_t and not text_f:
        if not ans:
            split_n= a_text.split('<n>')
            ans = re.findall(r'(?:[本此题的答案选项故是为择符合题目的入选符合题意正确错误应综上叙述说法]+[\u4e00-\u9fa5]*?[:：（\(]?[ABCDE]+[^F-Za-z\d]+)|(?:[ABCDE]+[\.)][^F-Za-z\d]*?[^。？！]*?[正确错误符合选项得体]{2,}).*?$',split_n[-1])
            if not ans and len(split_n)>=2:
                ans = re.findall(r'(?:[本此题的答案选项故是为择符合题目的入选符合题意正确错误应综上叙述说法]+[\u4e00-\u9fa5]*?[:：（\(]?[ABCDE]+[^F-Za-z\d]+)|(?:[ABCDE]+[\.)][^F-Za-z\d]*?[^。？！]*?[正确错误符合选项得体]{2,}).*?$',split_n[-2])
            reversed_list = ans[::-1]
            for value in reversed_list:
                if '答案' in value:
                    ans = [value]
                    break
                if '选项' in value:
                    ans = [value]
                    break
        if not ans:
            split_n = a_text.split('<n>')
            flag_n = re.findall(r'选项[ABCD]','<n>'+a_text)
            if not flag_n:
                flag_n = re.findall(r'（[ABCD]）',a_text)
            if not flag_n:
                flag_n = re.findall(r'[^ABCD][ABCD]\.','<n>'+a_text)
            
            if flag_n and '多选题' not in q_text and len(flag_n)>1:
                ans = []
            else:
                if '多选题' in q_text and flag_n:
                    for k in range(len(flag_n)):
                        flag_n[k] = re.findall(r'[ABCDEF]+',flag_n[k])[-1]
                    ans = [''.join(flag_n)]
                else:
                    ans = flag_n
    
    if ans:
        ans = re.findall(r'[ABCDEF]+',ans[-1])[-1]
        ans = ''.join(sorted(list(set(ans))))
        return ans
    else:
        return None


def process_txt_files(txt_files_path, content_yuan_all_dict, len_ori):

    if not txt_files_path:
        return
    qa_content = process_gen_files(txt_files_path,len_ori)
    text_answer_true, text_answer_false = [], []

    for i in range(len(qa_content)):
        qa_content[i] = qa_content[i].strip()
        if not qa_content[i]:
            continue
        qa_content[i] = re.sub('(\s*<n>\s*)+?$','',qa_content[i])
        qa_content[i] = re.sub(r'(?<=\[SEP\])\s+','',qa_content[i])
        q_text = qa_content[i].split('[SEP]')[0].strip()
        ans_gen = get_ans_gen_letter(qa_content[i])
        ans_true = ''
        for key in list(content_yuan_all_dict.keys()):
            if q_text in key or key in q_text:
                ans_true = content_yuan_all_dict[key]
                break
        if not ans_true:
            print(f'There is no original correct answer found for the question: {q_text}')
            continue
        if not ans_gen:
            ans_gen = 'None'
        else:
            if ans_true == ans_gen:
                text_answer_true.append(qa_content[i]+' [gen_ans] '+ ans_gen+' [true_ans] '+ ans_true+'\n')
                continue
        
        text_answer_false.append(qa_content[i]+' [gen_ans] '+ ans_gen+' [true_ans] '+ ans_true+'\n')
    
    accuracy = len(text_answer_true)/len(qa_content)
    print(f'Number of correct answers:{len(text_answer_true)}')
    print(f'Number of incorrect answers:{len(text_answer_false)}')
    print(f'accuracy: {accuracy}')
    if not os.path.exists(txt_eval_res_dir):
        os.makedirs(txt_eval_res_dir)
    save_a_file(text_answer_true, txt_eval_res_dir + "/math_res_true.txt")
    save_a_file(text_answer_false, txt_eval_res_dir + "/math_res_false.txt")


def main():
    global file_type_list, txt_eval_res_dir
    origin_file_path = "<Specify path>"
    eval_file_path = "<Specify path>"
    txt_eval_res_dir = "<Specify path>"

    file_type_list = ['jsonl']
    content_qa = load_a_file(origin_file_path)
    content_yuan_all_dict = {}
    for i in range(len(content_qa)):
        g = content_qa[i].strip()
        if not g:
            continue
        qa_lst = g.split('[SEP]')
        ans = re.findall('(?<=答案：)[ABCDEF]+',qa_lst[1].replace(' ',''))[0]
        content_yuan_all_dict[qa_lst[0]] = ans

    process_txt_files(eval_file_path, content_yuan_all_dict, len(content_qa))
    
if __name__ == "__main__":
    main()
