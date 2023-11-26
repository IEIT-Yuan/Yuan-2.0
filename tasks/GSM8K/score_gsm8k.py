import os
import re
import logging
from fractions import Fraction


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
    

def replace_transmean(text):
    if not text:
        return
    if '\\' in text:
        text = text.replace("\\", "\\\\")
    pattern = r'([\[\]\(\)\{\}\.\*\+\?\^\-])'
    replacement = r'\\\1'
    replaced_text = re.sub(pattern, replacement, text)
    
    return replaced_text


def replace_fraction(text):
    if 'frac' not in text:
        return text
    f_t = re.findall(r'(?:\s*\}){3,}',text)
    if f_t:
        return text
    pattern = r'\\[d]?frac\{([\d,.]+)\}\{([\d,.]+)\}'
    replacement = r'\1/\2'
    replaced_text = re.sub(pattern, replacement, text)
    
    return replaced_text


def has_numbers(input_string):
    pattern = r'\d+'
    match = re.search(pattern, input_string)
    return bool(match)


def eval_gen_res(txt_files_path,content_yuan_all_dict,len_ori):
    if not txt_files_path:
        return

    qa_content = process_gen_files(txt_files_path,len_ori)
    text_answer_true, text_answer_false = [], []

    for i in range(len(qa_content)):
        qa_content[i] = qa_content[i].strip()
        if not qa_content[i]:
            continue
        qa_content[i] = re.sub('(\s*<n>\s*)+?$','',qa_content[i])
        qa_content[i] = re.sub(r'\\boxed{\s*([\d.,/]+)\s*}[.]?','\\1',qa_content[i])
        q_text = qa_content[i].split('[SEP]')[0].strip()
        ans_all = qa_content[i].split('[SEP]')[1].replace('<br>','<n>').replace('\!','').replace('\;','').replace('\,','')
        ans_all = ans_all.replace('$$',' ').replace('$','').replace(' ','').replace('≈','=')

        ans_true = content_yuan_all_dict[q_text]
        ans_true = replace_transmean(ans_true)
        match_regex_ans = '(?<=[^0-9+\-*/])'+ans_true+'(?![xyzXYZ\d.+\-*/=()√₀\²³‰¼½¾_×¬^,!:±×÷∶∧∨∑∏∪∷√≡≠><≤≥])'
        split_n = ans_all.split('<n>')
        end_2,ans_end_2 = '',[]
        if '<n>' in ans_all:
            end = split_n[-1]
            if '=' not in end:
                end_2 = '<n>'+split_n[-2].split('=')[-1]
            if end and not has_numbers(end):
                end = split_n[-2]
            if end and not has_numbers(end) and len(split_n)>2:
                end = split_n[-3]
        else:
            end = ans_all
        
        end = '<n>'+end.split('=')[-1]
        end = end.replace('\\%','%').replace('\%','%')
        end = replace_fraction(end)
        ans_gen = re.findall(r'(?:(?:[-+]?\d+\.?\d+/\d+\.?\d+)|(?:[-+]?\d+/\d+)|(?:[-+]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-+]?\d+))(?![xyzXYZ\d+\-*/=()√₀\²³‰¼½¾_×¬^,!:±÷∶∧∨∑∏∪∷√≡≠><≤≥])', end)
        if end_2:
            end_2 = replace_fraction(end_2)
            ans_end_2 = re.findall(match_regex_ans, end_2)
        
        ans_end = re.findall(match_regex_ans, end)

        qa_content[i] = qa_content[i]+'[EOD]'+ans_true
        if ans_end:
            text_answer_true.append(qa_content[i]+'\n')
            continue
        else:
            if ans_end_2 and not ans_gen:
                text_answer_true.append(qa_content[i]+'\n')
                continue
            flag = False
            if ans_gen:
                ans_true = content_yuan_all_dict[q_text]
                ans_true = ans_true.replace(',','').replace(' ','')
                num_v = max(-1,len(ans_gen)-4)
                for vi in range(len(ans_gen)-1,num_v,-1):
                    value = ans_gen[vi]
                    value = value.replace(',','').replace(' ','')
                    if '/' in value:
                        frac_gen = Fraction(value)
                        value = frac_gen.numerator / frac_gen.denominator
                    if float(value)==float(ans_true):
                        flag = float(value)==float(ans_true)
                        text_answer_true.append(qa_content[i]+'\n')
                        break
            if flag:
                continue
        text_answer_false.append(qa_content[i]+'\n')
    
    accuracy = len(text_answer_true)/(len(text_answer_true)+len(text_answer_false))
    print(f'Number of correct answers:{len(text_answer_true)}')
    print(f'Number of incorrect answers:{len(text_answer_false)}')
    print('accuracy:', accuracy)

    if not os.path.exists(txt_eval_res_dir):
        os.makedirs(txt_eval_res_dir)
    save_a_file(text_answer_true, txt_eval_res_dir + "/gsm_res_true.txt")
    save_a_file(text_answer_false, txt_eval_res_dir + "/gsm_res_false.txt") 


def main():
    global file_type_list, txt_eval_res_dir
    origin_file_path = "<Specify path>"
    eval_file_path = "<Specify path>"
    txt_eval_res_dir = "<Specify path>"

    file_type_list = ['jsonl']
    content_qa = load_a_file(origin_file_path)
    content_yuan_all_dict = {}
    for i in range(len(content_qa)):
        g = content_qa[i]
        if g == "":
            continue
        g = g.strip('\n')
        g = g.split('[SEP]')
        content_yuan_all_dict[g[0].strip()] = g[1].replace(' ','')
    
    eval_gen_res(eval_file_path, content_yuan_all_dict, len(content_qa))


if __name__ == "__main__":
    main()
    