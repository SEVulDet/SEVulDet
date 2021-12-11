import re


def remove_comment(lines: str) -> str:
    """remove /*...*/ /.../"""
    state = 0
    ret = []
    r"""
    state:
    0: normal text [5/3] [secure/_stdio.h]
    1: [/]
    2: [/*] [/*sth] [/*sth*sth] block comments
    3: [/*sth*]
    4: [//] [//sth\] inline comments
    5: ['] ['\n] ['\t] etc. char
    6: ['\]
    7: ["] ["\n] ["\"] ["\t] etc. string
    8: ["\]
    9: [//sth\] [//sth\\\]
    """
    for c in lines:
        if state == 0 and c == '/':  # ex. [/]
            state = 1
        elif state == 1 and c == '*':  # ex. [/*]
            state = 2
        elif state == 1 and c == '/':  # ex. [//]
            state = 4
        elif state == 1:  # ex. [<secure/_stdio.h> or 5/3]
            ret.append('/')
            state = 0

        elif state == 2 and c == '*':  # ex. [/*he*]
            state = 3
        elif state == 2:  # ex. [/*heh]
            state = 2
            if c == '\n':
                ret.append('\n')

        elif state == 3 and c == '/':  # ex. [/*heh*/]
            state = 0
            ret.append(' ')  # prevent 'int/**/a' -> 'inta'
        elif state == 3:  # ex. [/*heh*e]
            state = 2

        elif state == 4 and c == '\\':  # ex. [//hehe\]
            state = 9
        elif state == 9 and c == '\\':  # ex. [//hehe\\\\\]
            state = 9
        elif state == 9:  # ex. [//hehe\<enter> or //hehe\a]
            state = 4
        elif state == 4 and c == '\n':  # ex. [//hehe<enter>]
            state = 0

        elif state == 0 and c == '\'':  # ex. [']
            state = 5
        elif state == 5 and c == '\\':  # ex. ['\]
            state = 6
        elif state == 6:  # ex. ['\n or '\' or '\t etc.]
            state = 5
        elif state == 5 and c == '\'':  # ex. ['\n' or '\'' or '\t' ect.]
            state = 0

        elif state == 0 and c == '\"':  # ex. ["]
            state = 7
        elif state == 7 and c == '\\':  # ex. ["\]
            state = 8
        elif state == 8:  # ex. ["\n or "\" or "\t ect.]
            state = 7
        elif state == 7 and c == '\"':  # ex. ["\n" or "\"" or "\t" ect.]
            state = 0

        if (state == 0 and c != '/') or state == 5 or state == 6 or \
                state == 7 or state == 8:
            ret.append(c)
    return ''.join(ret)


def remove(lines: str) -> str:
    """remove non-ASCII char, comment, and #include"""
    # remove non-ASCII
    lines = re.sub(r'[^(\x00-\x7f)]*', '', lines)
    lines = remove_comment(lines)
    # remove #include
    lines = re.sub(r'^\s*#include\s*["<][\s\w./]+[">]', '', lines, flags=re.M)
    # remove blank lines
    # lines = '\n'.join(line for line in lines.split('\n')
    #                   if line and not line.isspace())
    return lines


if __name__ == '__main__':
    code = open('data/test.cpp', 'r', encoding='utf-8').read()
    open('data/result.cpp', 'w', encoding='utf-8').write(remove(code))
