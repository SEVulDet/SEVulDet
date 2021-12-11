import sys
import re
import os
import xlrd
import csv
from application.split.preprocess import *

filePath = ''
insteadStr = '3r45fkkifdxjh'  # 随便定义的字符串，用于替换双引号及里面的内容
contentList = []
sourceCode = {}
removeWords = ['if', 'for', 'switch', 'while', 'sizeof']  # 移除函数定义混淆
removeVarWords = ['namespace', 'return']  # 移除变量定义混淆
funcSplitChar = '\n********************************************************\n'


# def openSourceCode():
#     # print(sys.argv)
#     fileNum = len(sys.argv) - 1
#     sourceDict = {}
#     try:
#         for i in range(fileNum):
#             sourceName = sys.argv[i + 1]
#             url = sys.argv[0]
#             try:
#                 r = re.findall(r'[\\][^\\]+\.py$', url)
#                 if not r:
#                     raise Exception('not found: regular exp1')
#                 url = url.replace(r[0], '') + '\\sourceCode\\' + sourceName
#             except Exception as e:
#                 try:
#                     r = re.findall(r'[/][^/]+\.py$', url)
#                     if not r:
#                         raise Exception('not found: regular exp2')
#                     url = url.replace(r[0], '') + '/sourceCode/' + sourceName
#                 except Exception as e:
#                     print(e)
#             print('读取文件：' + url)
#             with open(url, 'r') as file:
#                 sourceDict[sourceName] = replaceQuote(replaceMacro(remove(file.read())))  # 去除注释头文件非ASCII并加入字典
#     except Exception as e:
#         print(e)
#         print('参数错误')
#     return sourceDict

def loadLibFunc(filename) -> list:
    try:
        url = sys.argv[0]
        url = re.sub(r'[/\\]\w+[.]py', '/' + filename, url)
        wb = xlrd.open_workbook(url)  # 打开Excel文件
        if wb is not None:
            sheet = wb.sheet_by_name('Sheet1')  # 通过excel表格名称(rank)获取工作表
            funcName = []  # 创建空list，用来保存人物名称一列
            for a in range(1, sheet.nrows):  # 循环读取表格内容（每次读取一行数据）
                cells = sheet.row_values(a)  # 每行数据赋值给cells
                funcName.append(cells[0])
            sheet = wb.sheet_by_name('Sheet2')
            for a in range(1, sheet.nrows):  # 循环读取表格内容（每次读取一行数据）
                cells = sheet.row_values(a)  # 每行数据赋值给cells
                funcName.append(cells[0])
            sheet = wb.sheet_by_name('Sheet3')
            for a in range(1, sheet.nrows):  # 循环读取表格内容（每次读取一行数据）
                cells = sheet.row_values(a)  # 每行数据赋值给cells
                funcName.append(cells[0])
        else:
            print('open a null xls')
        return funcName
    except Exception as e:
        print('loadLibFunc fail')
        return []


def openSourceCode():
    global filePath
    sourceDict = {}
    filename = re.findall(r'\b\w+\b[.]\b\w+\b', filePath)[-1]
    try:
        with open(filePath, 'r') as file:
            content = file.read()
            print('inputting file lines:', printLines(content))
            sourceDict[filename] = replaceQuote(replaceMacro(remove(content)))  # 去除注释头文件非ASCII并加入字典
            print('open file: ', filename)
    except Exception as e:
        print('filePath error: ', e)
    return sourceDict


def printLines(strings: str):
    ind = 0
    i = 0
    while ind < len(strings):
        if strings[ind] == '\n':
            i += 1
        ind += 1
    return i


def findsub(pattern, content, funNameContainer):  # 查找被调用函数里的函数，即嵌套调用
    sublist = []
    if content:
        for i in content:
            funNameContainer.append(i[0])  # 添加函数名到列表
            var = re.findall(pattern, i[1])  # 搜索函数括号中的内容
            for s in var:
                sublist.append(s)  # 添加元组到list
        # print(sublist)
        findsub(pattern, sublist, funNameContainer)


def findFunc(str: str) -> tuple:  # 查找被调用的函数
    global removeWords
    mylist = []
    pattern = r'\b([\w]+)\b *[(](.*)[)] *'
    findsub(r'\b([\w]+)\b *[(](.*)[)] *', re.findall(pattern, str), mylist)  # 包含函数名个元组，每个元组有两个值，整个函数名和函数括号里的内容值
    temp = set(mylist)
    mylist = list(temp)
    # 最初只匹配带分号的函数，也就是调用函数而不包括定义函数
    # 该表达式存在问题：('sizeof', 'wchar_t) * (100 - dataLen - 1')，但不影响结果
    for i in removeWords:
        if i in mylist:
            mylist.remove(i)
    return set(mylist)  # 去除重复


def isFunc(str: str):  # 防止回溯过多的字符导致函数交叉
    mystr = str
    funclist = re.findall(r'(\b\w+\b *[(].*[)])', mystr, flags=re.M)  # 应该从后开始搜索，找距离左花括号最近的
    # print(funclist)
    if funclist.__len__() > 0:
        prefix = re.findall(r'[^(] *\b(\w+)\b *[*]? *\w+', str, flags=re.MULTILINE)
        if prefix.__len__() != 0:
            return prefix[-1] + ' ' + funclist[-1]  # 返回距离花括号最近的函数
        else:
            return funclist[-1]
    else:
        return ''


def findFuncDefine(str: str):
    state = 0
    flag = False
    slideWindow = ''  # 用于回溯函数名
    index = 0
    start = 0
    funcList = []
    lineStart = 0
    lineCount = 0
    # print('str lines: ', printLines(str))
    # numberCount = 0
    while index < len(str):
        s = str[index]
        if s == '\n':
            lineCount += 1
        if not flag:
            slideWindow += s
            if slideWindow.__len__() > 100:  # 只保留左花括号前100字符，防止字符中有过多的上一函数的字符
                slideWindow = slideWindow[1:]
        if s == '{' and state == 0:
            lineStart = lineCount
            start = index  # 记录左花括号开始坐标
            flag = True
            state += 1
        elif s == '{':  # state > 0:
            state += 1
        elif s == '}':
            if state > 0:
                state -= 1
        if state == 0 and flag:
            flag = False
            funcHead = isFunc(slideWindow[:-1])  # 查找slidewindow中的字符是否符合函数定于规则
            if funcHead is not '':
                funcList.append([funcHead + str[start: index + 1], lineStart, lineCount])

                # (funcSplitChar + 'Number: ' + numberCount.__str__() + '     '
                #              + 'Line: ' + lineStart.__str__() + '~' +
                #             lineCount.__str__() + funcSplitChar + funcHead + str[start: index + 1])
                # numberCount += 1
            else:  # 解决嵌套问题:类嵌套函数，命名空间内的函数
                index = start
                lineCount = lineStart
                slideWindow = ''
                flag = False
                # state += 1
        index += 1
    # print(len(funcList))
    return funcList  # list里保存了代码的每个函数，行号起止


def findFunDefine(str) -> tuple:  # 返回自定义函数的函数名列表，不包含花括号的内容
    global removeWords
    mylist = re.findall(r'(\w+) *[(].*[)]\s*[{]', str, flags=re.M)
    temp = set(mylist)
    mylist = list(temp)
    for i in removeWords:
        if i in mylist:
            mylist.remove(i)
    return set(mylist)


def replaceQuote(str) -> str:  # 使用有编号的混乱字符串暂时代替引号和里面的内容，防止给后面的处理函数造成干扰
    # 记得最后调用restoreQuote()
    global insteadStr
    global contentList
    number = 0
    mystr = str
    contentList = re.findall(r'["][^"]*["]', str)  # 注意\"
    # print(contentList)
    for i in contentList:
        mystr = mystr.replace(i, insteadStr + number.__str__(), 1)
        number += 1
    return mystr


def restoreQuote(funclist) -> str:
    global insteadStr
    global contentList
    number = 0
    for content in contentList:
        for i in range(funclist.__len__()):
            if re.search(insteadStr + number.__str__(), funclist[i][0]):
                funclist[i][0] = funclist[i][0].replace(insteadStr + number.__str__(), content)
                break
        number += 1
    contentList = []
    return funclist

def findsubVar(varlist: list):
    subvar = []
    for i in varlist:
        subvar.append(i[0])
        if isinstance(i,tuple):
            if i[1] != '':
                temp = re.findall(r'[^=] *\b([a-zA-Z_]\w+)\b', i[1])
                for t in temp:
                    subvar.append(t)
        elif isinstance(i,str):
            subvar.append(i)
    # print(subvar)
    return subvar

def findVar(str: str) -> tuple:
    regularexp =[
        # r'(struc)t +\b(\w+)\b +{'
        # r'^[;\s( ]*(\w+)\b *(\w+)'
        # r'^[;\s( ]*\bstruct\b +([\w]+)\b +([\w]+)'
        r'struct *[{][^{}]*[}] *\b(\w+)\b *;',  # 匹配
        r'\b(?:(?!struct)(?!return)(?!namespace)\w+)\b[*]? *[*]* *\b(\w+)\b((?: *, *[*]? *\b\w+\b)*) *(?! )(?!\w)(?!\()'  # 匹配int var; int var1,var2; int* var...
                                                                                                # 只匹配struct struct_define struct_var; 后面两个单词 by 正向否定零宽断言
                                                                                                # 不匹配函数定义如 int func(...)...
    ]
    varList = []
    for regr in regularexp:
        temp = re.findall(regr, str, flags=re.MULTILINE)
        # print(temp)
        varList += findsubVar(temp)



    for word in removeVarWords:
        for lst in varList:
            if lst == word:
                varList.remove(lst)

    # list_remove1 = re.findall(r'(return) +\b([0-9]+)\b', str, flags=re.MULTILINE)
    # for rm in list_remove1:  # 移除return *
    #     if rm in list2:
    #         list2.remove(rm)
    # print(list)
    # 暂时没有考虑其他情况

    # newlist = []
    # for i in list:
    #     if i[0] != "struct":  # 去除结构体判断错误
    #         newlist.append(i[1])

    return set(varList)


def replaceMacro(str: str) -> str:
    macroList = re.findall(r'#define +\b(\w+)\b +(.*)', str)  # 暂时没有考虑#ifdef和定义一个多行的#define
    newStr = re.sub(r'#define +\b\w+\b +.*', ' ', str)  # 没有删除多余空行
    for i in macroList:
        newStr = newStr.replace(i[0], i[1])
    return newStr


def tokenNormalized():
    libfunc = loadLibFunc('库函数.xls')
    for i, j in sourceCode.items():  # i，j 文件名，内容
        var = findVar(j)
        func = []
        func1 = findFunDefine(j)  # 自定义的函数直接加入列表
        func2 = findFunc(j)
        # print(func1)
        # print(func2)
        for ii in func2:
            if ii not in libfunc:  # 库函数则不替换
                func.append(ii)
        func += list(func1)
        # print(var)
        # print(func)

        code = j
        number = 0

        for v in var:  #
            code = re.sub(r'\b' + v + r'\b', 'var' + number.__str__(), code)
            # code = code.replace(v, 'var' + number.__str__())  # 由于丢失了位置信息，可能会出问题
            number += 1
        number = 0
        for f in func:
            # code = code.replace(f, 'func' + number.__str__())
            code = re.sub(r'(' + r'\b' + f + r'\b' + r')', 'func' + number.__str__(), code)  # 命名空间也被替换了
            number += 1
        funclist = restoreQuote(findFuncDefine(code))
        # funclist = findFuncDefine(code)
        retPath = writeFile(i[:i.rfind('.')], funclist)
        return retPath


def writeFile(filename: str, code: list):
    global filePath
    path = re.findall(r'(.+[/\\])\w+[/\\]\b\w+\b[.]\b\w+\b', filePath)[-1] + r'output/'
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
    head = ['FileContent', 'CWE-078', 'CWE-122', 'CWE-121', 'CWE-762', 'CWE-others', 'Line_start', 'Line_end']
    WriteList = []
    content = open(path + '/code.cpp', 'w')  # 测试
    for i in code:
        content.write(i[0] + '\n\n' + funcSplitChar + '\n\n\n')
        WriteList.append({'FileContent': i[0], 'Line_start': i[1], 'Line_end': i[2]})
    content.close()
    retPath = path + filename + '.csv'
    with open(retPath, 'w', newline='') as f:
        write = csv.DictWriter(f, head)
        write.writeheader()
        write.writerows(WriteList)
        f.close()
    return retPath
    # with open(path + filename + '.csv') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         print(row)


# def writeFile(filename: str, code: list):
#     url = sys.argv[0]
#     url = re.sub(r'[/][^/]+\.py', '/output/', url)
#     if not os.path.exists(url):
#         os.mkdir(url)
#     if len(code) > 0:
#         try:
#             open(url + filename + '.txt', 'w', encoding='utf-8').write('')
#         except IOError as e:
#             print(e)
#         for c in code:
#             number = 0
#             for contest in contentList:
#                 c = c.replace(insteadStr + number.__str__(), contest)
#                 number += 1
#             try:
#                 open(url + filename + '.txt', 'a', encoding='utf-8').write(c)
#             except IOError as e:
#                 print(e)


def init(path: str):
    global sourceCode
    global filePath
    filePath = path
    sourceCode = openSourceCode()


def codeSplit(path: str):  # 给定一个源码文件的绝对路径，程序会在源码所在文件夹的同级目录创建拥有一个output文件夹，并输出xxx.csv
    init(path)
    retPath = tokenNormalized()
    # print(retPath)
    return retPath


if __name__ == '__main__':
    codeSplit(r'D:\myfiles\2020\pythonprojects\NSC\NSC\application\split\sourceCode\CWE78_OS_Command_Injection__wchar_t_listen_socket_w32_spawnvp_84_bad.cpp')
    # linux 和 windows 的路径分割符可以统一用'/'或者变量os.sep
