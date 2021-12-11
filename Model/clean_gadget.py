import re
import xlrd

keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL'})

main_set = frozenset(
    {'main', 'memcpy', 'wmemcpy', '_memccpy', 'memmove', 'wmemmove', 'memset', 'wmemset', 'memcmp', 'wmemcmp', 'memchr',
     'wmemchr', 'strncpy', 'lstrcpyn', 'wcsncpy', 'strncat', 'bcopy', 'cin', 'strcpy', 'lstrcpy', 'wcscpy', '_tcscpy',
     '_mbscpy', 'CopyMemory', 'strcat', 'lstrcat', 'fgets', 'main', '_main', '_tmain', 'Winmain', 'AfxWinMain',
     'getchar', 'getc', 'getch', 'getche', 'kbhit', 'stdin', 'm_lpCmdLine', 'getdlgtext', 'getpass', 'istream.get',
     'istream.getline', 'istream.peek', 'istream.putback', 'streambuf.sbumpc', 'streambuf.sgetc', 'streambuf.sgetn',
     'streambuf.snextc', 'streambuf.sputbackc', 'SendMessage', 'SendMessageCallback', 'SendNotifyMessage',
     'PostMessage', 'PostThreadMessage', 'recv', 'recvfrom', 'Receive', 'ReceiveFrom', 'ReceiveFromEx', 'CEdit.GetLine',
     'CHtmlEditCtrl.GetDHtmlDocument', 'CListBox.GetText', 'CListCtrl.GetItemText', 'CRichEditCtrl.GetLine',
     'GetDlgItemText', 'CCheckListBox.GetCheck', 'DISP_FUNCTION', 'DISP_PROPERTY_EX', 'getenv', 'getenv_s', '_wgetenv',
     '_wgetenv_s', 'snprintf', 'vsnprintf', 'scanf', 'sscanf', 'catgets', 'gets', 'fscanf', 'vscanf', 'vfscanf',
     'printf', 'vprintf', 'CString.Format', 'CString.FormatV', 'CString.FormatMessage', 'CStringT.Format',
     'CStringT.FormatV', 'CStringT.FormatMessage', 'CStringT.FormatMessageV', 'vsprintf', 'asprintf', 'vasprintf',
     'fprintf', 'sprintf', 'syslog', 'swscanf', 'sscanf_s', 'swscanf_s', 'swprintf', 'malloc', 'readlink', 'lstrlen',
     'strchr', 'strcmp', 'strcoll', 'strcspn', 'strerror', 'strlen', 'strpbrk', 'strrchr', 'strspn', 'strstr', 'strtok',
     'strxfrm', 'kfree', '_alloca'})

xread = xlrd.open_workbook('function.xls')
keywords_2 = []
for sheet in xread.sheets():
    col = sheet.col_values(0)[1:]
    keywords_2 += col

keywords_3 = ('_strncpy*', '_tcsncpy*', '_mbsnbcpy*', '_wcsncpy*', '_strncat*', '_mbsncat*', 'wcsncat*', 'CEdit.Get*',
              'CRichEditCtrl.Get*',
              'CComboBox.Get*', 'GetWindowText*', 'istream.read*', 'Socket.Receive*', 'DDX_*', '_snprintf*',
              '_snwprintf*')

keywords_4 = ('*malloc',)


def notinKeyword_3(token):
    for key in keywords_3:
        if len(token) < len(key) - 1:
            return True
        if key[:-1] == token[:len(key) - 1]:
            return False
        else:
            return True


def notinKeyword_4(token):
    for key in keywords_4:
        if len(token) < len(key) - 1:
            return True
        if token.find(key[1:]) != -1:
            return False
        else:
            return True


main_args = frozenset({'argc', 'argv'})


def clean_gadget(gadget):
    fun_symbols, var_symbols = {}, {}
    fun_count, var_count = 1, 1
    rx_comment = re.compile('\*/\s*$')
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')
    cleaned_gadget = []

    for line in gadget:
        if rx_comment.search(line) is None:
            nostrlit_line = re.sub(r'".*?"', '""', line)
            nocharlit_line = re.sub(r"'.*?'", "''", nostrlit_line)
            ascii_line = re.sub(r'[^\x00-\x7f]', r'', nocharlit_line)
            user_fun = rx_fun.findall(ascii_line)
            user_var = rx_var.findall(ascii_line)
            for fun_name in user_fun:
                if len({fun_name}.difference(main_set)) != 0 and len(
                        {fun_name}.difference(keywords)) != 0 and fun_name not in keywords_2 and notinKeyword_3(
                    fun_name) and notinKeyword_4(fun_name):
                    if fun_name not in fun_symbols.keys():
                        fun_symbols[fun_name] = 'FUN' + str(fun_count)
                        fun_count += 1
                    ascii_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], ascii_line)

            for var_name in user_var:
                if len({var_name}.difference(keywords)) != 0 and len({var_name}.difference(main_args)) != 0:
                    if var_name not in var_symbols.keys():
                        var_symbols[var_name] = 'VAR' + str(var_count)
                        var_count += 1
                    ascii_line = re.sub(r'\b(' + var_name + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', \
                                        var_symbols[var_name], ascii_line)
            cleaned_gadget.append(ascii_line)
    return cleaned_gadget


