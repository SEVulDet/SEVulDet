
def transform_type(c1,c2,c3,c4,c5,item):
    type_list = []
    detail_list = []
    for i in range(item):
        item_list=[]
        description_list = []
        if c1[i] == 0 and c2[i] == 0 and c3[i] == 0 and c4[i] == 0 and c5[i] == 0:
            item_list.append('None')
            description_list.append('No vulnerabilities in the current program')
            type_list.append(item_list)
            detail_list.append(description_list)
            # print('typelist:', type_list)
            # print('detaillist:', detail_list)
            continue
        if c1[i] == 1:
            item_list.append('CWE-078')
            description_list.append('CWE-078: Improper Neutralization of Special Elements used in an OS Command (\'OS Command Injection\')')
        if c2[i] == 1:
            item_list.append('CWE-122')
            description_list.append('CWE-122: Heap-based Buffer Overflow')
        if c3[i] == 1:
            item_list.append('CWE-121')
            description_list.append('CWE-121: Stack-based Buffer Overflow')
        if c4[i] == 1:
            item_list.append('CWE-762')
            description_list.append('CWE-762: Mismatched Memory Management Routines')
        if c5[i] == 1:
            item_list.append('CWE-others')
            description_list.append('This is a common vulnerability， please check the program carefully.')
        type_list.append(item_list)
        detail_list.append(description_list)
        # print('typelist:',type_list)
        # print('detaillist:',detail_list)
    return type_list,detail_list

