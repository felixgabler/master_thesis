import re


def dayhoff_6_recode_file(input_file, output_file):
    with open(input_file, 'r') as protAlignment, open(output_file, 'w') as dayhoffRecoded:
        for line in protAlignment:
            if line.startswith(">"):
                dayhoffRecoded.write(line)
            else:
                dayhoffRecoded.write(dayhoff_6_recode(line))


def dayhoff_6_recode(input_string: str) -> str:
    recode = re.sub(r'[ASTGP]', '0', input_string)
    recode = re.sub(r'[DNEQ]', '1', recode)
    recode = re.sub(r'[RKH]', '2', recode)
    recode = re.sub(r'[MVIL]', '3', recode)
    recode = re.sub(r'[FYW]', '4', recode)
    recode = re.sub(r'C', '5', recode)
    return recode


def dayhoff_18_recode(input_string: str) -> str:
    recode = re.sub(r'[ASTGP]', '0', input_string)
    recode = re.sub(r'[DNEQ]', '1', recode)
    recode = re.sub(r'[RKH]', '2', recode)
    recode = re.sub(r'[MVIL]', '3', recode)
    recode = re.sub(r'[FYW]', '4', recode)
    recode = re.sub(r'C', '5', recode)
    return recode
