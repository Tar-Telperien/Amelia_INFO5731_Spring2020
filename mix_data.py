import random
from sys import stdin

#create mixed training data from a source and target file, with a percentage value to determine how much of the data is from the source file
def pseudo_bilingual(source, target, perc):
    s = open(source)
    s_lines = [line.strip() for line in s]
    s_len = len(s_lines)
    s.close()

    t = open(target)
    t_lines = [line.strip() for line in t]
    t_len = len(t_lines)
    t.close()

    dest = open("new_data.trn", "w")
    for ln in s_lines:
        rand = random.randint(1, 11)
        if rand <= perc/10:
            dest.write(ln)
        else:
            rand2 = random.randint(1, t_len)
            dest.write(t_lines[rand2])
    dest.close()

def main():
    print("Please enter the abbreviations for your chosen source and target language and the desired percentage of the source language in your data, in that order. For the source language, enter eng for English, swe for Swedish, isl for Icelandic, or deu for German. For the target language, enter nno for Norwegian Nynorsk or gml for Middle Low German. For the percentage, enter an integer between 1 and 100. Source: ")
    sorc = stdin.readline().rstrip() + ".trn"
    print("Target: ")
    targ = stdin.readline().rstrip() + ".trn"
    print("Percent: ")
    perc = int(stdin.readline())

    pseudo_bilingual(sorc, targ, perc)

if __name__ == "__main__":
    main()

