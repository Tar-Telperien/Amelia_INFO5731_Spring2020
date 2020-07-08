import random
from sys import stdin

#create mixed training data from a source and target file, with a percentage value to determine how much of the data is from the source file
def pseudo_bilingual(source, target, perc):
    dest = open("new_data.trn", "w")
    counter = 0
    for lang in source:
        s = open(lang + ".trn")
        s_lines = [line.strip() for line in lang]
        s_len = len(s_lines)
        for ln in s_lines:
            if random.randint(1, 11) < perc[counter]/10:
                dest.write(line)
        s.close()
        counter += 1

    for lang in target:
        t = open(lang + ".trn")
        t_lines = [line.strip() for line in t]
        t_len = len(t_lines)
        for ln in t_lines:
            if random.randint(1, 11) < perc[counter]/10:
                dest.write(line)
        t.close()
        counter += 1

    dest.close()

def main():
    print("Please enter the abbreviations for your chosen source and target languages and the desired percentages of each, in that order. For the source language, enter eng for English, swe for Swedish, isl for Icelandic, or deu for German. For the target language, enter nno for Norwegian Nynorsk or gml for Middle Low German. For the percentage, enter an integer between 1 and 100 for each language entered above. Separate all values with spaces, not commas. Source: ")
    sorc = stdin.readline().rstrip().split()
    print("Target: ")
    targ = stdin.readline().rstrip().split()
    print("Percent: ")
    perc = map(int, stdin.readline().strip().split())

    pseudo_bilingual(sorc, targ, perc)

if __name__ == "__main__":
    main()

