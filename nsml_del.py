import subprocess


team_name = "kozistr_team"
data_sets = ["NER", "SRL"]

src, dst = 200, 600
ner_except_no = [432, 437, 440, 446, 456, 532, 542]
srl_except_no = [75, 78, 84, 112, 226, 230, 233, 342, 347]
assert dst > src > 0

for data_set in data_sets:

    target = team_name + "/" + data_set

    for i in range(src, dst):
        if data_set == "NER" and i in ner_except_no:
            continue

        if data_set == "SRL" and i in srl_except_no:
            continue

        cmd = "nsml model rm %s/%d *" % (target, i)
        print("[*] Do : %s" % cmd)

        data = subprocess.getoutput(cmd)

        if data in "FATA":
            print("%d " % i, data)
