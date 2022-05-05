from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score,classification_report

def calculate_score(gold_file, pred_file):
    gold = open(gold_file, 'r', encoding='utf-8')
    gold = gold.readlines()

    pred = open(pred_file, 'r', encoding='utf-8')
    pred = pred.readlines()

    gold_tags = []
    gold_words = []
    tmp_tags = []
    tmp_words = []
    for line in gold:
        if len(line.strip()) != 0:
            word,tag = line.split()
            tmp_words.append(word)
            if tag == "B-Target":
                tmp_tags.append('B')
            elif tag == "I-Target":
                tmp_tags.append('I')
            else:
                tmp_tags.append('O')
        if line.strip() == "":
            gold_tags.append(tmp_tags)
            gold_words.append(tmp_words)
            tmp_tags = []
            tmp_words = []

    sents = []
    spans = []
    tags = []
    for i,line in enumerate(pred):
        parts = line.split('\t')
        sents.append((parts[0], len(parts[0].split(" "))))
    #     tags.append(['O']*len(texts[i]))
    #     if sents[-1][1] > len(texts[i]):
    #                 err += 1
        tags.append(['O']*sents[-1][1])
        spans.append(parts[1:-1])

    count = 0
    for ind,i in enumerate(spans):
        for j in i:
            k = j.split(":: ")
            sp = k[1].split(",")
            start,end = int(sp[0]), int(sp[1])
            if k[2]==k[3]:
                tags[ind][start] = "B-"+k[2]
                for t in range(start+1,end):
                    tags[ind][t] = "I-"+k[2]
            elif k[2] != k[3]:
                count += 1
                if k[2]=="O":
                    count += 1
                    tags[ind][start] = "B-"+k[3]
                    for t in range(start+1,end):
                        tags[ind][t] = "I-"+k[3]
                elif k[3]=="O":
                    tags[ind][start] = "B-"+k[2]
                    for t in range(start+1,end):
                        tags[ind][t] = "I-"+k[2]
                else:
                    tags[ind][start] = "B-"+k[2]
                    for t in range(start+1,end):
                        tags[ind][t] = "I-"+k[2]

    #Converting the BIO to IO
    pred_tags = []
    for i in tags:
    #     if len(i) <= 5:
            for tag in i:
                if tag != "O":
                    pred_tags.append('I')
                else:
                    pred_tags.append('O')

    gold_tags_flat = []
    for i in gold_tags:
    #     if len(i) <= 5:
            for j in i:
                if j != 'O':
                    gold_tags_flat.append('I')
                else:
                    gold_tags_flat.append('O')

    classification_report_dict = classification_report(
                gold_tags_flat,
                pred_tags,
                zero_division=0,
                output_dict=True,
            )

    print(classification_report_dict)
    
    
calculate_score(gold_file, pred_file)
