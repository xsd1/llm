import json
import random

with open("/data/xsd/data/train/train_4.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    random.shuffle(data)
    # train = data[0:-1000]
    # test = data[-1000:]
    # with open("./data/prompt4/data_train_38839.json", "w", encoding="utf-8") as f1:
    #     json.dump(train, f1, indent=4, ensure_ascii=False)
    # with open("./data_test_1000.json", "w", encoding="utf-8") as f1:
    #     json.dump(test, f1, indent=4, ensure_ascii=False)
    train = data[0:20000]
    with open("./dataset/prompt4/data_train_20000.json", "w", encoding="utf-8") as f1:
        json.dump(train, f1, indent=4, ensure_ascii=False)
