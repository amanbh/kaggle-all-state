DATA = "lgbm_train.csv"
num_folds  = 5
block_size = 37663

train_files = []
cv_files = []
for i in range(num_folds):
    train_files.append(open("train_" + repr(i) + ".csv", "w"))
    cv_files.append(open("cv_" + repr(i) + ".csv", "w"))

line_id = 0 # Set to -1 if header is present in DATA, else set to 0
cv_id = 0
with open(DATA_FILE, "r") as f:
    for line in f:
        if line_id == -1:
            line_id += 1
            continue
        if line_id == block_size:
            line_id = 0
            cv_id = min(cv_id + 1, num_folds - 1)

        for train_id in range(num_folds):
            if train_id != cv_id:
                train_files[train_id].write(line)
        cv_files[cv_id].write(line)
        line_id += 1

for f in train_files:
    f.close()
for f in cv_files:
    f.close()

