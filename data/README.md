# Data Folder

Put the preprocessed MIT-BIH heartbeat CSV files here:

```text
data/processed/mitbih_train.csv
data/processed/mitbih_test.csv
```

The expected format is one heartbeat per row:

```text
sample_0, sample_1, ..., sample_186, label
```

For the first baseline, use the Kaggle "Heartbeat Categorization Dataset" CSV files:

- `mitbih_train.csv`
- `mitbih_test.csv`

Do not commit the CSV files to Git. They are ignored because they are generated/downloaded data.
