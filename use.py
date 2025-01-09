def preprocess_data(examples):
    inputs = [ex['input'] for ex in examples]
    targets = [json.dumps(ex['output']) for ex in examples]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs['labels'] = labels
    return model_inputs

train_data = train_data.map(preprocess_data, batched=True)
val_data = val_data.map(preprocess_data, batched=True)