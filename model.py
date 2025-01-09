from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

# โหลดข้อมูล
dataset = load_dataset('json', data_files='data.json')

# แบ่งข้อมูล train และ validation
train_data = dataset['train'].train_test_split(test_size=0.1)
train_data, val_data = train_data['train'], train_data['test']

# โหลด Tokenizer และ Model
#tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# เตรียมข้อมูลสำหรับโมเดล
def preprocess_data(examples):
    inputs = [ex['input'] for ex in examples]
    targets = [json.dumps(ex['output']) for ex in examples]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs['labels'] = labels
    return model_inputs

train_data = train_data.map(preprocess_data, batched=True)
val_data = val_data.map(preprocess_data, batched=True)

# ฝึกโมเดล
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
)

if __name__ == "__main__":
    trainer.train()
