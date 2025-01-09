from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

# โหลดโมเดลและ tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# ฟังก์ชันแปลงข้อความ
def transform_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return json.loads(decoded_output)

if __name__ == "__main__":
    input_text = "ให้เช่า คอนโด Aspace Asoke-Ratchada ตึก D ชั้น 5 พื้นที่35 ตรม. ค่าเช่า 10,000/เดือน สัญญา 1 ปี"
    print(transform_text(input_text))
