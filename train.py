import torch
import json
import os
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import argparse
from transformers import Trainer, TrainingArguments
from utils import find_all_linear_names
from model.af3 import AudioFlamingo3ForTemporalConditionalGeneration
from processor.af3 import AudioFlamingo3TemporalProcessor


class AudioDataset(Dataset):
    def __init__(self, ds):
        path = ds['path']
        model_name = ds['model_name']
        with open(path) as f:
            self.datas = json.load(f)
        self.prompt = 'This is a sequence of audio stream. Your task is to identify the temporal window (start and end timestamps) when the given query appears. The query is: '
        self.prompt_1 = 'This is a sequence of audio stream. Your task is to identify whether the sound event in the query occurs. The query is: '

        # load tokenizer and processor
        self.processor = AudioFlamingo3TemporalProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        
        # set pad_token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        audio_path = data['audio_path']
        query_prompt = self.prompt + data['caption'] + ' Answer: '

        interval_text = ''
        if isinstance(data['answer'], str):
            if data['answer'] == "Yes." or "No.":
                interval_text = data['answer']
                conversation = [
                        {"role": "user", "content": [
                            {"type": "audio", "path": audio_path},
                            {"type": "text", "text": self.prompt_1 + data['caption'] + ' Answer: '},]},
                        {"role": "assistant", "content": [{"type": "text", "text": f"{interval_text}"}]}
                    ]
            
            else:
                interval_text = data['answer']
                conversation = [
                        {"role": "user", "content": [
                            {"type": "audio", "path": audio_path},
                            {"type": "text", "text": query_prompt},]},
                        {"role": "assistant", "content": [{"type": "text", "text": f"{interval_text}"}]}
                    ]
        else:
            for [start, end] in data['answer']:
                interval_text += str(f"from {start:.2f}s to {end:.2f}s, ")

            interval_text = interval_text[:-2]
            conversation = [
                    {"role": "user", "content": [
                        {"type": "audio", "path": audio_path},
                        {"type": "text", "text": query_prompt},]},
                    {"role": "assistant", "content": [{"type": "text", "text": f"{interval_text}"}]}
                ]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            output_labels=True,
        )
        
        return inputs
     

def main():

    print('starting....')

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default='path/audio-flamingo-3-hf')
    parser.add_argument('--output_str', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--train_data', type=str, default='/train.json')
    parser.add_argument('--val_data', type=str, default='/valid.json')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    model_name = args.pretrain_model

    train_dataset = AudioDataset({'path': args.train_data,'model_name':model_name})
    eval_dataset = AudioDataset({'path': args.val_data,'model_name':model_name})

    train_dataset.__getitem__(0)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")

    # load pretrain model and processor
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_modules = []
    llm_keys = ["language_model"]
    named_modules = {n: m for n, m in model.named_modules()}
    lora_modules.extend(find_all_linear_names(named_modules, llm_keys))

    # configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=2*args.lora_r,
        lora_dropout=0.1,
        target_modules=lora_modules,
        bias="none",
        inference_mode=False,
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    

    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        if 'audio_tower' in name or 'multi_modal_projector' in name:
            param.requires_grad = False
            
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"\ntrainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")

    # set training parameters
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.output_str),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        warmup_steps=1000,
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        bf16=True,  # if GPU supports
        report_to="tensorboard",
        dataloader_num_workers=4,
        max_grad_norm=1.0,  # gradient clipping
        learning_rate=args.lr,  # very small learning rate
    )

    # create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # start training
    trainer.train()
    trainer.save_state()

if __name__ == "__main__":
    main()
