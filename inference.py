import argparse
from model.af3 import AudioFlamingo3ForTemporalConditionalGeneration
from processor.af3 import AudioFlamingo3TemporalProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default='audio-flamingo-3-hf')
    parser.add_argument('--checkpoint', type=str, default='spotsound_path')
    parser.add_argument('--audio_path', type=str, default='audio_path.wav')
    parser.add_argument('--query', type=str, default='dog barking')
    args = parser.parse_args()

    # load processor and model
    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.pretrain_model)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(args.checkpoint, device_map="auto")

    # prepare input
    wav_path = args.audio_path
    phrase = args.query
    prompt = 'This is a sequence of audio stream. Your task is to identify the temporal window (start and end timestamps) when the given query appears. The query is: '
    query_prompt = prompt + phrase + ' Answer: '

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query_prompt},
                {"type": "audio", "path": wav_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device).to(model.dtype)

    # inference
    outputs = model.generate(**inputs, max_new_tokens=500)
    response = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)

    
if __name__ == '__main__':
    main()