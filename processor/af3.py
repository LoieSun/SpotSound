import torch
from typing import Union, Optional
import numpy as np
from transformers.models.audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor, AudioFlamingo3ProcessorKwargs
from transformers.tokenization_utils_base import TextInput
from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.processing_utils import Unpack
from transformers.feature_extraction_utils import BatchFeature


class AudioFlamingo3TemporalProcessor(AudioFlamingo3Processor):

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Optional[AudioInput] = None,
        output_labels: Optional[bool] = False,
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:

        # Merge defaults with user kwargs
        call_kwargs = self._merge_kwargs(
            AudioFlamingo3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        audio_inputs = {}
        if audio is not None:
            audio = make_list_of_audio(audio)
            if len(text) != len(audio):
                raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

            # Determine number of chunks per sample, and flatten
            window_size = int(audio_kwargs["sampling_rate"] * audio_kwargs["chunk_length"])
            max_windows = int(self.max_audio_len // audio_kwargs["chunk_length"])

            per_sample_windows: list[int] = []
            flat_chunks: list[np.ndarray] = []

            for audio_el in audio:
                n_samples = int(audio_el.shape[0])
                n_win = max(1, (n_samples + window_size - 1) // window_size)
                if n_win > max_windows:
                    logger.warning(
                        f"Audio duration ({n_samples / audio_kwargs['sampling_rate']:.1f}s) exceeds {self.max_audio_len}s; truncating to first {self.max_audio_len}s."
                    )
                    n_win = max_windows
                per_sample_windows.append(n_win)

                time_cap = min(n_samples, n_win * window_size)
                for i in range(n_win):
                    start = i * window_size
                    end = min((i + 1) * window_size, time_cap)
                    flat_chunks.append(audio_el[start:end])

            # Feature extraction
            audio_inputs = self.feature_extractor(flat_chunks, **audio_kwargs)
            padding_mask = audio_inputs.pop("attention_mask")
            audio_inputs["input_features_mask"] = padding_mask

            # Compute sequence lengths token counting
            audio_lengths = torch.stack([s.sum() for s in torch.split(padding_mask.sum(-1), per_sample_windows)])
            audio_tokens_lengths = self._get_audio_token_length(audio_lengths)


            # --- Replace the original simple expansion to implement timestamp-interleaved audio token expansion ---
            expanded_text = []
            flat_token_counts = audio_tokens_lengths.tolist()  
            
            for sample in text:
                replace_str = []

                # Loop through all audio tokens in the current sample (usually just one)
                while self.audio_token in sample:

                    # Pop the corresponding number of audio tokens in order
                    num_audio_tokens = flat_token_counts.pop(0)

                    # Calculate the number of timestamp chunks (25 tokens per chunk)
                    audio_duration = num_audio_tokens // 25
                    timestamps = [f"timestamp: {t} seconds; feature: " for t in range(audio_duration)]

                    # Build the expanded string: each timestamp followed by bos + 25 audio tokens + eos
                    expanded_audio_token = ""
                    for ts in timestamps:
                        expanded_audio_token += (
                            ts
                            + self.audio_token * 25
                        )

                    replace_str.append(expanded_audio_token)
                    
                    # Replace the currently found audio token with a placeholder (to avoid repeated replacements)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                # Replace the placeholders with the previously generated expanded strings in order
                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)

                expanded_text.append(sample)

            text = expanded_text

        # Tokenize
        text_inputs = self.tokenizer(text, **text_kwargs)
        data = {**text_inputs, **audio_inputs}


        return BatchFeature(data=data, tensor_type=return_tensors)
