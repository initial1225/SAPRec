from transformers import (
    T5ForConditionalGeneration,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    HammingDiversityLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    BeamSearchScorer,
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn
import torch


class SapModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def init_templates(self, task_num, seg_templates_len, device):
        # Embase size from T5 (usually 512 for t5-small)
        emsize = self.shared.weight.size(1)
        
        self.seg_templates_len = seg_templates_len
        self.model_device = device
        self.seg_templates_num = 3  # 3 segments per task: prefix, middle, suffix
        
        # Initialize prompt and whole-word embeddings
        self.templates_embeddings = nn.Embedding(task_num * self.seg_templates_len * self.seg_templates_num, emsize)
        self.whole_word_embeddings = nn.Embedding(self.config.n_positions, emsize)
        
        # Uniform initialization for stability
        initrange = 0.1
        self.templates_embeddings.weight.data.uniform_(-initrange, initrange)
        
        # Base offset tensor: [0, 1, 2, ... seg_templates_len-1]
        self.templates_offset = torch.arange(self.seg_templates_len, device=self.model_device)

    def input_plus_whole_word(self, input_user_ids, input_item_ids, user_whole_word_ids, item_whole_word_ids):
        # Combine standard text embeddings with whole-word embeddings
        user_emb = self.shared(input_user_ids) + self.whole_word_embeddings(user_whole_word_ids)
        item_emb = self.shared(input_item_ids) + self.whole_word_embeddings(item_whole_word_ids)
        return torch.cat([user_emb, item_emb], dim=1)

    def append_templates(self, task_id, user_input_ids, item_input_ids, user_whole_word_ids, item_whole_word_ids, user_attention_mask, item_attention_mask):
        batch_size = task_id.size(0)
        
        # Use .expand() instead of .repeat() for memory efficiency
        offset = self.templates_offset.unsqueeze(0).expand(batch_size, -1)
        
        # Base index for the current task
        base_idx = (task_id * self.seg_templates_num * self.seg_templates_len).unsqueeze(1)

        # Generate 3 template segments
        t1 = self.templates_embeddings(base_idx + offset)
        t2 = self.templates_embeddings(base_idx + self.seg_templates_len + offset)
        t3 = self.templates_embeddings(base_idx + self.seg_templates_len * 2 + offset)

        # User and item embeddings
        user_emb = self.shared(user_input_ids) + self.whole_word_embeddings(user_whole_word_ids)
        item_emb = self.shared(item_input_ids) + self.whole_word_embeddings(item_whole_word_ids)

        # Concatenate: [T1, User, T2, Item, T3]
        input_emb = torch.cat([t1, user_emb, t2, item_emb, t3], dim=1)

        # --- Block-wise Attention Mask ---
        total_len = self.seg_templates_len * 3 + user_attention_mask.size(1) + item_attention_mask.size(1)
        input_mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.float32, device=self.model_device)

        # Block A: P1 + User (Attends to Block A)
        len_a = self.seg_templates_len + user_attention_mask.size(1)
        input_mask[:, 0:len_a, 0:len_a] = 1

        # Block B: P2 + Item (Attends to Block A + B)
        len_b = self.seg_templates_len + item_attention_mask.size(1)
        end_b = len_a + len_b
        input_mask[:, len_a:end_b, 0:end_b] = 1

        # Block C: P3 (Attends to All: Block A + B + C)
        input_mask[:, end_b:, :] = 1

        return input_emb, input_mask

    def forward(
        self,
        task_id=None,
        input_user_ids=None,
        input_item_ids=None,
        user_whole_word_ids=None,
        item_whole_word_ids=None,
        user_attention_mask=None,
        item_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=True,
        output_hidden_states=None,
        return_dict=True,
    ):
        if encoder_outputs is None:
            if task_id is None:
                input_emb = self.input_plus_whole_word(input_user_ids, input_item_ids, user_whole_word_ids, item_whole_word_ids)
            else:
                input_emb, attention_mask = self.append_templates(
                    task_id, input_user_ids, input_item_ids, user_whole_word_ids, 
                    item_whole_word_ids, user_attention_mask, item_attention_mask
                )
            
            # Pass constructed embeddings through the encoder
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_emb,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return super().forward(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )    

    def beam_search(
        self,
        task_id=None,
        input_user_ids=None,
        input_item_ids=None,
        user_whole_word_ids=None,
        item_whole_word_ids=None,
        user_attention_mask=None,
        item_attention_mask=None,
        max_length=50,
        num_beams=20,
        num_beam_groups=1,
        early_stopping=True,
        min_length=1,
        diversity_penalty=0.0,
        repetition_penalty=1.0,
        num_return_sequences=20,
        bad_words_ids=None,
    ):
        batch_size = input_user_ids.size(0)
        
        # Define decoder start token ids
        decoder_input_ids = torch.ones((num_beams * batch_size, 1), dtype=torch.int64, device=self.model_device)
        decoder_input_ids *= self.config.decoder_start_token_id

        # Prepare encoder inputs
        if task_id is None:
            input_emb = self.input_plus_whole_word(input_user_ids, input_item_ids, user_whole_word_ids, item_whole_word_ids)
        else:
            input_emb, attention_mask = self.append_templates(
                task_id, input_user_ids, input_item_ids, user_whole_word_ids, 
                item_whole_word_ids, user_attention_mask, item_attention_mask
            )
            
        model_kwargs = {
            "encoder_outputs": self.encoder(
                attention_mask=attention_mask.repeat_interleave(num_beams, dim=0) if attention_mask is not None else None,
                inputs_embeds=input_emb.repeat_interleave(num_beams, dim=0),
                return_dict=True,
            )
        }

        # Instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=self.model_device,
            num_beam_groups=num_beam_groups,
            num_beam_hyps_to_keep=num_return_sequences,
            do_early_stopping=early_stopping,
        )

        criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

        # Instantiate logits processors
        logits_processor = LogitsProcessorList([
            MinLengthLogitsProcessor(min_length, eos_token_id=self.config.eos_token_id)
        ])
        
        if bad_words_ids is not None:
            logits_processor.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id=self.config.eos_token_id))

        if num_beam_groups == 1:
            return super().beam_search(
                decoder_input_ids,
                beam_scorer,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                **model_kwargs
            )
        else:
            if diversity_penalty > 0.0:
                logits_processor.append(
                    HammingDiversityLogitsProcessor(
                        diversity_penalty,
                        num_beams=num_beams,
                        num_beam_groups=num_beam_groups,
                    )
                )
            if repetition_penalty != 1.0:
                logits_processor.append(
                    RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
                )

            return super().group_beam_search(
                decoder_input_ids,
                beam_scorer,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                **model_kwargs
            )