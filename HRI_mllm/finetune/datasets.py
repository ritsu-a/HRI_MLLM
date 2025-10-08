from torch.utils.data import Dataset
from functools import lru_cache
import torch
from typing import Dict, List
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from HRI_mllm.finetune.data import KimiAMotionContent
import librosa

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data_list, whisper_model, text_tokenizer, max_len: int, kimia_token_offset: int):
        super(LazySupervisedDataset, self).__init__()
        self.whisper_model = whisper_model
        self.max_len = max_len

        print("There are {} samples in the dataset".format(len(raw_data_list)))
        self.whisper_model = whisper_model

        print(f"Loading text tokenizer")
        self.text_tokenizer = text_tokenizer

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)

        self.pad_token = self.extra_tokens.pad
        self.kimia_token_offset = kimia_token_offset
        self.raw_data = []
        
        ### remove too long sequences
        for data in raw_data_list:
            if len(data['conversation'][1]['audio_tokens']) < 1000:
                self.raw_data.append(data)
        print("Loading  {} samples in the dataset".format(len(self.raw_data)))

        # 存储motion_blank用于collate_fn
        self.motion_blank = self.extra_tokens.motion_blank

        ### TODO: segment too long sequence in raw_data

        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def extract_whisper_feat(self, wav: str):
        wav = librosa.load(wav, sr=16000)[0]
        return wav

    def _tokenize_text(self, text):
        if text is None:
            return None
        token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
        return token_ids

    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
        extract_whisper_feature=False,
        output_type: str = "text",
    ):
        kimia_content_msg = KimiAMotionContent()

        role = message["role"]

        has_loss = role == "assistant"

        if tokenize_role:
            if role == "user":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
                kimia_content_msg.motion_append(self.extra_tokens.motion_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_assistant_msg_start
                )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
                kimia_content_msg.motion_append(self.extra_tokens.motion_start)
            else:
                raise NotImplementedError(f"role: {role}")

        if message["message_type"] == "text":
            text = message["content"]
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.text_extend(text_tokens, has_loss)
            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] * len(text_tokens)
            )
            kimia_content_msg.motion_extend(
                [self.extra_tokens.motion_blank] * len(text_tokens)
            )

            if role == "assistant":
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss) # eos for text stream
                kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)
                kimia_content_msg.motion_append(self.extra_tokens.motion_blank, motion_token_loss_mask=False)

        elif message["message_type"] == "audio":
            speech_tokens = message["audio_tokens"]

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=True, audio_token_loss_mask=has_loss)
            kimia_content_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss) # EOS for audio stream
            kimia_content_msg.text_extend(
                [self.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2)
            )
            kimia_content_msg.motion_extend(
                [self.extra_tokens.motion_blank] * (len(speech_tokens) + 2)
            )

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ctd_id
                    )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
                kimia_content_msg.motion_append(self.extra_tokens.motion_blank)


            if extract_whisper_feature:
                whisper_feature = self.extract_whisper_feat(message["content"])
                kimia_content_msg.continuous_feature.append(whisper_feature)

        elif message["message_type"] == "audio_motion":
            speech_tokens = message["audio_tokens"]
            motion_tokens = message["motion_tokens"]

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=False, audio_token_loss_mask=has_loss)
            kimia_content_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss) # EOS for audio stream

            kimia_content_msg.motion_append(self.extra_tokens.motion_start)
            kimia_content_msg.motion_extend(motion_tokens, motion_token_loss_mask=has_loss)
            kimia_content_msg.motion_append(self.extra_tokens.motion_end, motion_token_loss_mask=has_loss) # EOS for motion stream

            kimia_content_msg.text_extend(
                [self.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2)
            )

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ctd_id
                    )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
                kimia_content_msg.motion_append(self.extra_tokens.motion_blank)

        elif message["message_type"] == None:
            pass
        else:
            raise NotImplementedError(f"message_type: {message['message_type']}")

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end, audio_token_loss_mask=False)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            kimia_content_msg.motion_append(self.extra_tokens.motion_blank)
        assert (
            kimia_content_msg.is_valid()
        ), f"kimia_content_msg is not valid: {kimia_content_msg}"

        return kimia_content_msg

    def tokenize_conversation(
        self, messages: List[Dict], output_type: str = "text", add_assistant_start_msg: bool = True
    ) -> KimiAMotionContent:
        """
        messages: List[Dict]
        messages[i] = {
            "role": "user" | "assistant" | "system",
            "content": str
        }
        """
        assert output_type in ["text", "both"]

        msgs: List[KimiAMotionContent] = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False

        previous_role = None
        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]

            if previous_role is None:
                tokenize_role = True
            else:
                if message["role"] == previous_role:
                    tokenize_role = False
                else:
                    tokenize_role = True

            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                if messages[msg_idx + 1]["role"] != message["role"]:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    has_ct_token = False
                    has_msg_end_token = False

            previous_role = message["role"]

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
                extract_whisper_feature=True,
                output_type=output_type,
            )
            msgs.append(msg)

        if add_assistant_start_msg:
            assistant_start_msg = self.tokenize_message(
                    message={
                        "role": "assistant",
                    "message_type": None,
                },
                tokenize_role=True,
                has_ct_token=False,
                has_msg_end_token=False,
            )

            msgs.append(assistant_start_msg)

        ret_msg = msgs[0]

        for msg in msgs[1:]:
            ret_msg.merge(msg)

        return ret_msg

    @lru_cache(maxsize=None)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        task_type = self.raw_data[i]["task_type"]
        conversation = self.raw_data[i]["conversation"]

        output_type = "text" if task_type == "understanding" else "both"

        tokenized_conversation = self.tokenize_conversation(conversation, output_type=output_type, add_assistant_start_msg=False)

        audio_input_ids, motion_input_ids, text_input_ids, is_continuous_mask, audio_token_loss_mask, motion_token_loss_mask, text_token_loss_mask = tokenized_conversation.to_tensor()

        audio_features = tokenized_conversation.continuous_feature

        audio_labels = torch.cat((audio_input_ids[:, 1:], audio_input_ids.new_full((1, 1), self.pad_token)), dim=1)
        motion_labels = torch.cat((motion_input_ids[:, 1:], motion_input_ids.new_full((1, 1), self.extra_tokens.motion_blank)), dim=1)
        text_labels = torch.cat((text_input_ids[:, 1:], text_input_ids.new_full((1, 1), self.pad_token)), dim=1)
        audio_loss_mask = torch.cat((audio_token_loss_mask[:, 1:], audio_token_loss_mask.new_full((1, 1), False)), dim=1)
        motion_loss_mask = torch.cat((motion_token_loss_mask[:, 1:], motion_token_loss_mask.new_full((1, 1), False)), dim=1)
        text_loss_mask = torch.cat((text_token_loss_mask[:, 1:], text_token_loss_mask.new_full((1, 1), False)), dim=1)

        ret = dict(
            input_ids=audio_input_ids,
            motion_input_ids=motion_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=audio_features,
            is_continuous_mask=is_continuous_mask,
            labels=(
                audio_labels,
                motion_labels,
                text_labels,
                audio_loss_mask,
                motion_loss_mask,
                text_loss_mask,
            ),
        )

        return ret

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1, "micro batch size is 1 for demo"

        return batch[0]

    # def collate_fn(self, batch):
    #     """支持更大batch size的collate函数"""
    #     if len(batch) == 0:
    #         return {}
        
    #     # 获取batch中所有样本
    #     batch_dict = {}
        
    #     # 处理普通张量字段
    #     tensor_fields = ['input_ids', 'motion_input_ids', 'text_input_ids', 'is_continuous_mask']
        
    #     for field in tensor_fields:
    #         if field in batch[0]:
    #             # 收集所有样本的该字段
    #             tensors = [item[field] for item in batch]
    #             # 获取最大长度
    #             max_len = max(tensor.shape[1] for tensor in tensors)
                
    #             # 对每个张量进行padding
    #             padded_tensors = []
    #             for tensor in tensors:
    #                 current_len = tensor.shape[1]
    #                 if current_len < max_len:
    #                     # 计算需要padding的长度
    #                     pad_len = max_len - current_len
    #                     if field == 'motion_input_ids':
    #                         # motion使用motion_blank进行padding
    #                         pad_value = self.motion_blank
    #                     else:
    #                         # 其他字段使用pad_token
    #                         pad_value = self.pad_token
                        
    #                     # 进行padding (在序列维度)
    #                     padding = tensor.new_full((tensor.shape[0], pad_len), pad_value)
    #                     padded_tensor = torch.cat([tensor, padding], dim=1)
    #                     padded_tensors.append(padded_tensor)
    #                 else:
    #                     padded_tensors.append(tensor)
                
    #             # 堆叠所有张量
    #             batch_dict[field] = torch.cat(padded_tensors, dim=0)
        
    #     # 处理labels元组
    #     if 'labels' in batch[0]:
    #         # 解构labels元组
    #         audio_labels_list = []
    #         motion_labels_list = []
    #         text_labels_list = []
    #         audio_loss_mask_list = []
    #         motion_loss_mask_list = []
    #         text_loss_mask_list = []
            
    #         for item in batch:
    #             labels = item['labels']
    #             audio_labels_list.append(labels[0])
    #             motion_labels_list.append(labels[1])
    #             text_labels_list.append(labels[2])
    #             audio_loss_mask_list.append(labels[3])
    #             motion_loss_mask_list.append(labels[4])
    #             text_loss_mask_list.append(labels[5])
            
    #         # 对每个labels组件进行padding
    #         labels_components = [
    #             (audio_labels_list, self.pad_token),
    #             (motion_labels_list, self.motion_blank),
    #             (text_labels_list, self.pad_token),
    #             (audio_loss_mask_list, False),
    #             (motion_loss_mask_list, False),
    #             (text_loss_mask_list, False)
    #         ]
            
    #         padded_labels_components = []
            
    #         for component_list, pad_value in labels_components:
    #             max_len = max(tensor.shape[1] for tensor in component_list)
    #             padded_components = []
                
    #             for tensor in component_list:
    #                 current_len = tensor.shape[1]
    #                 if current_len < max_len:
    #                     pad_len = max_len - current_len
    #                     padding = tensor.new_full((tensor.shape[0], pad_len), pad_value)
    #                     padded_tensor = torch.cat([tensor, padding], dim=1)
    #                     padded_components.append(padded_tensor)
    #                 else:
    #                     padded_components.append(tensor)
                
    #             padded_labels_components.append(torch.cat(padded_components, dim=0))
            
    #         # 重新组合labels元组
    #         batch_dict['labels'] = tuple(padded_labels_components)
        
    #     # 处理whisper_input_feature（如果是列表）
    #     if 'whisper_input_feature' in batch[0]:
    #         # 直接收集所有特征，不进行padding（因为可能是变长特征）
    #         whisper_features = [item['whisper_input_feature'] for item in batch]
    #         batch_dict['whisper_input_feature'] = whisper_features
        
    #     return batch_dict
