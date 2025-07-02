import glob
import os
import shutil
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from cehrbert.data_generators.hf_data_generator.meds_utils import CacheFileCollector
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer_utils import is_main_process
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_finetuning_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import (
    CehrGptDataCollator,
    SamplePackingCehrGptDataCollator,
)
from cehrgpt.data.sample_packing_sampler import SamplePackingBatchSampler
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPT2Model,
    extract_features_from_packed_sequence,
)
from cehrgpt.models.special_tokens import LINEAR_PROB_TOKEN
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.data_utils import prepare_finetune_dataset
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import tokenizer_exists
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import get_torch_dtype
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
)
from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_pretraining_dataset
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_from_disk


LOG = logging.get_logger("transformers")

if __name__ == "__main__":
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path
    )
    torch_dtype = get_torch_dtype(model_args.torch_dtype)
    cehrgpt_model = (
        CEHRGPT2Model.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
            torch_dtype=torch_dtype,
        )
        .eval()
        .to(device)
    )

    dataset = load_parquet_as_dataset(
                    os.path.expanduser(data_args.data_folder),
                    split="train",
                    streaming=data_args.streaming,
                )

    # dataset = dataset.select(range(1000))

    dataset = dataset.train_test_split(
                        test_size=data_args.validation_split_percentage,
                        seed=training_args.seed,
                    )

    dataset = create_cehrgpt_pretraining_dataset(
        dataset,
        cehrgpt_tokenizer,
        data_args,
        cache_file_collector=None
    )
    # dataset = DatasetDict(
    #     {"train": dataset["train"], "validation": dataset["test"]}
    # )
    train_set = dataset["train"]
    processed_dataset = dataset

    per_device_eval_batch_size = 1
    data_collator_fn = partial(
        SamplePackingCehrGptDataCollator,
        cehrgpt_args.max_tokens_per_batch,
        cehrgpt_model.config.max_position_embeddings,
        add_end_token_in_sample_packing=cehrgpt_args.add_end_token_in_sample_packing,
    )
    train_batch_sampler = SamplePackingBatchSampler(
        lengths=train_set["num_of_concepts"],
        max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
        max_position_embeddings=cehrgpt_model.config.max_position_embeddings,
        drop_last=training_args.dataloader_drop_last,
        seed=training_args.seed,
    )
    test_batch_sampler = SamplePackingBatchSampler(
        lengths=processed_dataset["test"]["num_of_concepts"],
        max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
        max_position_embeddings=cehrgpt_model.config.max_position_embeddings,
        drop_last=training_args.dataloader_drop_last,
        seed=training_args.seed,
    )
    data_collator = data_collator_fn(
        tokenizer=cehrgpt_tokenizer,
        max_length=(
            cehrgpt_args.max_tokens_per_batch
            if cehrgpt_args.sample_packing
            else model_args.max_position_embeddings
        ),
        include_values=cehrgpt_model.config.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=cehrgpt_args.include_demographics,
        add_linear_prob_token=True,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=train_batch_sampler,
    )

    test_dataloader = DataLoader(
        dataset=processed_dataset["test"],
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=test_batch_sampler,
    )

    # print("Loading demographics as a dictionary")
    # demographics_df = pd.concat(
    #     [
    #         pd.read_parquet(
    #             data_dir,
    #             columns=[
    #                 "person_id",
    #                 "index_date",
    #                 "gender_concept_id",
    #                 "race_concept_id",
    #             ],
    #         )
    #         for data_dir in [data_args.data_folder, data_args.test_data_folder]
    #     ]
    # )
    # # This is a pre-caution in case the index_date is not a datetime type
    # demographics_df["index_date"] = pd.to_datetime(
    #     demographics_df["index_date"]
    # ).dt.date
    # demographics_dict = {
    #     (row["person_id"], row["index_date"]): {
    #         "gender_concept_id": row["gender_concept_id"],
    #         "race_concept_id": row["race_concept_id"],
    #     }
    #     for _, row in demographics_df.iterrows()
    # }

    ve_token_id = cehrgpt_tokenizer._convert_token_to_id("[VE]")
    for split, data_loader in [("train", train_loader), ("test", test_dataloader)]:
        # Ensure prediction folder exists
        feature_output_folder = (
            Path(training_args.output_dir) / "features_without_label" / f"{split}_features"
        )
        feature_output_folder.mkdir(parents=True, exist_ok=True)

        LOG.info("Generating features for %s set at %s", split, feature_output_folder)

        with torch.no_grad():
            for index, batch in enumerate(
                tqdm(data_loader, desc="Generating features")
            ):

                # person_ids = batch.pop("person_id").numpy().astype(int).squeeze()
                # if person_ids.ndim == 0:
                #     person_ids = np.asarray([person_ids])

                batch = {k: v.to(device) for k, v in batch.items()}
                # Forward pass
                cehrgpt_output = cehrgpt_model(
                    **batch, output_attentions=False, output_hidden_states=False
                )
                
                if cehrgpt_args.sample_packing:
                    if cehrgpt_args.average_over_sequence:
                        ve_token_indicators: torch.BoolTensor = (
                            batch["input_ids"] == ve_token_id
                        )
                        features = (
                            extract_averaged_embeddings_from_packed_sequence(
                                cehrgpt_output.last_hidden_state,
                                batch["attention_mask"],
                                ve_token_indicators,
                            )
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                        )
                    else:
                        features = (
                            extract_features_from_packed_sequence(
                                cehrgpt_output.last_hidden_state,
                                batch["attention_mask"],
                            )
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                            .squeeze(axis=0)
                        )
                else:
                    if cehrgpt_args.average_over_sequence:
                        features = torch.where(
                            batch["attention_mask"].unsqueeze(dim=-1).to(torch.bool),
                            cehrgpt_output.last_hidden_state,
                            0,
                        )
                        # Average across the sequence
                        features = features.mean(dim=1)
                    else:
                        last_end_token = any(
                            [
                                cehrgpt_tokenizer.end_token_id == input_id
                                for input_id in batch.pop("input_ids")
                                .cpu()
                                .numpy()
                                .squeeze()
                                .tolist()
                            ]
                        )
                        last_token_index = -2 if last_end_token else -1
                        LOG.debug(
                            "The last token is [END], we need to use the token index before that: %s",
                            last_token_index,
                        )
                        features = (
                            cehrgpt_output.last_hidden_state[..., last_token_index, :]
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                        )

                # Flatten features or handle them as a list of arrays (one array per row)
                features_list = [feature for feature in features]
                # race_concept_ids = []
                # gender_concept_ids = []
                # for person_id, index_date in zip(person_ids, prediction_time):
                #     key = (person_id, index_date.date())
                #     if key in demographics_dict:
                #         demographics = demographics_dict[key]
                #         # gender_concept_ids.append(demographics["gender_concept_id"])
                #         race_concept_ids.append(demographics["race_concept_id"])
                #     else:
                #         gender_concept_ids.append(0)
                #         race_concept_ids.append(0)

                features_pd = pd.DataFrame(
                    # {
                    #     "subject_id": person_ids,
                    # }
                )
                # Adding features as a separate column where each row contains a feature array
                features_pd["features"] = features_list
                # features_pd["race_concept_id"] = race_concept_ids
                # features_pd["gender_concept_id"] = gender_concept_ids
                features_pd.to_parquet(
                    feature_output_folder / f"{uuid.uuid4()}.parquet"
                )
