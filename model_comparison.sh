python visualize_training_reconstruction.py \
  --compare_models \
  --model1_path /root/workspace/HRI_MLLM/output/motion_adaptor_v6/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1500.pt \
  --model2_path /root/workspace/HRI_MLLM/output/motion_adaptor_v7/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1500.pt \
  --jsonl_path /root/workspace/HRI_MLLM/data/single_motion_sentence_version2_kimi_labeled_tokens_test.jsonl \
  --num_samples 50 \
  --output_dir ./model_comparison_results \
  --temperature 1.0 \
  --top_k 50 \
  --repetition_penalty 1.2