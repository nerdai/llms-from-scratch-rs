# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/),  
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

- Updated candle deps (#382)

## [0.1.4] - 2025-02-27

### Added

- EG 07.23 — example usage of train_model_dpo_simple (#339)
- train_model_dpo_simple (#338)
- evaluate_dpo_loss_loader (#337)
- EG 07.22 (#335)
- compute_dpo_loss_loader (#334)
- EG 07.21 - example usage of compute_dpo_loss_batch (#333)
- dpo_loss_batch (#332)
- compute_logprobs (#331)
- compute_dpo_loss (#326)
- preference data loader (#319)
- [Fix] rejected/chosen masking & update EG 07.19 (#318)
- EG 07.19 Example usage of PreferenceDataCollator (#314)
- PreferenceDatasetCollator (#309)
- PreferenceDataset (#308)
- EncodedPreferenceExample and InstructionExample trait (#303)
- generate_preference_dataset + EG 07.18 (#299)
- Bonus DPO - Use Ollama to generate chosen/rejection response for an instruction entry + EG 07.17 (#294)

### Changed

Exercise 7.4 + use of GPT trait in listings::ch05 instead of GPTModel (#289)

## [0.1.3] - 2025-01-23

### Added

- docs Appendix E (#287)
- EG E.07 (#286)
- EG E.06 (#285)
- Listing E.7 re-export of train_classifier_simple (#284)
- EG E.05 (#283)
- EG E.04 (#282)
- EG E.03 (LoRA model loading) (#281)
- GPTModelWithLoRA (#279)
- TransformerBlockWithLoRA (#278)
- FeedForwardWithLoRA (#277)
- MultiHeadAttentionWithLoRA (#270)
- Listing E.6 LinearWithLoRA (#269)
- Listing E.5 (LoRALayer) (#268)
- EG E.02 and set listing E.4 as re-export (#267)
- Listing E.4 (#266)
- Example E.01 (#265)
- Listing E.3 (#264)
- Listing E.2 (#263)
- Listing E.1 (#262)

### Changed

- parametrize batch size (#285)
- GPT trait to "consolidate" GPTModel and GPTModelWithLoRA (#279)
- Rip out Sequential and SequentialT in favor of explicit sequential-like structs (#276)

## [0.1.2] - 2025-01-13

### Added

- Exercise 7.3 (#252)

### Fixed

- [docs] make listings more visible for ch07 (#260)

## [0.1.1] - 2025-01-11

### Added

- Exercise 7.2 (#248)

### Changed

- Make `listings::ch07::InstructionDataBatcher` more generic (#248)
- Add associated type to `listings::ch07::CustomCollator` (#248)

### Fixed

- Incorrect cast of `keep` indices to `u8` in `calc_loss_loader` (#250)
- Missing `ignore_index` in `calc_loss_loader` fn params (#250)

## [0.1.0] - 2025-01-09

### Added

- Listings ch02 up to ch07
- Examples ch02 up to ch07
- Exercise ch02 up to ch06 and Exercise 7.1
