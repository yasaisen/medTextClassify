"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506171442
"""

from medTextClassify.common.utils import set_seed, load_json_data, MetricsTracker
from medTextClassify.trainer.building_BERTTrainer import train_biobert_classifier, load_and_predict


def main():
    set_seed(42)
    model_path = train_biobert_classifier()
    
    if model_path:
        sample_text = "Experiences of Women Who Underwent Predictive BRCA 1/2 Mutation Testing Before the Age of 30."
        result = load_and_predict(model_path, sample_text)
        if result:
            print(f"\nSample Prediction:")
            print(f"Text: {sample_text}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print("Prediction failure")
    else:
        print("Training failed, unable to perform prediction test")


if __name__ == "__main__":
    main()